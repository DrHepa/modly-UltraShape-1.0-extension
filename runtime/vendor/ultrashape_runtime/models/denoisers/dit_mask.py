"""Portable upstream-shaped DiT subset for the vendored runtime."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_parameter_map, checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature
from .moe_layers import moe_enabled, voxel_cond_signal

try:
    import flash_attn  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on MVP installs
    flash_attn = None


def flash_attn_available() -> bool:
    return flash_attn is not None


def _group_mean(values: list[float], groups: int) -> list[float]:
    if groups <= 0:
        return []
    if not values:
        return [0.0] * groups

    grouped: list[float] = []
    for group_index in range(groups):
        bucket = [value for index, value in enumerate(values) if index % groups == group_index]
        grouped.append(clamp_unit(sum(bucket) / len(bucket) if bucket else 0.0))
    return grouped


def _chunk_tokens(values: list[float], width: int = 4) -> list[list[float]]:
    if not values:
        return [[0.0 for _ in range(width)]]
    tokens: list[list[float]] = []
    for offset in range(0, len(values), width):
        token = [float(value) for value in values[offset : offset + width]]
        if len(token) < width:
            token.extend([0.0] * (width - len(token)))
        tokens.append(token)
    return tokens


def _mean_token(tokens: list[list[float]]) -> list[float]:
    if not tokens:
        return [0.0, 0.0, 0.0, 0.0]
    width = max(len(token) for token in tokens)
    means: list[float] = []
    for index in range(width):
        values = [float(token[index]) for token in tokens if index < len(token)]
        means.append(clamp_unit(sum(values) / len(values) if values else 0.0))
    return means[:4] if len(means) >= 4 else means + [0.0] * (4 - len(means))


def _timestep_embedding(timestep: float) -> list[float]:
    normalized = clamp_unit(abs(float(timestep)) / max(abs(float(timestep)), 1.0))
    return [normalized, clamp_unit(1.0 - normalized), clamp_unit((normalized * 0.5) + 0.25), clamp_unit((normalized * 0.25) + 0.5)]


def _guidance_scalar(candidate: object) -> float:
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, list):
        values = [float(value) for value in candidate if isinstance(value, (int, float))]
        if values:
            return sum(values) / len(values)
    return 0.0


def _hydrate_module_family(
    state_dict: dict[str, object],
    *,
    allowed_roots: tuple[str, ...],
    strict: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    recognized: dict[str, object] = {}
    unexpected_keys: list[str] = []
    for parameter_name, parameter_value in state_dict.items():
        if any(parameter_name == root or parameter_name.startswith(f'{root}.') for root in allowed_roots):
            recognized[parameter_name] = parameter_value
        else:
            unexpected_keys.append(parameter_name)

    missing_roots = [root for root in allowed_roots if not any(name == root or name.startswith(f'{root}.') for name in recognized)]
    legacy_keys = [name for name in state_dict if name == 'tensors' or name.startswith('tensors.')]
    if legacy_keys and not recognized and all(name in legacy_keys for name in state_dict):
        return dict(state_dict), [], []
    if strict and (unexpected_keys or missing_roots or not recognized):
        raise ValueError(
            'RefineDiT strict hydration requires upstream module-family keys for '
            f'{allowed_roots}; missing={missing_roots}, unexpected={unexpected_keys}.'
        )
    return recognized, missing_roots, unexpected_keys


class TimestepEmbedder:
    def __call__(self, timesteps: list[float], guidance=None) -> list[list[float]]:
        del guidance
        return [_timestep_embedding(value) for value in timesteps]


class RefineDiT:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state
        self.t_embedder = TimestepEmbedder()
        self.state_dict: dict[str, object] | None = None
        self.hydration: dict[str, object] | None = None
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        normalized_state = checkpoint_parameter_map({'state_dict': state_dict}) or dict(state_dict)
        hydrated_state, missing_roots, unexpected_keys = _hydrate_module_family(
            normalized_state,
            allowed_roots=('x_embedder', 't_embedder', 'final_layer'),
            strict=strict,
        )
        self.checkpoint_state = {
            'state_dict': hydrated_state,
            'state_dict_metadata': {
                'parameter_count': len(hydrated_state),
                'module_roots': ['final_layer', 't_embedder', 'x_embedder'],
                'module_family': self.__class__.__name__,
            },
            'representation': 'module-state-dict-v2',
        }
        self.state_dict = hydrated_state
        self.hydrated = True
        return {'missing_keys': missing_roots, 'unexpected_keys': unexpected_keys, 'strict': strict}

    def forward(self, x, t, contexts, **kwargs) -> list[list[float]]:
        token_rows = [list(map(float, row)) for row in x if isinstance(row, list)] or [[0.0, 0.0, 0.0, 0.0]]
        timesteps = [float(value) for value in t if isinstance(value, (int, float))] or [0.0]
        main_context = contexts.get('main') if isinstance(contexts, dict) and isinstance(contexts.get('main'), list) else []
        context_tokens = [list(map(float, row)) for row in main_context if isinstance(row, list)] or [[0.0, 0.0, 0.0, 0.0]]
        mask_values = contexts.get('main_mask') if isinstance(contexts, dict) and isinstance(contexts.get('main_mask'), list) else []
        context_mask = [clamp_unit(float(value)) for value in mask_values if isinstance(value, (int, float))] or [1.0] * len(context_tokens)
        voxel_cond = kwargs.get('voxel_cond') if isinstance(kwargs.get('voxel_cond'), dict) else {}
        guidance_cond = _guidance_scalar(kwargs.get('guidance_cond'))
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_profile = _group_mean(checkpoint_tokens(checkpoint_reference, limit=16), 4)
        voxel_profile = _group_mean(voxel_cond_signal(voxel_cond, limit=16), 4)
        context_profile = _mean_token(context_tokens)
        mask_profile = _group_mean(context_mask, 4)
        timestep_profile = _mean_token(self.t_embedder(timesteps, kwargs.get('guidance_cond')))
        guidance_ratio = clamp_unit(guidance_cond / 10.0) if guidance_cond > 0 else 0.0

        output_tokens: list[list[float]] = []
        for token_index, token in enumerate(token_rows):
            context_token = context_tokens[token_index % len(context_tokens)] if context_tokens else [0.0, 0.0, 0.0, 0.0]
            output_tokens.append(
                [
                    clamp_unit(
                        (float(token[channel]) * 0.45)
                        + (context_token[channel % len(context_token)] * (0.2 + (guidance_ratio * 0.2)))
                        + (context_profile[channel] * 0.1)
                        + (voxel_profile[channel] * 0.1)
                        + (checkpoint_profile[channel] * 0.1)
                        + (timestep_profile[channel] * 0.05)
                        + (mask_profile[channel] * 0.05)
                    )
                    for channel in range(min(len(token), 4))
                ]
            )
        return output_tokens

    def denoise(
        self,
        *,
        latents: list[float],
        timesteps: list[float],
        context: list[float],
        context_mask: list[float],
        voxel_cond: dict[str, object],
        guidance_scale: float,
        schedule: dict[str, object],
        seed: int | None,
    ) -> dict[str, object]:
        del seed
        attention = 'flash_attn' if flash_attn_available() else 'sdpa'
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        current_tokens = _chunk_tokens([float(value) for value in latents])
        conditional_contexts = {'main': _chunk_tokens([float(value) for value in context]), 'main_mask': _group_mean([float(value) for value in context_mask], max(len(context) // 4, 1))}
        unconditional_contexts = {'main': [[0.0, 0.0, 0.0, 0.0] for _ in conditional_contexts['main']], 'main_mask': list(conditional_contexts['main_mask'])}
        per_step_signatures: list[float] = []

        for timestep in timesteps:
            conditional_tokens = self.forward(
                current_tokens,
                [float(timestep)],
                conditional_contexts,
                voxel_cond=voxel_cond,
                guidance_cond=[float(guidance_scale)],
            )
            unconditional_tokens = self.forward(
                current_tokens,
                [float(timestep)],
                unconditional_contexts,
                voxel_cond=voxel_cond,
                guidance_cond=[0.0],
            )
            current_tokens = [
                [
                    clamp_unit(
                        unconditional_tokens[token_index][channel]
                        + ((conditional_tokens[token_index][channel] - unconditional_tokens[token_index][channel]) * clamp_unit(float(guidance_scale) / 12.0))
                    )
                    for channel in range(len(conditional_tokens[token_index]))
                ]
                for token_index in range(len(conditional_tokens))
            ]
            per_step_signatures.append(float(stable_signature([value for token in current_tokens for value in token])))

        current_latents = [value for token in current_tokens for value in token]
        return {
            'model': self.__class__.__name__,
            'attention': attention,
            'latents': current_latents,
            'latent_count': len(current_latents),
            'latent_mean': clamp_unit(sum(current_latents) / len(current_latents) if current_latents else 0.0),
            'latent_signature': stable_signature(current_latents),
            'checkpoint_signature': checkpoint_state_signature,
            'inputs': {
                'latents': {'count': len(latents), 'signature': stable_signature([float(value) for value in latents])},
                'timestep': {'count': len(timesteps), 'signature': stable_signature([float(value) for value in timesteps])},
                'context': {'count': len(context), 'signature': stable_signature([float(value) for value in context])},
                'context_mask': {'count': len(context_mask), 'signature': stable_signature([float(value) for value in context_mask])},
                'voxel_cond': {
                    'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else 0,
                    'signature': stable_signature(voxel_cond_signal(voxel_cond)),
                },
            },
            'per_step_signatures': per_step_signatures,
            'schedule_object_type': str(schedule.get('object_type', '')),
            'timestep_count': len(timesteps),
            'moe_enabled': moe_enabled(),
            'state_hydrated': self.hydrated,
            'hydration': dict(self.hydration) if isinstance(self.hydration, dict) else None,
        }


def denoise_conditioned_latents(
    *,
    latents: list[float],
    timesteps: list[float],
    context: list[float],
    context_mask: list[float],
    voxel_cond: dict[str, object],
    guidance_scale: float,
    schedule: dict[str, object],
    checkpoint_state: object = None,
    seed: int | None,
) -> dict[str, object]:
    return RefineDiT(checkpoint_state=checkpoint_state).denoise(
        latents=latents,
        timesteps=timesteps,
        context=context,
        context_mask=context_mask,
        voxel_cond=voxel_cond,
        guidance_scale=guidance_scale,
        schedule=schedule,
        seed=seed,
    )
