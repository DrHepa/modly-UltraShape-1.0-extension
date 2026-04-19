"""Patched DiT mask seam with optional flash-attn fallback."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature
from .moe_layers import moe_enabled, route_denoise_experts, voxel_cond_signal

try:
    import flash_attn  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on ARM64 MVP installs
    flash_attn = None


def flash_attn_available() -> bool:
    return flash_attn is not None


class RefineDiT:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state
        self.state_dict: dict[str, object] | None = None
        self.hydration: dict[str, object] | None = None
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        self.state_dict = dict(state_dict)
        self.hydrated = True
        return {'missing_keys': [], 'unexpected_keys': [], 'strict': strict}

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
        seed_value = 0 if seed is None else seed
        attention = 'flash_attn' if flash_attn_available() else 'sdpa'
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_values = checkpoint_tokens(checkpoint_reference)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        current_latents = [float(value) for value in latents]
        per_step_signatures: list[float] = []

        if not current_latents:
            current_latents = [0.0 for _ in range(max(len(context), 1))]

        for step_index, timestep in enumerate(timesteps):
            step_fraction = float(step_index + 1) / float(max(len(timesteps), 1))
            expert_tokens = route_denoise_experts(
                latents=current_latents,
                context=context,
                context_mask=context_mask,
                voxel_cond=voxel_cond,
                checkpoint_signal=checkpoint_values,
                timestep=float(timestep),
            )
            next_latents: list[float] = []
            for index, latent in enumerate(current_latents):
                context_value = context[index % len(context)] if context else 0.0
                mask_value = context_mask[index % len(context_mask)] if context_mask else 1.0
                checkpoint_value = checkpoint_values[index % len(checkpoint_values)] if checkpoint_values else 0.0
                expert_value = expert_tokens[index % len(expert_tokens)] if expert_tokens else 0.0
                updated = clamp_unit(
                    (latent * 0.37)
                    + (context_value * 0.27 * mask_value)
                    + (float(timestep) * 0.03)
                    + (step_fraction * 0.11)
                    + (guidance_scale / 80.0)
                    + (((seed_value + step_index + index) % 13) / 200.0)
                    + (checkpoint_value * 0.12)
                    + (expert_value * 0.14)
                    + ((checkpoint_state_signature % 29) / 1000.0)
                )
                next_latents.append(updated)
            current_latents = next_latents
            per_step_signatures.append(float(stable_signature(current_latents)))

        timestep_scale = float(len(timesteps)) / float(len(timesteps) + 40 if timesteps else 40)
        current_latents = [clamp_unit((latent * 0.72) + (timestep_scale * 0.28)) for latent in current_latents]

        context_signature = stable_signature(context)
        context_mask_signature = stable_signature(context_mask)
        voxel_signature = stable_signature(voxel_cond_signal(voxel_cond))
        input_latent_signature = stable_signature(latents)

        return {
            'model': self.__class__.__name__,
            'attention': attention,
            'latents': current_latents,
            'latent_count': len(current_latents),
            'latent_mean': clamp_unit(sum(current_latents) / len(current_latents) if current_latents else 0.0),
            'latent_signature': stable_signature(current_latents),
            'checkpoint_signature': checkpoint_state_signature,
            'inputs': {
                'latents': {
                    'count': len(latents),
                    'signature': input_latent_signature,
                },
                'timestep': {
                    'count': len(timesteps),
                    'signature': stable_signature(timesteps),
                },
                'context': {
                    'count': len(context),
                    'signature': context_signature,
                },
                'context_mask': {
                    'count': len(context_mask),
                    'signature': context_mask_signature,
                },
                'voxel_cond': {
                    'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else 0,
                    'signature': voxel_signature,
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
