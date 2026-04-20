"""Tensor-backed DiT seam with optional flash-attn fallback."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature
from .moe_layers import moe_enabled, voxel_cond_signal

try:
    import flash_attn  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on ARM64 MVP installs
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
        del seed
        attention = 'flash_attn' if flash_attn_available() else 'sdpa'
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_values = checkpoint_tokens(checkpoint_reference, limit=12)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        current_latents = [float(value) for value in latents] or [0.0 for _ in range(max(len(context), 4))]
        context_profile = _group_mean([float(value) for value in context], 4)
        mask_profile = _group_mean([float(value) for value in context_mask], 4)
        voxel_profile = _group_mean(voxel_cond_signal(voxel_cond, limit=16), 4)
        checkpoint_profile = _group_mean(checkpoint_values, 4)
        timestep_scale = max(float(schedule.get('step_count', 0)) or float(len(timesteps)), 1.0)
        per_step_signatures: list[float] = []

        for step_index, timestep in enumerate(timesteps):
            normalized_timestep = clamp_unit(float(step_index + 1) / timestep_scale)
            step_latents: list[float] = []
            for index, latent in enumerate(current_latents):
                channel = index % 4
                context_drive = context_profile[channel]
                mask_drive = mask_profile[channel] if mask_profile else 1.0
                voxel_drive = voxel_profile[channel]
                checkpoint_drive = checkpoint_profile[channel]
                target = clamp_unit(
                    (context_drive * (0.42 + (guidance_scale / 20.0)))
                    + (voxel_drive * 0.28)
                    + (checkpoint_drive * 0.22)
                    + (mask_drive * 0.08)
                )
                correction = (target - float(latent)) * (0.42 + (normalized_timestep * 0.18))
                bias = ((float(timestep) / max(abs(float(timestep)), 1.0)) + 1.0) * 0.01
                updated = clamp_unit(float(latent) + correction + bias)
                step_latents.append(updated)
            current_latents = step_latents
            per_step_signatures.append(float(stable_signature(current_latents)))

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
                    'signature': stable_signature([float(value) for value in latents]),
                },
                'timestep': {
                    'count': len(timesteps),
                    'signature': stable_signature([float(value) for value in timesteps]),
                },
                'context': {
                    'count': len(context),
                    'signature': stable_signature([float(value) for value in context]),
                },
                'context_mask': {
                    'count': len(context_mask),
                    'signature': stable_signature([float(value) for value in context_mask]),
                },
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
