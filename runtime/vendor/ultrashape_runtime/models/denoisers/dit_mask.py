"""Patched DiT mask seam with optional flash-attn fallback."""

from __future__ import annotations

import json

from ...utils.tensors import clamp_unit, stable_signature

try:
    import flash_attn  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on ARM64 MVP installs
    flash_attn = None


def flash_attn_available() -> bool:
    return flash_attn is not None


def denoise_conditioned_latents(
    *,
    conditioning: dict[str, object],
    schedule: dict[str, object],
    checkpoint_state: object = None,
    seed: int | None,
) -> dict[str, object]:
    conditioning_tokens = conditioning.get('tokens') if isinstance(conditioning.get('tokens'), list) else []
    timesteps = schedule.get('timesteps') if isinstance(schedule.get('timesteps'), list) else []
    seed_value = 0 if seed is None else seed
    attention = 'flash_attn' if flash_attn_available() else 'sdpa'
    checkpoint_tensors = checkpoint_state.get('tensors') if isinstance(checkpoint_state, dict) else {}
    checkpoint_values: list[float] = []
    if isinstance(checkpoint_tensors, dict):
        for values in checkpoint_tensors.values():
            if isinstance(values, list):
                checkpoint_values.extend(float(value) for value in values if isinstance(value, (int, float)))
    checkpoint_signature = stable_signature(checkpoint_values)
    latents: list[float] = []

    for index, token in enumerate(conditioning_tokens[:8]):
        schedule_value = timesteps[index % len(timesteps)] if timesteps else 0.0
        checkpoint_value = checkpoint_values[index % len(checkpoint_values)] if checkpoint_values else 0.0
        latent = clamp_unit(
            (token * 0.5)
            + (schedule_value * 0.3)
            + (((seed_value + index) % 11) / 100.0)
            + (checkpoint_value * 0.25)
            + ((checkpoint_signature % 13) / 200.0)
        )
        latents.append(latent)

    return {
        'attention': attention,
        'latents': latents,
        'latent_signature': stable_signature(latents),
    }
