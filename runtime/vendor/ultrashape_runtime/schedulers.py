"""Scheduler placeholders for the vendored UltraShape runtime subset."""

from __future__ import annotations

from .utils.tensors import clamp_unit


def default_scheduler_name() -> str:
    return 'flow-matching'


def build_flow_matching_schedule(*, steps: int, guidance_scale: float) -> dict[str, object]:
    timesteps = [round(index / max(steps, 1), 6) for index in range(steps)]
    return {
        'family': default_scheduler_name(),
        'step_count': steps,
        'guidance_scale': guidance_scale,
        'timesteps': timesteps,
        'sigma_start': clamp_unit(1.0 - (1.0 / max(steps, 1))),
        'sigma_end': clamp_unit(1.0 / max(steps, 1)),
    }
