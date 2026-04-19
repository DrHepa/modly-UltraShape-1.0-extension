"""Scheduler seam for the vendored UltraShape runtime subset."""

from __future__ import annotations

from collections.abc import Iterable

try:
    from diffusers import FlowMatchEulerDiscreteScheduler  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - dependency classification handled above the seam
    FlowMatchEulerDiscreteScheduler = None

from .utils import clamp_unit, stable_signature


class SchedulerDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


def default_scheduler_name() -> str:
    return 'flow-matching-euler-discrete'


def _to_float_list(values: object) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values if isinstance(value, (int, float))]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
        return [float(value) for value in values if isinstance(value, (int, float))]
    return []


def _derive_sigmas(timesteps: list[float], step_count: int) -> list[float]:
    if not timesteps:
        return []

    length = max(step_count, 1)
    if length == 1:
        return [1.0]

    return [clamp_unit(1.0 - (index / (length - 1))) for index, _ in enumerate(timesteps)]


def build_flow_matching_schedule(*, steps: int, guidance_scale: float) -> dict[str, object]:
    if FlowMatchEulerDiscreteScheduler is None:
        raise SchedulerDependencyError('Required runtime import is unavailable: diffusers.FlowMatchEulerDiscreteScheduler.')

    step_count = max(int(steps), 1)
    scheduler_config = {
        'num_train_timesteps': step_count,
        'shift': 1.0,
        'use_dynamic_shifting': False,
    }
    if hasattr(FlowMatchEulerDiscreteScheduler, 'from_config'):
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

    if hasattr(scheduler, 'set_timesteps'):
        scheduler.set_timesteps(step_count)

    raw_timesteps = getattr(scheduler, 'timesteps', None)
    timesteps = _to_float_list(raw_timesteps)
    if not timesteps:
        timesteps = [float(index) for index in range(step_count)]

    raw_sigmas = getattr(scheduler, 'sigmas', None)
    sigmas = _to_float_list(raw_sigmas)
    if len(sigmas) != len(timesteps):
        sigmas = _derive_sigmas(timesteps, len(timesteps))

    return {
        'family': default_scheduler_name(),
        'target': 'diffusers.FlowMatchEulerDiscreteScheduler',
        'instance': scheduler,
        'object_type': scheduler.__class__.__name__,
        'step_count': step_count,
        'guidance_scale': guidance_scale,
        'timesteps': timesteps,
        'consumed_timesteps': timesteps,
        'sigmas': sigmas,
        'consumed_sigmas': sigmas,
        'timestep_count': len(timesteps),
        'timestep_signature': stable_signature(timesteps),
        'sigma_start': sigmas[0] if sigmas else 0.0,
        'sigma_end': sigmas[-1] if sigmas else 0.0,
    }
