"""Scheduler seam for the vendored UltraShape runtime subset."""

from __future__ import annotations

try:
    from diffusers import FlowMatchEulerDiscreteScheduler  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - dependency classification handled above the seam
    FlowMatchEulerDiscreteScheduler = None

from .utils import clamp_unit, stable_signature


class SchedulerDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


def default_scheduler_name() -> str:
    return 'flow-matching-euler-discrete'


def build_flow_matching_schedule(*, steps: int, guidance_scale: float) -> dict[str, object]:
    if FlowMatchEulerDiscreteScheduler is None:
        raise SchedulerDependencyError('Required runtime import is unavailable: diffusers.FlowMatchEulerDiscreteScheduler.')

    scheduler_config = {
        'num_train_timesteps': max(int(steps), 1),
        'shift': 1.0,
        'use_dynamic_shifting': False,
    }
    if hasattr(FlowMatchEulerDiscreteScheduler, 'from_config'):
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

    if hasattr(scheduler, 'set_timesteps'):
        scheduler.set_timesteps(max(int(steps), 1))

    raw_timesteps = getattr(scheduler, 'timesteps', None)
    if isinstance(raw_timesteps, list) and raw_timesteps:
        timesteps = [round(index / max(len(raw_timesteps), 1), 6) for index, _ in enumerate(raw_timesteps)]
    else:
        timesteps = [round(index / max(steps, 1), 6) for index in range(steps)]

    return {
        'family': default_scheduler_name(),
        'target': 'diffusers.FlowMatchEulerDiscreteScheduler',
        'instance': scheduler,
        'step_count': steps,
        'guidance_scale': guidance_scale,
        'timesteps': timesteps,
        'timestep_count': len(timesteps),
        'timestep_signature': stable_signature(timesteps),
        'sigma_start': clamp_unit(1.0 - (1.0 / max(steps, 1))),
        'sigma_end': clamp_unit(1.0 / max(steps, 1)),
        'evidence': {
            'timestep_count': len(timesteps),
            'sigma_start': clamp_unit(1.0 - (1.0 / max(steps, 1))),
            'sigma_end': clamp_unit(1.0 / max(steps, 1)),
        },
    }
