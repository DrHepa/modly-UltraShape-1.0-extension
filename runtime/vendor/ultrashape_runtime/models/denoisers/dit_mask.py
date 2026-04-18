"""Patched DiT mask seam with optional flash-attn fallback."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature
from .moe_layers import mix_expert_sequences, moe_enabled

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
        conditioning: dict[str, object],
        schedule: dict[str, object],
        seed: int | None,
    ) -> dict[str, object]:
        conditioning_tokens = conditioning.get('tokens') if isinstance(conditioning.get('tokens'), list) else []
        timesteps = schedule.get('timesteps') if isinstance(schedule.get('timesteps'), list) else []
        guidance_scale = float(schedule.get('guidance_scale', 0.0)) if isinstance(schedule.get('guidance_scale'), (int, float)) else 0.0
        schedule_signature = int(schedule.get('timestep_signature', 0)) if isinstance(schedule.get('timestep_signature'), int) else 0
        conditioning_signature = int(conditioning.get('conditioning_signature', conditioning.get('signature', 0))) if isinstance(conditioning.get('conditioning_signature', conditioning.get('signature', 0)), int) else 0
        seed_value = 0 if seed is None else seed
        attention = 'flash_attn' if flash_attn_available() else 'sdpa'
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_values = checkpoint_tokens(checkpoint_reference)
        expert_tokens = mix_expert_sequences(conditioning_tokens, checkpoint_values, timesteps)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        latents: list[float] = []

        for index, token in enumerate(conditioning_tokens[:8]):
            schedule_value = timesteps[index % len(timesteps)] if timesteps else 0.0
            checkpoint_value = checkpoint_values[index % len(checkpoint_values)] if checkpoint_values else 0.0
            expert_value = expert_tokens[index % len(expert_tokens)] if expert_tokens else 0.0
            latent = clamp_unit(
                (token * 0.31)
                + (schedule_value * 0.19)
                + (guidance_scale / 60.0)
                + (((seed_value + index) % 11) / 150.0)
                + (checkpoint_value * 0.17)
                + (expert_value * 0.11)
                + ((checkpoint_state_signature % 17) / 500.0)
                + ((schedule_signature % 19) / 700.0)
                + ((conditioning_signature % 23) / 800.0)
            )
            latents.append(latent)

        return {
            'model': self.__class__.__name__,
            'attention': attention,
            'latents': latents,
            'latent_count': len(latents),
            'latent_mean': clamp_unit(sum(latents) / len(latents) if latents else 0.0),
            'latent_signature': stable_signature(latents),
            'checkpoint_signature': checkpoint_state_signature,
            'scheduler_signature': schedule_signature,
            'conditioning_signature': conditioning_signature,
            'timestep_count': len(timesteps),
            'moe_enabled': moe_enabled(),
            'state_hydrated': self.hydrated,
            'hydration': dict(self.hydration) if isinstance(self.hydration, dict) else None,
        }


def denoise_conditioned_latents(
    *,
    conditioning: dict[str, object],
    schedule: dict[str, object],
    checkpoint_state: object = None,
    seed: int | None,
) -> dict[str, object]:
    return RefineDiT(checkpoint_state=checkpoint_state).denoise(
        conditioning=conditioning,
        schedule=schedule,
        seed=seed,
    )
