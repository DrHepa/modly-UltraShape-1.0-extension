"""Autoencoder model subset for local runtime wiring."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils import blend_sequences, clamp_unit, stable_signature


def _numeric_sequence(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def _expand_signal(values: list[float], target_length: int) -> list[float]:
    if target_length <= 0:
        return []
    if not values:
        return [0.0] * target_length

    expanded: list[float] = []
    while len(expanded) < target_length:
        remaining = target_length - len(expanded)
        expanded.extend(values[:remaining])
    return expanded[:target_length]


def _voxel_resolution(voxel_cond: dict[str, object]) -> int:
    resolution = voxel_cond.get('resolution') if isinstance(voxel_cond, dict) else 0
    if isinstance(resolution, int) and resolution > 0:
        return resolution
    return 4


class ShapeVAE:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state
        self.state_dict: dict[str, object] | None = None
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        self.checkpoint_state = state_dict
        self.state_dict = dict(state_dict)
        self.hydrated = True
        return {'missing_keys': [], 'unexpected_keys': [], 'strict': strict}

    def decode_latents(
        self,
        latents: dict[str, object],
        reference_asset: dict[str, object],
        conditioning: dict[str, object] | None = None,
        coarse_surface: dict[str, object] | None = None,
    ) -> dict[str, object]:
        latent_values = _numeric_sequence(latents.get('latents'))
        reference_tokens = _numeric_sequence(reference_asset.get('image_features'))
        conditioning_context = _numeric_sequence(conditioning.get('context')) if isinstance(conditioning, dict) else []
        conditioning_mask = _numeric_sequence(conditioning.get('context_mask')) if isinstance(conditioning, dict) else []
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxels'), dict) else {}
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('mesh'), dict) else {}
        voxel_values = _numeric_sequence(voxel_cond.get('occupancies')) or _numeric_sequence(voxels.get('voxel_values'))
        max_voxel_value = max(voxel_values, default=0.0)
        normalized_voxel_values = [clamp_unit(value / max(max_voxel_value, 1.0)) for value in voxel_values]
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_signal = checkpoint_tokens(checkpoint_reference, limit=16)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        resolution = _voxel_resolution(voxel_cond)
        field_value_count = max(resolution ** 3, 64)

        decoded_signal = blend_sequences(
            latent_values,
            reference_tokens,
            conditioning_context,
            conditioning_mask,
            normalized_voxel_values,
            checkpoint_signal,
        )
        decoded = [
            clamp_unit((value * 0.73) + (((index % 11) + 1) / 140.0))
            for index, value in enumerate(_expand_signal(decoded_signal, field_value_count))
        ]
        field_logits = [
            round(((value - 0.5) * 1.6) + (((index % 7) - 3) / 20.0), 6)
            for index, value in enumerate(decoded)
        ]
        occupancy_field = [clamp_unit((logit + 1.0) / 2.0) for logit in field_logits]
        field_signature = stable_signature(field_logits)

        return {
            'vae': self.__class__.__name__,
            'authority': 'field_logits',
            'decoded_latents': decoded,
            'field_logits': field_logits,
            'occupancy_field': occupancy_field,
            'field_value_count': len(field_logits),
            'field_resolution': resolution,
            'field_signature': field_signature,
            'decoded_signature': stable_signature(decoded),
            'checkpoint_signature': checkpoint_state_signature,
            'spatial_context': {
                'voxel_coords': list(voxel_cond.get('coords')) if isinstance(voxel_cond.get('coords'), list) else list(voxels.get('voxel_coords')) if isinstance(voxels.get('voxel_coords'), list) else [],
                'voxel_values': voxel_values,
                'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else int(voxels.get('voxel_count', 0)) if isinstance(voxels.get('voxel_count'), int) else len(voxel_values),
                'resolution': resolution,
                'voxel_signature': int(voxel_cond.get('signature', 0)) if isinstance(voxel_cond.get('signature'), int) else int(voxels.get('voxel_signature', 0)) if isinstance(voxels.get('voxel_signature'), int) else 0,
                'bounds': voxel_cond.get('bounds') if isinstance(voxel_cond.get('bounds'), dict) else voxels.get('bounds') if isinstance(voxels.get('bounds'), dict) else {},
                'mesh_bounds': mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {},
                'mesh_signature': int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else 0,
                'reference_signature': int(reference_asset.get('signature', 0)) if isinstance(reference_asset.get('signature'), int) else 0,
                'conditioning_signature': int(conditioning.get('conditioning_signature', 0)) if isinstance(conditioning, dict) and isinstance(conditioning.get('conditioning_signature'), int) else 0,
            },
            'evidence': {
                'decoded_count': len(decoded),
                'field_value_count': len(field_logits),
                'field_signature': field_signature,
            },
            'state_hydrated': self.hydrated,
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
