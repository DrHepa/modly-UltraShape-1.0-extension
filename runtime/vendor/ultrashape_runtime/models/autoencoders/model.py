"""Autoencoder model subset for local runtime wiring."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils.tensors import blend_sequences, clamp_unit, stable_signature


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
        reference_tokens = _numeric_sequence(reference_asset.get('tokens')) or _numeric_sequence(reference_asset.get('image_features'))
        conditioning_tokens = _numeric_sequence(conditioning.get('tokens')) if isinstance(conditioning, dict) else []
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxels'), dict) else {}
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('mesh'), dict) else {}
        voxel_tokens = _numeric_sequence(voxels.get('tokens'))
        voxel_values = _numeric_sequence(voxels.get('voxel_values'))
        max_voxel_value = max(voxel_values, default=0.0)
        normalized_voxel_values = [value / max(max_voxel_value, 1.0) for value in voxel_values]
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_signal = checkpoint_tokens(checkpoint_reference, limit=16)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)

        decoded_signal = blend_sequences(
            latent_values,
            reference_tokens,
            conditioning_tokens,
            voxel_tokens,
            checkpoint_signal,
        )
        decoded = [
            clamp_unit((value * 0.76) + (((index % 11) + 1) / 120.0))
            for index, value in enumerate(_expand_signal(decoded_signal, 96))
        ]
        mesh_seed = [
            clamp_unit((value * 0.81) + (((index % 7) + 1) / 160.0))
            for index, value in enumerate(
                _expand_signal(
                    blend_sequences(decoded[:48], latent_values, normalized_voxel_values, checkpoint_signal),
                    48,
                )
            )
        ]
        volume_tokens = [
            clamp_unit((value * 0.84) + (((index % 5) + 1) / 180.0))
            for index, value in enumerate(
                _expand_signal(
                    blend_sequences(decoded, mesh_seed, conditioning_tokens, normalized_voxel_values, reference_tokens),
                    128,
                )
            )
        ]

        return {
            'vae': self.__class__.__name__,
            'decoded_latents': decoded,
            'mesh_seed': mesh_seed,
            'volume_tokens': volume_tokens,
            'decoded_signature': stable_signature(decoded),
            'volume_signature': stable_signature(volume_tokens),
            'checkpoint_signature': checkpoint_state_signature,
            'spatial_context': {
                'voxel_coords': list(voxels.get('voxel_coords')) if isinstance(voxels.get('voxel_coords'), list) else [],
                'voxel_values': voxel_values,
                'voxel_count': int(voxels.get('voxel_count', 0)) if isinstance(voxels.get('voxel_count'), int) else len(voxel_values),
                'resolution': int(voxels.get('resolution', 0)) if isinstance(voxels.get('resolution'), int) else 0,
                'voxel_signature': int(voxels.get('voxel_signature', 0)) if isinstance(voxels.get('voxel_signature'), int) else 0,
                'bounds': voxels.get('bounds') if isinstance(voxels.get('bounds'), dict) else {},
                'mesh_bounds': mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {},
                'mesh_signature': int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else 0,
                'reference_signature': int(reference_asset.get('signature', 0)) if isinstance(reference_asset.get('signature'), int) else 0,
                'conditioning_signature': int(conditioning.get('conditioning_signature', 0)) if isinstance(conditioning, dict) and isinstance(conditioning.get('conditioning_signature'), int) else 0,
            },
            'state_hydrated': self.hydrated,
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
