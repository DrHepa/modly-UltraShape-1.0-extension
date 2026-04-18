"""Checkpoint-backed image conditioner subset for the vendored runtime."""

from __future__ import annotations

from ..utils.checkpoint import checkpoint_signature, checkpoint_tensor_count, checkpoint_tokens, checkpoint_value_count
from ..utils import blend_sequences, clamp_unit, stable_signature


class SingleImageEncoder:
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

    def build(self, *, reference_asset: dict[str, object], coarse_surface: dict[str, object]) -> dict[str, object]:
        image_tokens = reference_asset.get('tokens') if isinstance(reference_asset.get('tokens'), list) else []
        image_values = reference_asset.get('image_tokens') if isinstance(reference_asset.get('image_tokens'), list) else []
        mask_values = reference_asset.get('mask_tokens') if isinstance(reference_asset.get('mask_tokens'), list) else []
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
        mesh_tokens = mesh.get('tokens') if isinstance(mesh.get('tokens'), list) else []
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface.get('voxels'), dict) else {}
        voxel_values = voxels.get('voxel_values') if isinstance(voxels.get('voxel_values'), list) else []
        voxel_tokens = [value / 12.0 for value in voxel_values] if voxel_values else []
        bounds = mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {}
        extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_signal = checkpoint_tokens(checkpoint_reference)
        encoded_tokens = blend_sequences(image_tokens, mesh_tokens, voxel_tokens, checkpoint_signal)[:8]
        conditioning_strength = clamp_unit(sum(encoded_tokens) / len(encoded_tokens) if encoded_tokens else 0.0)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        image_mean = clamp_unit(sum(float(value) for value in image_values) / len(image_values) if image_values else 0.0)
        mesh_extent_sum = round(sum(float(axis) for axis in extents), 6)
        occupied_ratio = clamp_unit(
            float(voxels.get('voxel_count', 0)) / max(float(reference_asset.get('pixel_count', 0)) + len(voxel_values), 1.0)
        )

        return {
            'encoder': self.__class__.__name__,
            'mask_tokens': len(encoded_tokens),
            'voxel_count': int(voxels.get('voxel_count', 0)),
            'conditioning_strength': conditioning_strength,
            'tokens': encoded_tokens,
            'image_token_signature': stable_signature([float(value) for value in image_values if isinstance(value, (int, float))]),
            'mask_token_signature': stable_signature([float(value) for value in mask_values if isinstance(value, (int, float))]),
            'checkpoint_signature': checkpoint_state_signature,
            'checkpoint_tensor_count': checkpoint_tensor_count(checkpoint_reference),
            'checkpoint_value_count': checkpoint_value_count(checkpoint_reference),
            'conditioning_signature': stable_signature(encoded_tokens),
            'conditioning_mean': conditioning_strength,
            'image_mean': image_mean,
            'mesh_extent_sum': mesh_extent_sum,
            'occupied_ratio': occupied_ratio,
            'signature': stable_signature(encoded_tokens),
            'state_hydrated': self.hydrated,
            'hydration': dict(self.hydration) if isinstance(self.hydration, dict) else None,
        }


class ConditionerMask(SingleImageEncoder):
    pass


def build_conditioning_mask(*, reference_asset: dict[str, object], coarse_surface: dict[str, object], checkpoint_state: object = None) -> dict[str, object]:
    return SingleImageEncoder(checkpoint_state=checkpoint_state).build(
        reference_asset=reference_asset,
        coarse_surface=coarse_surface,
    )
