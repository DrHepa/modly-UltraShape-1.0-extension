"""Conditioner mask subset for the vendored runtime."""

from __future__ import annotations

from ..utils.checkpoint import checkpoint_tokens
from ..utils.tensors import blend_sequences, clamp_unit, stable_signature


class ConditionerMask:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state

    def build(self, *, reference_asset: dict[str, object], coarse_surface: dict[str, object]) -> dict[str, object]:
        image_tokens = reference_asset.get('tokens') if isinstance(reference_asset.get('tokens'), list) else []
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
        mesh_tokens = mesh.get('tokens') if isinstance(mesh.get('tokens'), list) else []
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface.get('voxels'), dict) else {}
        voxel_values = voxels.get('voxel_values') if isinstance(voxels.get('voxel_values'), list) else []
        voxel_tokens = [value / 12.0 for value in voxel_values] if voxel_values else []
        checkpoint_signal = checkpoint_tokens(self.checkpoint_state)
        mask_tokens = blend_sequences(image_tokens, mesh_tokens, voxel_tokens, checkpoint_signal)[:8]

        return {
            'mask_tokens': len(mask_tokens),
            'voxel_count': int(voxels.get('voxel_count', 0)),
            'conditioning_strength': clamp_unit(sum(mask_tokens) / len(mask_tokens) if mask_tokens else 0.0),
            'tokens': mask_tokens,
            'signature': stable_signature(mask_tokens),
        }


def build_conditioning_mask(*, reference_asset: dict[str, object], coarse_surface: dict[str, object], checkpoint_state: object = None) -> dict[str, object]:
    return ConditionerMask(checkpoint_state=checkpoint_state).build(
        reference_asset=reference_asset,
        coarse_surface=coarse_surface,
    )
