"""Checkpoint-backed image conditioner subset for the vendored runtime."""

from __future__ import annotations

from ..utils.checkpoint import checkpoint_signature, checkpoint_tensor_count, checkpoint_tokens, checkpoint_value_count
from ..utils import blend_sequences, clamp_unit, stable_signature


class _CompatState(dict):
    def __init__(self, *args, compat: dict[str, object] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compat = compat or {}

    def _resolve(self, key: str):
        value = self._compat[key]
        return value(self) if callable(value) else value

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if key in self._compat:
            return self._resolve(key)
        return default

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._compat:
            return self._resolve(key)
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self._compat


def _numeric_list(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def _surface_signal(points: object, *, limit: int = 8) -> list[float]:
    if not isinstance(points, list):
        return []

    signal: list[float] = []
    for point in points[:limit]:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        signal.extend(clamp_unit((float(axis) + 1.5) / 3.0) for axis in point[:3])
    return signal[:limit]


def _voxel_signal(voxel_cond: dict[str, object], *, limit: int = 8) -> list[float]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    occupancies = _numeric_list(voxel_cond.get('occupancies'))
    resolution = int(voxel_cond.get('resolution', 0)) if isinstance(voxel_cond.get('resolution'), int) else 0
    signal: list[float] = []

    for index, coords_triplet in enumerate(coords[:limit]):
        if isinstance(coords_triplet, (list, tuple)):
            signal.extend(
                clamp_unit(float(axis) / max(float(resolution), 1.0))
                for axis in coords_triplet[:3]
                if isinstance(axis, (int, float))
            )
        if index < len(occupancies):
            signal.append(clamp_unit(float(occupancies[index]) / max(len(coords), 1)))
    return signal[:limit]


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
        image_features = _numeric_list(reference_asset.get('image_features'))
        mask_features = _numeric_list(reference_asset.get('mask_features'))
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        surface_points = coarse_surface.get('sampled_surface_points')
        if not isinstance(surface_points, list):
            surface_points = mesh.get('sampled_surface_points') if isinstance(mesh.get('sampled_surface_points'), list) else []

        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_signal = checkpoint_tokens(checkpoint_reference)
        surface_signal = _surface_signal(surface_points)
        voxel_signal = _voxel_signal(voxel_cond)
        context = blend_sequences(image_features, mask_features, surface_signal, voxel_signal, checkpoint_signal)[:8]
        context_mask = [1.0 for _ in context]
        unconditional_context = blend_sequences(mask_features, voxel_signal, checkpoint_signal)[: len(context)]
        if len(unconditional_context) < len(context):
            unconditional_context.extend([0.0] * (len(context) - len(unconditional_context)))

        image_signature = int(reference_asset.get('image_signature', stable_signature(image_features)))
        mask_signature = int(reference_asset.get('mask_signature', stable_signature(mask_features)))
        mesh_signature = int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else stable_signature(surface_signal)
        voxel_signature = (
            int(voxel_cond.get('voxel_signature', 0))
            if isinstance(voxel_cond.get('voxel_signature'), int)
            else stable_signature(voxel_signal)
        )
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        context_signature = stable_signature(context)
        context_mask_signature = stable_signature(context_mask)
        conditioning_mean = clamp_unit(sum(context) / len(context) if context else 0.0)
        bounds = mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {}
        extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
        occupied_ratio = clamp_unit(float(voxel_cond.get('occupied_ratio', 0.0)))

        metadata = {
            'image_signature': image_signature,
            'mask_signature': mask_signature,
            'mesh_signature': mesh_signature,
            'voxel_signature': voxel_signature,
            'checkpoint_signature': checkpoint_state_signature,
            'checkpoint_tensor_count': checkpoint_tensor_count(checkpoint_reference),
            'checkpoint_value_count': checkpoint_value_count(checkpoint_reference),
            'image_tensor_shape': reference_asset.get('image_tensor_shape'),
            'mask_tensor_shape': reference_asset.get('mask_tensor_shape'),
            'surface_point_count': len(surface_points),
            'voxel_count': int(voxel_cond.get('voxel_count', 0)),
            'context_signature': context_signature,
            'context_mask_signature': context_mask_signature,
        }

        cfg_pairing = {
            'mode': 'classifier-free-guidance',
            'positive_context_tokens': len(context),
            'negative_context_tokens': len(unconditional_context),
            'positive_context_signature': context_signature,
            'negative_context_signature': stable_signature(unconditional_context),
        }

        return _CompatState(
            {
                'encoder': self.__class__.__name__,
                'context': context,
                'context_mask': context_mask,
                'cfg_pairing': cfg_pairing,
                'metadata': metadata,
                'state_hydrated': self.hydrated,
                'hydration': dict(self.hydration) if isinstance(self.hydration, dict) else None,
            },
            compat={
                'tokens': lambda state: list(state['context']),
                'checkpoint_signature': lambda state: state['metadata']['checkpoint_signature'],
                'checkpoint_tensor_count': lambda state: state['metadata']['checkpoint_tensor_count'],
                'checkpoint_value_count': lambda state: state['metadata']['checkpoint_value_count'],
                'conditioning_signature': lambda state: state['metadata']['context_signature'],
                'conditioning_mean': lambda state: conditioning_mean,
                'image_mean': lambda state: clamp_unit(sum(image_features) / len(image_features) if image_features else 0.0),
                'mesh_extent_sum': lambda state: round(sum(float(axis) for axis in extents), 6),
                'occupied_ratio': lambda state: occupied_ratio,
                'image_token_signature': lambda state: state['metadata']['image_signature'],
                'mask_token_signature': lambda state: state['metadata']['mask_signature'],
                'voxel_count': lambda state: state['metadata']['voxel_count'],
                'signature': lambda state: state['metadata']['context_signature'],
            },
        )


class ConditionerMask(SingleImageEncoder):
    pass


def build_conditioning_mask(*, reference_asset: dict[str, object], coarse_surface: dict[str, object], checkpoint_state: object = None) -> dict[str, object]:
    return SingleImageEncoder(checkpoint_state=checkpoint_state).build(
        reference_asset=reference_asset,
        coarse_surface=coarse_surface,
    )
