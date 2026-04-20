"""Checkpoint-backed image/voxel conditioner for the vendored runtime."""

from __future__ import annotations

from ..utils.checkpoint import checkpoint_signature, checkpoint_tensor_count, checkpoint_tokens, checkpoint_value_count
from ..utils import clamp_unit, stable_signature


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


def _flatten_numeric(candidate: object) -> list[float]:
    values: list[float] = []

    def visit(node: object) -> None:
        if isinstance(node, (int, float)):
            values.append(float(node))
            return
        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(candidate)
    return values


def _group_mean(values: list[float], groups: int) -> list[float]:
    if groups <= 0:
        return []
    if not values:
        return [0.0] * groups

    grouped: list[float] = []
    for group_index in range(groups):
        bucket = [value for index, value in enumerate(values) if index % groups == group_index]
        grouped.append(clamp_unit(sum(bucket) / len(bucket) if bucket else 0.0))
    return grouped


def _image_region_vectors(reference_asset: dict[str, object]) -> list[list[float]]:
    image_tensor = reference_asset.get('image_tensor')
    mask_tensor = reference_asset.get('mask_tensor')
    if not isinstance(image_tensor, list) or not image_tensor or not isinstance(image_tensor[0], list):
        return []

    rows = image_tensor[0]
    mask_rows = mask_tensor[0] if isinstance(mask_tensor, list) and mask_tensor and isinstance(mask_tensor[0], list) else []
    height = len(rows)
    width = len(rows[0]) if rows and isinstance(rows[0], list) else 0
    if height <= 0 or width <= 0:
        return []

    quadrants: list[list[list[float]]] = [[], [], [], []]
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for column_index, pixel in enumerate(row):
            if not isinstance(pixel, list) or len(pixel) < 4:
                continue
            quadrant = (0 if row_index < max(height / 2, 1) else 2) + (0 if column_index < max(width / 2, 1) else 1)
            alpha = 1.0
            if row_index < len(mask_rows) and isinstance(mask_rows[row_index], list) and column_index < len(mask_rows[row_index]):
                mask_pixel = mask_rows[row_index][column_index]
                if isinstance(mask_pixel, list) and mask_pixel and isinstance(mask_pixel[0], (int, float)):
                    alpha = float(mask_pixel[0])
            quadrants[quadrant].append([float(pixel[0]), float(pixel[1]), float(pixel[2]), alpha])

    vectors: list[list[float]] = []
    for pixels in quadrants:
        if not pixels:
            vectors.append([0.0, 0.0, 0.0, 0.0])
            continue
        red = clamp_unit(sum(pixel[0] for pixel in pixels) / len(pixels))
        green = clamp_unit(sum(pixel[1] for pixel in pixels) / len(pixels))
        blue = clamp_unit(sum(pixel[2] for pixel in pixels) / len(pixels))
        alpha = clamp_unit(sum(pixel[3] for pixel in pixels) / len(pixels))
        luminance = clamp_unit((red * 0.299) + (green * 0.587) + (blue * 0.114))
        chroma = clamp_unit(abs(red - blue))
        vectors.append([luminance, chroma, alpha, green])
    return vectors


def _voxel_region_vectors(voxel_cond: dict[str, object], *, groups: int = 4) -> list[list[float]]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    occupancies = _flatten_numeric(voxel_cond.get('occupancies'))
    resolution = int(voxel_cond.get('resolution', 0)) if isinstance(voxel_cond.get('resolution'), int) else 0
    scale = float(max(resolution, 1))
    buckets: list[list[list[float]]] = [[] for _ in range(groups)]

    for index, coord in enumerate(coords):
        if not isinstance(coord, (list, tuple)) or len(coord) < 3:
            continue
        occupancy = occupancies[index] if index < len(occupancies) else 0.0
        bucket_index = index % groups
        buckets[bucket_index].append(
            [
                clamp_unit(float(coord[0]) / scale),
                clamp_unit(float(coord[1]) / scale),
                clamp_unit(float(coord[2]) / scale),
                clamp_unit(float(occupancy)),
            ]
        )

    vectors: list[list[float]] = []
    for bucket in buckets:
        if not bucket:
            vectors.append([0.0, 0.0, 0.0, 0.0])
            continue
        vectors.append(
            [
                clamp_unit(sum(item[0] for item in bucket) / len(bucket)),
                clamp_unit(sum(item[1] for item in bucket) / len(bucket)),
                clamp_unit(sum(item[2] for item in bucket) / len(bucket)),
                clamp_unit(sum(item[3] for item in bucket) / len(bucket)),
            ]
        )
    return vectors


def _checkpoint_channels(checkpoint_state: object, *, width: int = 4) -> list[float]:
    return _group_mean(checkpoint_tokens(checkpoint_state, limit=max(width * 2, width)), width)


def _flatten_vectors(vectors: list[list[float]]) -> list[float]:
    return [value for vector in vectors for value in vector]


def _project_context_vectors(
    *,
    image_vectors: list[list[float]],
    voxel_vectors: list[list[float]],
    checkpoint_channels: list[float],
) -> tuple[list[list[float]], list[list[float]]]:
    vector_count = max(len(image_vectors), len(voxel_vectors), 1)
    checkpoint = checkpoint_channels or [0.0, 0.0, 0.0, 0.0]
    context_vectors: list[list[float]] = []
    unconditional_vectors: list[list[float]] = []

    for index in range(vector_count):
        image_vector = image_vectors[index % len(image_vectors)] if image_vectors else [0.0, 0.0, 0.0, 0.0]
        voxel_vector = voxel_vectors[index % len(voxel_vectors)] if voxel_vectors else [0.0, 0.0, 0.0, 0.0]
        context_vectors.append(
            [
                clamp_unit((image_vector[0] * (0.55 + checkpoint[0] * 0.25)) + (voxel_vector[3] * 0.25)),
                clamp_unit((image_vector[1] * 0.45) + (voxel_vector[0] * (0.35 + checkpoint[1] * 0.2))),
                clamp_unit((image_vector[2] * 0.5) + (voxel_vector[1] * 0.3) + (checkpoint[2] * 0.2)),
                clamp_unit((image_vector[3] * 0.35) + (voxel_vector[2] * 0.35) + (checkpoint[3] * 0.3)),
            ]
        )
        unconditional_vectors.append(
            [
                clamp_unit((voxel_vector[3] * 0.45) + (checkpoint[0] * 0.2)),
                clamp_unit((voxel_vector[0] * 0.4) + (checkpoint[1] * 0.2)),
                clamp_unit((voxel_vector[1] * 0.4) + (checkpoint[2] * 0.2)),
                clamp_unit((voxel_vector[2] * 0.4) + (checkpoint[3] * 0.2)),
            ]
        )

    return context_vectors, unconditional_vectors


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
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict

        image_vectors = _image_region_vectors(reference_asset)
        voxel_vectors = _voxel_region_vectors(voxel_cond)
        checkpoint_channels = _checkpoint_channels(checkpoint_reference)
        context_vectors, unconditional_vectors = _project_context_vectors(
            image_vectors=image_vectors,
            voxel_vectors=voxel_vectors,
            checkpoint_channels=checkpoint_channels,
        )

        context = _flatten_vectors(context_vectors)
        unconditional_context = _flatten_vectors(unconditional_vectors)
        context_mask = [1.0 for _ in context]

        image_signature = int(reference_asset.get('image_signature', stable_signature(_flatten_vectors(image_vectors))))
        mask_signature = int(reference_asset.get('mask_signature', 0)) if isinstance(reference_asset.get('mask_signature'), int) else 0
        mesh_signature = int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else 0
        voxel_signature = int(voxel_cond.get('voxel_signature', 0)) if isinstance(voxel_cond.get('voxel_signature'), int) else 0
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        context_signature = stable_signature(context)
        context_vector_signature = stable_signature(_flatten_vectors(context_vectors))

        metadata = {
            'image_signature': image_signature,
            'mask_signature': mask_signature,
            'mesh_signature': mesh_signature,
            'voxel_signature': voxel_signature,
            'checkpoint_signature': checkpoint_state_signature,
            'checkpoint_tensor_count': checkpoint_tensor_count(checkpoint_reference),
            'checkpoint_value_count': checkpoint_value_count(checkpoint_reference),
            'checkpoint_channel_count': len(checkpoint_channels),
            'image_tensor_shape': reference_asset.get('image_tensor_shape'),
            'mask_tensor_shape': reference_asset.get('mask_tensor_shape'),
            'surface_point_count': len(coarse_surface.get('sampled_surface_points', [])) if isinstance(coarse_surface.get('sampled_surface_points'), list) else 0,
            'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else 0,
            'context_signature': context_signature,
            'context_mask_signature': stable_signature(context_mask),
            'image_region_count': len(image_vectors),
            'voxel_region_count': len(voxel_vectors),
            'context_vector_signature': context_vector_signature,
        }

        cfg_pairing = {
            'mode': 'classifier-free-guidance',
            'positive_context_tokens': len(context),
            'negative_context_tokens': len(unconditional_context),
            'positive_context_signature': context_signature,
            'negative_context_signature': stable_signature(unconditional_context),
        }

        conditioning_mean = clamp_unit(sum(context) / len(context) if context else 0.0)
        image_features = _flatten_numeric(reference_asset.get('image_features'))
        bounds = mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {}
        extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
        occupied_ratio = clamp_unit(float(voxel_cond.get('occupied_ratio', 0.0)))

        return _CompatState(
            {
                'encoder': self.__class__.__name__,
                'context': context,
                'context_mask': context_mask,
                'context_vectors': context_vectors,
                'unconditional_context': unconditional_context,
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
