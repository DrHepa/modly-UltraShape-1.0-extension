"""Portable upstream-shaped conditioning subset for the vendored runtime."""

from __future__ import annotations

from ..utils.checkpoint import checkpoint_parameter_map, checkpoint_signature, checkpoint_tensor_count, checkpoint_tokens, checkpoint_value_count
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


def _image_rows(image: object) -> list[list[list[float]]]:
    if not isinstance(image, list) or not image or not isinstance(image[0], list):
        return []
    rows = image[0]
    return rows if isinstance(rows, list) else []


def _mask_rows(mask: object) -> list[list[list[float]]]:
    if not isinstance(mask, list) or not mask or not isinstance(mask[0], list):
        return []
    rows = mask[0]
    return rows if isinstance(rows, list) else []


def _patch_tokens(image: object, mask: object | None = None) -> tuple[list[list[float]], list[float]]:
    rows = _image_rows(image)
    mask_rows = _mask_rows(mask)
    if not rows:
        return ([[0.0, 0.0, 0.0, 0.0]], [0.0])

    tokens: list[list[float]] = []
    token_mask: list[float] = []

    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        row_pixels = [pixel for pixel in row if isinstance(pixel, list) and len(pixel) >= 4]
        if not row_pixels:
            continue

        reds = [float(pixel[0]) for pixel in row_pixels]
        greens = [float(pixel[1]) for pixel in row_pixels]
        blues = [float(pixel[2]) for pixel in row_pixels]
        alphas = [float(pixel[3]) for pixel in row_pixels]

        row_mask_values: list[float] = []
        if row_index < len(mask_rows) and isinstance(mask_rows[row_index], list):
            for mask_pixel in mask_rows[row_index]:
                if isinstance(mask_pixel, list) and mask_pixel and isinstance(mask_pixel[0], (int, float)):
                    row_mask_values.append(float(mask_pixel[0]))

        valid_mask = clamp_unit(sum(row_mask_values) / len(row_mask_values) if row_mask_values else 1.0)
        luminance = clamp_unit(((sum(reds) / len(reds)) * 0.299) + ((sum(greens) / len(greens)) * 0.587) + ((sum(blues) / len(blues)) * 0.114))
        chroma = clamp_unit(abs((sum(reds) / len(reds)) - (sum(blues) / len(blues))))
        alpha = clamp_unit(sum(alphas) / len(alphas))
        green = clamp_unit(sum(greens) / len(greens))
        tokens.append([luminance, chroma, alpha, green])
        token_mask.append(valid_mask)

    if not tokens:
        return ([[0.0, 0.0, 0.0, 0.0]], [0.0])
    return tokens, token_mask


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
        bucket = buckets[index % groups]
        bucket.append(
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
        vectors.append([clamp_unit(sum(item[index] for item in bucket) / len(bucket)) for index in range(4)])
    return vectors


def _checkpoint_channels(checkpoint_state: object, *, width: int = 4) -> list[float]:
    return _group_mean(checkpoint_tokens(checkpoint_state, limit=max(width * 2, width)), width)


def _hydrate_module_family(
    state_dict: dict[str, object],
    *,
    allowed_roots: tuple[str, ...],
    strict: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    recognized: dict[str, object] = {}
    unexpected_keys: list[str] = []
    for parameter_name, parameter_value in state_dict.items():
        if any(parameter_name == root or parameter_name.startswith(f'{root}.') for root in allowed_roots):
            recognized[parameter_name] = parameter_value
        else:
            unexpected_keys.append(parameter_name)

    missing_roots = [root for root in allowed_roots if not any(name == root or name.startswith(f'{root}.') for name in recognized)]
    legacy_keys = [name for name in state_dict if name == 'tensors' or name.startswith('tensors.')]
    if legacy_keys and not recognized and all(name in legacy_keys for name in state_dict):
        return dict(state_dict), [], []
    if strict and (unexpected_keys or missing_roots or not recognized):
        raise ValueError(
            'SingleImageEncoder strict hydration requires upstream module-family keys for '
            f'{allowed_roots}; missing={missing_roots}, unexpected={unexpected_keys}.'
        )
    return recognized, missing_roots, unexpected_keys


def _flatten_vectors(vectors: list[list[float]]) -> list[float]:
    return [value for vector in vectors for value in vector]


def _expand_mask(token_mask: list[float], feature_dim: int) -> list[float]:
    expanded: list[float] = []
    for value in token_mask:
        expanded.extend([clamp_unit(value)] * feature_dim)
    return expanded


class PortableImageEncoder:
    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state

    def __call__(self, image, mask=None, **kwargs):
        del kwargs
        main, token_mask = _patch_tokens(image, mask)
        return {'main': main, 'main_mask': token_mask}

    def unconditional_embedding(self, batch_size: int, **kwargs) -> list[list[float]] | list[list[list[float]]]:
        num_tokens = kwargs.get('num_tokens', 1)
        feature_dim = kwargs.get('feature_dim', 4)
        zero_tokens = [[0.0 for _ in range(feature_dim)] for _ in range(max(int(num_tokens), 1))]
        if batch_size <= 1:
            return zero_tokens
        return [list(zero_tokens) for _ in range(batch_size)]


class SingleImageEncoder:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None, drop_ratio: float = 0.1):
        self.checkpoint_state = checkpoint_state
        self.drop_ratio = drop_ratio
        self.main_image_encoder = PortableImageEncoder(checkpoint_state=checkpoint_state)
        self.state_dict: dict[str, object] | None = None
        self.hydration: dict[str, object] | None = None
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        normalized_state = checkpoint_parameter_map({'state_dict': state_dict}) or dict(state_dict)
        hydrated_state, missing_roots, unexpected_keys = _hydrate_module_family(
            normalized_state,
            allowed_roots=('main_image_encoder',),
            strict=strict,
        )
        self.checkpoint_state = {
            'state_dict': hydrated_state,
            'state_dict_metadata': {
                'parameter_count': len(hydrated_state),
                'module_roots': ['main_image_encoder'],
                'module_family': self.__class__.__name__,
            },
            'representation': 'module-state-dict-v2',
        }
        self.main_image_encoder.checkpoint_state = {
            'state_dict': {
                name: value
                for name, value in hydrated_state.items()
                if name == 'main_image_encoder' or name.startswith('main_image_encoder.')
            },
            'representation': 'module-state-dict-v2',
        }
        self.state_dict = hydrated_state
        self.hydrated = True
        return {'missing_keys': missing_roots, 'unexpected_keys': unexpected_keys, 'strict': strict}

    def forward(self, image, disable_drop: bool = True, mask=None, **kwargs) -> dict[str, list[list[float]] | list[float]]:
        del kwargs
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_profile = _checkpoint_channels(checkpoint_reference)
        encoded = self.main_image_encoder(image, mask=mask)
        image_tokens = encoded.get('main', []) if isinstance(encoded.get('main'), list) else []
        token_mask = encoded.get('main_mask', []) if isinstance(encoded.get('main_mask'), list) else []

        projected_tokens: list[list[float]] = []
        for token_index, token in enumerate(image_tokens):
            checkpoint_value = checkpoint_profile[token_index % len(checkpoint_profile)] if checkpoint_profile else 0.0
            projected_tokens.append(
                [
                    clamp_unit((token[0] * 0.7) + (token[2] * 0.1) + (checkpoint_value * 0.2)),
                    clamp_unit((token[1] * 0.55) + (token[3] * 0.2) + (checkpoint_value * 0.25)),
                    clamp_unit((token[2] * 0.6) + (token[0] * 0.15) + (checkpoint_value * 0.25)),
                    clamp_unit((token[3] * 0.65) + (token[1] * 0.1) + (checkpoint_value * 0.25)),
                ]
            )

        if not disable_drop and self.drop_ratio >= 1.0:
            projected_tokens = [[0.0 for _ in token] for token in projected_tokens]

        return {
            'main': projected_tokens,
            'main_mask': token_mask,
        }

    def unconditional_embedding(self, batch_size: int, **kwargs) -> dict[str, list[list[float]]]:
        return {'main': self.main_image_encoder.unconditional_embedding(batch_size, **kwargs)}

    def build(self, *, reference_asset: dict[str, object], coarse_surface: dict[str, object]) -> dict[str, object]:
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict

        forward_tokens = self.forward(reference_asset.get('image_tensor'), mask=reference_asset.get('mask_tensor'))
        image_vectors = list(forward_tokens.get('main', [])) if isinstance(forward_tokens.get('main'), list) else []
        token_mask = [float(value) for value in forward_tokens.get('main_mask', []) if isinstance(value, (int, float))]
        voxel_vectors = _voxel_region_vectors(voxel_cond)
        checkpoint_channels = _checkpoint_channels(checkpoint_reference)
        unconditional_tokens = self.unconditional_embedding(1, num_tokens=max(len(image_vectors), 1)).get('main', [])

        vector_count = max(len(image_vectors), len(voxel_vectors), 1)
        context_vectors: list[list[float]] = []
        unconditional_vectors: list[list[float]] = []

        for index in range(vector_count):
            image_vector = image_vectors[index % len(image_vectors)] if image_vectors else [0.0, 0.0, 0.0, 0.0]
            voxel_vector = voxel_vectors[index % len(voxel_vectors)] if voxel_vectors else [0.0, 0.0, 0.0, 0.0]
            checkpoint_value = checkpoint_channels[index % len(checkpoint_channels)] if checkpoint_channels else 0.0
            unconditional_vector = unconditional_tokens[index % len(unconditional_tokens)] if unconditional_tokens else [0.0, 0.0, 0.0, 0.0]
            context_vectors.append(
                [
                    clamp_unit((image_vector[0] * 0.55) + (voxel_vector[3] * 0.2) + (checkpoint_value * 0.25)),
                    clamp_unit((image_vector[1] * 0.45) + (voxel_vector[0] * 0.3) + (checkpoint_value * 0.25)),
                    clamp_unit((image_vector[2] * 0.5) + (voxel_vector[1] * 0.25) + (checkpoint_value * 0.25)),
                    clamp_unit((image_vector[3] * 0.45) + (voxel_vector[2] * 0.25) + (checkpoint_value * 0.3)),
                ]
            )
            unconditional_vectors.append(
                [
                    clamp_unit((unconditional_vector[0] * 0.4) + (voxel_vector[3] * 0.35) + (checkpoint_value * 0.25)),
                    clamp_unit((unconditional_vector[1] * 0.4) + (voxel_vector[0] * 0.35) + (checkpoint_value * 0.25)),
                    clamp_unit((unconditional_vector[2] * 0.4) + (voxel_vector[1] * 0.35) + (checkpoint_value * 0.25)),
                    clamp_unit((unconditional_vector[3] * 0.4) + (voxel_vector[2] * 0.35) + (checkpoint_value * 0.25)),
                ]
            )

        context = _flatten_vectors(context_vectors)
        unconditional_context = _flatten_vectors(unconditional_vectors)
        context_mask = _expand_mask(token_mask or [1.0] * len(context_vectors), 4)

        metadata = {
            'image_signature': int(reference_asset.get('image_signature', stable_signature(_flatten_vectors(image_vectors)))),
            'mask_signature': int(reference_asset.get('mask_signature', stable_signature(token_mask))) if isinstance(reference_asset.get('mask_signature', 0), int) else stable_signature(token_mask),
            'mesh_signature': int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else 0,
            'voxel_signature': int(voxel_cond.get('voxel_signature', 0)) if isinstance(voxel_cond.get('voxel_signature'), int) else 0,
            'checkpoint_signature': checkpoint_signature(checkpoint_reference),
            'checkpoint_tensor_count': checkpoint_tensor_count(checkpoint_reference),
            'checkpoint_value_count': checkpoint_value_count(checkpoint_reference),
            'checkpoint_channel_count': len(checkpoint_channels),
            'image_tensor_shape': reference_asset.get('image_tensor_shape'),
            'mask_tensor_shape': reference_asset.get('mask_tensor_shape'),
            'surface_point_count': len(coarse_surface.get('sampled_surface_points', [])) if isinstance(coarse_surface.get('sampled_surface_points'), list) else 0,
            'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else 0,
            'context_signature': stable_signature(context),
            'context_mask_signature': stable_signature(context_mask),
            'image_region_count': len(image_vectors),
            'voxel_region_count': len(voxel_vectors),
            'context_vector_signature': stable_signature(_flatten_vectors(context_vectors)),
        }

        cfg_pairing = {
            'mode': 'classifier-free-guidance',
            'positive_context_tokens': len(context),
            'negative_context_tokens': len(unconditional_context),
            'positive_context_signature': metadata['context_signature'],
            'negative_context_signature': stable_signature(unconditional_context),
        }

        image_features = _flatten_numeric(reference_asset.get('image_features'))
        bounds = mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {}
        extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
        occupied_ratio = clamp_unit(float(voxel_cond.get('occupied_ratio', 0.0)))
        conditioning_mean = clamp_unit(sum(context) / len(context) if context else 0.0)

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
