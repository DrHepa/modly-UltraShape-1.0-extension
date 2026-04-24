"""Portable upstream-shaped autoencoder subset for the vendored runtime."""

from __future__ import annotations

import math

from ...utils.checkpoint import checkpoint_parameter_map, checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature
from .volume_decoders import decode_volume


SHAPEVAE_REQUIRED_ROOTS = ('post_kl', 'transformer', 'geo_decoder')
SHAPEVAE_UPSTREAM_ROOTS = (*SHAPEVAE_REQUIRED_ROOTS, 'encoder', 'pre_kl')


def _numeric_sequence(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


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


def _chunk_tokens(values: list[float], width: int = 4) -> list[list[float]]:
    if not values:
        return [[0.0 for _ in range(width)]]
    tokens: list[list[float]] = []
    for offset in range(0, len(values), width):
        token = [float(value) for value in values[offset : offset + width]]
        if len(token) < width:
            token.extend([0.0] * (width - len(token)))
        tokens.append(token)
    return tokens


def _voxel_resolution(voxel_cond: dict[str, object]) -> int:
    resolution = voxel_cond.get('resolution') if isinstance(voxel_cond, dict) else 0
    if isinstance(resolution, int) and resolution > 1:
        return resolution
    return 4


def _voxel_coords(voxel_cond: dict[str, object]) -> list[tuple[int, int, int]]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    parsed: list[tuple[int, int, int]] = []
    for coord in coords:
        if isinstance(coord, (list, tuple)) and len(coord) >= 3:
            parsed.append((int(coord[0]), int(coord[1]), int(coord[2])))
    return parsed


def _fourier_features(query: list[float]) -> list[float]:
    x_axis, y_axis, z_axis = [float(query[index]) if index < len(query) else 0.0 for index in range(3)]
    return [
        clamp_unit((math.sin(math.pi * x_axis) + 1.0) / 2.0),
        clamp_unit((math.cos(math.pi * y_axis) + 1.0) / 2.0),
        clamp_unit((math.sin(math.pi * z_axis * 0.5) + 1.0) / 2.0),
        clamp_unit((math.cos(math.pi * (x_axis + y_axis + z_axis) * 0.25) + 1.0) / 2.0),
    ]


def _hydrate_module_family(
    state_dict: dict[str, object],
    *,
    allowed_roots: tuple[str, ...],
    required_roots: tuple[str, ...] | None = None,
    strict: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    recognized: dict[str, object] = {}
    unexpected_keys: list[str] = []
    for parameter_name, parameter_value in state_dict.items():
        if any(parameter_name == root or parameter_name.startswith(f'{root}.') for root in allowed_roots):
            recognized[parameter_name] = parameter_value
        else:
            unexpected_keys.append(parameter_name)

    roots_required = required_roots or allowed_roots
    missing_roots = [root for root in roots_required if not any(name == root or name.startswith(f'{root}.') for name in recognized)]
    legacy_keys = [name for name in state_dict if name == 'tensors' or name.startswith('tensors.')]
    if legacy_keys and not recognized and all(name in legacy_keys for name in state_dict):
        return dict(state_dict), [], []
    if strict and (unexpected_keys or missing_roots or not recognized):
        raise ValueError(
            'ShapeVAE strict hydration requires upstream module-family keys for '
            f'{allowed_roots}; missing={missing_roots}, unexpected={unexpected_keys}.'
        )
    return recognized, missing_roots, unexpected_keys


class DiagonalGaussianDistribution:
    def __init__(self, parameters: object, deterministic: bool = False):
        self.parameters = parameters
        self.deterministic = deterministic

    def sample(self):
        return self.parameters

    def mode(self):
        return self.parameters


class VectsetVAE:
    def latents2mesh(self, latents, **kwargs):
        decoded_latents = {
            'field_logits': [float(value) for token in latents for value in token] if isinstance(latents, list) else [],
            'field_grid': [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]]],
            'field_signature': stable_signature([float(value) for token in latents for value in token]) if isinstance(latents, list) else 0,
            'spatial_context': {'voxel_coords': [[0, 0, 0], [1, 1, 1]]},
        }
        decoded_volume = decode_volume(decoded_latents)
        mesh = {
            'extractor': 'VanillaMCSurfaceExtractor',
            'authority': 'VectsetVAE.latents2mesh',
            'vertices': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            'faces': [(0, 1, 2)],
            'source_field_signature': decoded_volume['field_signature'],
        }
        return [mesh], decoded_volume


class ShapeVAE(VectsetVAE):
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state
        self.state_dict: dict[str, object] | None = None
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        normalized_state = checkpoint_parameter_map({'state_dict': state_dict}) or dict(state_dict)
        hydrated_state, missing_roots, unexpected_keys = _hydrate_module_family(
            normalized_state,
            allowed_roots=SHAPEVAE_UPSTREAM_ROOTS,
            required_roots=SHAPEVAE_REQUIRED_ROOTS,
            strict=strict,
        )
        module_roots = sorted({name.split('.', 1)[0] for name in hydrated_state if isinstance(name, str) and name.strip()})
        self.checkpoint_state = {
            'state_dict': hydrated_state,
            'state_dict_metadata': {
                'parameter_count': len(hydrated_state),
                'module_roots': module_roots,
                'module_family': self.__class__.__name__,
            },
            'representation': 'module-state-dict-v2',
        }
        self.state_dict = hydrated_state
        self.hydrated = True
        return {'missing_keys': missing_roots, 'unexpected_keys': unexpected_keys, 'strict': strict}

    def decode(self, latents, voxel_idx=None):
        del voxel_idx
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_profile = _group_mean(checkpoint_tokens(checkpoint_reference, limit=16), 4)
        latent_tokens = [list(map(float, token)) for token in latents if isinstance(token, list)] or [[0.0, 0.0, 0.0, 0.0]]
        decoded_tokens: list[list[float]] = []
        for token_index, token in enumerate(latent_tokens):
            checkpoint_value = checkpoint_profile[token_index % len(checkpoint_profile)] if checkpoint_profile else 0.0
            decoded_tokens.append(
                [
                    clamp_unit((float(token[channel]) * 0.7) + (checkpoint_value * 0.3))
                    for channel in range(min(len(token), 4))
                ]
            )
        return decoded_tokens

    def query(self, latents, queries, voxel_idx=None):
        del voxel_idx
        latent_tokens = [list(map(float, token)) for token in latents if isinstance(token, list)] or [[0.0, 0.0, 0.0, 0.0]]
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_profile = _group_mean(checkpoint_tokens(checkpoint_reference, limit=16), 4)
        latent_profile = _group_mean([value for token in latent_tokens for value in token], 4)
        logits: list[float] = []
        for index, query in enumerate(queries):
            features = _fourier_features(list(query) if isinstance(query, (list, tuple)) else [0.0, 0.0, 0.0])
            token = latent_tokens[index % len(latent_tokens)]
            value = sum(
                (features[channel] * 0.35)
                + (token[channel % len(token)] * 0.45)
                + ((checkpoint_profile[channel] if checkpoint_profile else 0.0) * 0.2)
                for channel in range(4)
            ) / 4.0
            logits.append(round((clamp_unit(value) - 0.5) * 2.0, 6))
        return logits

    def decode_latents(
        self,
        latents: dict[str, object],
        reference_asset: dict[str, object],
        conditioning: dict[str, object] | None = None,
        coarse_surface: dict[str, object] | None = None,
    ) -> dict[str, object]:
        latent_values = _numeric_sequence(latents.get('latents'))
        latent_tokens = _chunk_tokens(latent_values)
        decoded_tokens = self.decode(latent_tokens)
        reference_tensor_values = _flatten_numeric(reference_asset.get('image_tensor'))
        conditioning_context = _numeric_sequence(conditioning.get('context')) if isinstance(conditioning, dict) else []
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxels'), dict) else {}
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('mesh'), dict) else {}
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        resolution = _voxel_resolution(voxel_cond)
        latent_profile = _group_mean(latent_values, 4)
        image_profile = _group_mean(reference_tensor_values, 4)
        context_profile = _group_mean(conditioning_context, 4)
        checkpoint_profile = _group_mean(checkpoint_tokens(checkpoint_reference, limit=16), 4)

        coords = _voxel_coords(voxel_cond)
        if not coords:
            coords = [
                (x_axis, y_axis, z_axis)
                for z_axis in range(resolution)
                for y_axis in range(resolution)
                for x_axis in range(resolution)
            ]

        field_values: list[float] = []
        field_grid: list[list[list[float]]] = []
        scale = float(max(resolution - 1, 1))
        for z_axis in range(resolution):
            plane: list[list[float]] = []
            for y_axis in range(resolution):
                row: list[float] = []
                for x_axis in range(resolution):
                    query = [float(x_axis) / scale, float(y_axis) / scale, float(z_axis) / scale]
                    query_value = self.query(decoded_tokens, [query])[0]
                    profile_channel = (x_axis + y_axis + z_axis) % 4
                    combined = clamp_unit(
                        ((query_value + 1.0) / 2.0 * 0.45)
                        + (latent_profile[profile_channel] * 0.2)
                        + (image_profile[profile_channel] * 0.1)
                        + (context_profile[profile_channel] * 0.15)
                        + (checkpoint_profile[profile_channel] * 0.1)
                    )
                    signed = round((combined - 0.5) * 2.0, 6)
                    row.append(signed)
                    field_values.append(signed)
                plane.append(row)
            field_grid.append(plane)

        field_signature = stable_signature(field_values)
        occupancy_field = [clamp_unit((value + 1.0) / 2.0) for value in field_values]
        return {
            'vae': self.__class__.__name__,
            'authority': 'latent-query-grid',
            'decoded_latents': latent_values,
            'decoded_tokens': decoded_tokens,
            'field_logits': field_values,
            'occupancy_field': occupancy_field,
            'field_grid': field_grid,
            'field_grid_shape': [resolution, resolution, resolution],
            'field_value_count': len(field_values),
            'field_resolution': resolution,
            'field_signature': field_signature,
            'field_grid_signature': field_signature,
            'occupied_grid_cells': sum(1 for value in occupancy_field if value >= 0.5),
            'decoded_signature': stable_signature(latent_values),
            'checkpoint_signature': checkpoint_state_signature,
            'spatial_context': {
                'voxel_coords': list(voxel_cond.get('coords')) if isinstance(voxel_cond.get('coords'), list) else list(voxels.get('voxel_coords')) if isinstance(voxels.get('voxel_coords'), list) else [],
                'voxel_values': _numeric_sequence(voxel_cond.get('occupancies')) or _numeric_sequence(voxels.get('voxel_values')),
                'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else int(voxels.get('voxel_count', 0)) if isinstance(voxels.get('voxel_count'), int) else len(coords),
                'resolution': resolution,
                'voxel_signature': int(voxel_cond.get('signature', 0)) if isinstance(voxel_cond.get('signature'), int) else int(voxels.get('voxel_signature', 0)) if isinstance(voxels.get('voxel_signature'), int) else 0,
                'bounds': voxel_cond.get('bounds') if isinstance(voxel_cond.get('bounds'), dict) else voxels.get('bounds') if isinstance(voxels.get('bounds'), dict) else {},
                'mesh_bounds': mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {},
                'mesh_signature': int(mesh.get('signature', 0)) if isinstance(mesh.get('signature'), int) else 0,
                'reference_signature': int(reference_asset.get('signature', 0)) if isinstance(reference_asset.get('signature'), int) else 0,
                'conditioning_signature': int(conditioning.get('conditioning_signature', 0)) if isinstance(conditioning, dict) and isinstance(conditioning.get('conditioning_signature'), int) else 0,
            },
            'evidence': {
                'decoded_count': len(latent_values),
                'field_value_count': len(field_values),
                'field_signature': field_signature,
            },
            'state_hydrated': self.hydrated,
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
