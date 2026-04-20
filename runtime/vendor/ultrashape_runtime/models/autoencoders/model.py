"""Autoencoder model subset for local runtime wiring."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_signature, checkpoint_tokens
from ...utils import clamp_unit, stable_signature


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


def _voxel_resolution(voxel_cond: dict[str, object]) -> int:
    resolution = voxel_cond.get('resolution') if isinstance(voxel_cond, dict) else 0
    if isinstance(resolution, int) and resolution > 1:
        return resolution
    return 4


def _voxel_occupancy_map(voxel_cond: dict[str, object]) -> dict[tuple[int, int, int], float]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    occupancies = _numeric_sequence(voxel_cond.get('occupancies'))
    occupancy_map: dict[tuple[int, int, int], float] = {}
    for index, coord in enumerate(coords):
        if not isinstance(coord, (list, tuple)) or len(coord) < 3:
            continue
        occupancy_map[(int(coord[0]), int(coord[1]), int(coord[2]))] = clamp_unit(occupancies[index] if index < len(occupancies) else 0.0)
    return occupancy_map


def _build_field_grid(
    *,
    resolution: int,
    latent_profile: list[float],
    image_profile: list[float],
    context_profile: list[float],
    checkpoint_profile: list[float],
    occupancy_map: dict[tuple[int, int, int], float],
) -> list[list[list[float]]]:
    grid: list[list[list[float]]] = []
    scale = float(max(resolution - 1, 1))
    center = scale / 2.0

    for z_axis in range(resolution):
        plane: list[list[float]] = []
        for y_axis in range(resolution):
            row: list[float] = []
            for x_axis in range(resolution):
                channel = (x_axis + y_axis + z_axis) % 4
                normalized_x = x_axis / scale
                normalized_y = y_axis / scale
                normalized_z = z_axis / scale
                occupancy = occupancy_map.get((x_axis, y_axis, z_axis), 0.0)
                radial = clamp_unit(1.0 - (abs(x_axis - center) + abs(y_axis - center) + abs(z_axis - center)) / max(scale * 1.5, 1.0))
                value = clamp_unit(
                    (latent_profile[channel] * 0.3)
                    + (image_profile[channel] * 0.15)
                    + (context_profile[channel] * 0.2)
                    + (checkpoint_profile[channel] * 0.15)
                    + (occupancy * 0.15)
                    + (radial * 0.05)
                )
                signed = round((value - 0.5) * 2.0 + ((normalized_x - normalized_y + normalized_z) * 0.1), 6)
                row.append(signed)
            plane.append(row)
        grid.append(plane)
    return grid


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
        reference_tensor_values = _flatten_numeric(reference_asset.get('image_tensor'))
        conditioning_context = _numeric_sequence(conditioning.get('context')) if isinstance(conditioning, dict) else []
        voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxel_cond'), dict) else {}
        voxels = coarse_surface.get('voxels') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('voxels'), dict) else {}
        mesh = coarse_surface.get('mesh') if isinstance(coarse_surface, dict) and isinstance(coarse_surface.get('mesh'), dict) else {}
        checkpoint_reference = self.checkpoint_state if self.checkpoint_state is not None else self.state_dict
        checkpoint_signal = checkpoint_tokens(checkpoint_reference, limit=16)
        checkpoint_state_signature = checkpoint_signature(checkpoint_reference)
        resolution = _voxel_resolution(voxel_cond)
        occupancy_map = _voxel_occupancy_map(voxel_cond)
        latent_profile = _group_mean(latent_values, 4)
        image_profile = _group_mean(reference_tensor_values, 4)
        context_profile = _group_mean(conditioning_context, 4)
        checkpoint_profile = _group_mean(checkpoint_signal, 4)
        field_grid = _build_field_grid(
            resolution=resolution,
            latent_profile=latent_profile,
            image_profile=image_profile,
            context_profile=context_profile,
            checkpoint_profile=checkpoint_profile,
            occupancy_map=occupancy_map,
        )
        field_logits = [value for plane in field_grid for row in plane for value in row]
        occupancy_field = [clamp_unit((value + 1.0) / 2.0) for value in field_logits]
        field_signature = stable_signature(field_logits)

        return {
            'vae': self.__class__.__name__,
            'authority': 'occupancy_field_grid',
            'decoded_latents': latent_values,
            'field_logits': field_logits,
            'occupancy_field': occupancy_field,
            'field_grid': field_grid,
            'field_grid_shape': [resolution, resolution, resolution],
            'field_value_count': len(field_logits),
            'field_resolution': resolution,
            'field_signature': field_signature,
            'field_grid_signature': field_signature,
            'occupied_grid_cells': sum(1 for value in occupancy_field if value >= 0.5),
            'decoded_signature': stable_signature(latent_values),
            'checkpoint_signature': checkpoint_state_signature,
            'spatial_context': {
                'voxel_coords': list(voxel_cond.get('coords')) if isinstance(voxel_cond.get('coords'), list) else list(voxels.get('voxel_coords')) if isinstance(voxels.get('voxel_coords'), list) else [],
                'voxel_values': _numeric_sequence(voxel_cond.get('occupancies')) or _numeric_sequence(voxels.get('voxel_values')),
                'voxel_count': int(voxel_cond.get('voxel_count', 0)) if isinstance(voxel_cond.get('voxel_count'), int) else int(voxels.get('voxel_count', 0)) if isinstance(voxels.get('voxel_count'), int) else len(occupancy_map),
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
                'field_value_count': len(field_logits),
                'field_signature': field_signature,
            },
            'state_hydrated': self.hydrated,
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
