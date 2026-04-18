"""Volume decoder subset for the vendored runtime."""

from __future__ import annotations

from ...utils.tensors import clamp_unit, stable_signature

CORNER_DIRECTIONS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)


def _numeric_values(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def _voxel_coords(candidate: object) -> list[tuple[int, int, int]]:
    if not isinstance(candidate, list):
        return []

    coords: list[tuple[int, int, int]] = []
    for item in candidate:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        coords.append((int(item[0]), int(item[1]), int(item[2])))
    return coords


def _dense_field(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * 64

    dense_field: list[float] = []
    for index in range(64):
        value = values[index % len(values)]
        dense_field.append(clamp_unit((value * 0.88) + (((index % 9) + 1) / 180.0)))
    return dense_field


def _grid_coords(voxel_coords: list[tuple[int, int, int]], resolution: int) -> list[tuple[int, int, int]]:
    if not voxel_coords:
        return []

    xs = [coord[0] for coord in voxel_coords]
    ys = [coord[1] for coord in voxel_coords]
    zs = [coord[2] for coord in voxel_coords]
    max_index = max(int(resolution), 1)
    min_x = max(min(xs), 0)
    max_x = min(max(xs), max_index)
    min_y = max(min(ys), 0)
    max_y = min(max(ys), max_index)
    min_z = max(min(zs), 0)
    max_z = min(max(zs), max_index)

    coords: list[tuple[int, int, int]] = []
    for z_axis in range(min_z, max_z + 1):
        for y_axis in range(min_y, max_y + 1):
            for x_axis in range(min_x, max_x + 1):
                coords.append((x_axis, y_axis, z_axis))
    return coords


def _mc_corners(
    values: list[float],
    coords: list[tuple[int, int, int]],
    occupied_voxels: set[tuple[int, int, int]],
) -> list[tuple[float, float, float, float, float, float, float, float]]:
    if not values or not coords:
        return []

    corners: list[tuple[float, float, float, float, float, float, float, float]] = []
    for cell_index, coord in enumerate(coords):
        occupancy_bias = 0.18 if coord in occupied_voxels else -0.08
        local_value = values[cell_index % len(values)]
        signed_corner_values: list[float] = []
        for corner_index, direction in enumerate(CORNER_DIRECTIONS):
            raw_value = values[(cell_index + (corner_index * 7)) % len(values)]
            directional_bias = (sum(direction) / 3.0) * 0.11
            signed_value = round(
                ((raw_value - 0.5) * 1.45)
                + ((local_value - 0.5) * 0.35)
                + occupancy_bias
                + directional_bias,
                6,
            )
            signed_corner_values.append(signed_value)

        if all(value <= 0.0 for value in signed_corner_values):
            signed_corner_values[-1] = abs(signed_corner_values[-1]) + 0.14
        elif all(value >= 0.0 for value in signed_corner_values):
            signed_corner_values[0] = -(abs(signed_corner_values[0]) + 0.14)

        corners.append(tuple(signed_corner_values[:8]))
    return corners


class VanillaVDMVolumeDecoding:
    def decode(self, decoded_latents: dict[str, object]) -> dict[str, object]:
        values = _numeric_values(decoded_latents.get('volume_tokens'))
        if not values:
            values = _numeric_values(decoded_latents.get('mesh_seed'))
        if not values:
            values = _numeric_values(decoded_latents.get('decoded_latents'))

        spatial_context = decoded_latents.get('spatial_context') if isinstance(decoded_latents.get('spatial_context'), dict) else {}
        voxel_coords = _voxel_coords(spatial_context.get('voxel_coords'))
        resolution = int(spatial_context.get('resolution', 0)) if isinstance(spatial_context.get('resolution'), int) else 0
        dense_field = _dense_field(values)
        coords = _grid_coords(voxel_coords, resolution)
        if not coords:
            fallback_resolution = max(min(len(dense_field) // 4, 4), 2)
            coords = [(x_axis, y_axis, z_axis) for z_axis in range(fallback_resolution) for y_axis in range(fallback_resolution) for x_axis in range(fallback_resolution)]
        corners = _mc_corners(dense_field, coords, set(voxel_coords))
        flattened_coords = [float(axis) / max(int(resolution) or 1, 1) for point in coords for axis in point]
        return {
            'decoder': self.__class__.__name__,
            'coords': coords,
            'corners': corners,
            'iso': 0.0,
            'field_density': clamp_unit(sum(dense_field) / len(dense_field) if dense_field else 0.0),
            'field_signature': stable_signature(dense_field + flattened_coords[:64]),
            'mesh_signature': stable_signature(flattened_coords[:128]),
            'cell_count': len(coords),
            'grid_resolution': max(int(resolution), 1),
        }


class FlashVDMVolumeDecoding(VanillaVDMVolumeDecoding):
    pass


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    return VanillaVDMVolumeDecoding().decode(decoded_latents)
