"""Volume decoder subset for the vendored runtime."""

from __future__ import annotations

from ...utils import clamp_unit, stable_signature

CORNER_DIRECTIONS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
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


def _field_grid(decoded_latents: dict[str, object]) -> tuple[list[list[list[float]]], int]:
    raw_grid = decoded_latents.get('field_grid')
    if isinstance(raw_grid, list) and raw_grid and isinstance(raw_grid[0], list):
        resolution = len(raw_grid)
        return raw_grid, resolution

    values = _numeric_values(decoded_latents.get('field_logits')) or _numeric_values(decoded_latents.get('occupancy_field'))
    resolution = int(round(max(len(values), 1) ** (1 / 3)))
    resolution = max(resolution, 2)
    cursor = 0
    grid: list[list[list[float]]] = []
    for z_axis in range(resolution):
        plane: list[list[float]] = []
        for y_axis in range(resolution):
            row: list[float] = []
            for x_axis in range(resolution):
                if values:
                    row.append(float(values[cursor % len(values)]))
                else:
                    row.append(0.0)
                cursor += 1
            plane.append(row)
        grid.append(plane)
    return grid, resolution


def _candidate_cells(voxel_coords: list[tuple[int, int, int]], resolution: int) -> list[tuple[int, int, int]]:
    max_index = max(resolution - 1, 1)
    candidates: set[tuple[int, int, int]] = set()

    for x_axis, y_axis, z_axis in voxel_coords:
        for dx in (0, -1):
            for dy in (0, -1):
                for dz in (0, -1):
                    candidate = (
                        min(max(x_axis + dx, 0), max_index - 1),
                        min(max(y_axis + dy, 0), max_index - 1),
                        min(max(z_axis + dz, 0), max_index - 1),
                    )
                    candidates.add(candidate)

    if candidates:
        return sorted(candidates)

    return [
        (x_axis, y_axis, z_axis)
        for z_axis in range(max_index)
        for y_axis in range(max_index)
        for x_axis in range(max_index)
    ]


def _corner_values(field_grid: list[list[list[float]]], coord: tuple[int, int, int]) -> tuple[float, float, float, float, float, float, float, float]:
    x_axis, y_axis, z_axis = coord
    values: list[float] = []
    for dx, dy, dz in CORNER_DIRECTIONS:
        values.append(round(float(field_grid[z_axis + dz][y_axis + dy][x_axis + dx]), 6))
    return tuple(values[:8])


class VanillaVDMVolumeDecoding:
    def decode(self, decoded_latents: dict[str, object]) -> dict[str, object]:
        field_grid, resolution = _field_grid(decoded_latents)
        spatial_context = decoded_latents.get('spatial_context') if isinstance(decoded_latents.get('spatial_context'), dict) else {}
        voxel_coords = _voxel_coords(spatial_context.get('voxel_coords'))
        coords = _candidate_cells(voxel_coords, resolution)
        corners = [_corner_values(field_grid, coord) for coord in coords]
        field_values = [value for plane in field_grid for row in plane for value in row]
        occupancy_field = [clamp_unit((value + 1.0) / 2.0) for value in field_values]
        flattened_coords = [float(axis) / max(resolution - 1, 1) for point in coords for axis in point]

        return {
            'decoder': self.__class__.__name__,
            'authority': 'occupancy_field_grid',
            'coords': coords,
            'corners': corners,
            'iso': 0.0,
            'field_density': clamp_unit(sum(occupancy_field) / len(occupancy_field) if occupancy_field else 0.0),
            'field_signature': int(decoded_latents.get('field_signature')) if isinstance(decoded_latents.get('field_signature'), int) else stable_signature(field_values),
            'field_grid_signature': int(decoded_latents.get('field_grid_signature')) if isinstance(decoded_latents.get('field_grid_signature'), int) else stable_signature(field_values),
            'field_grid_shape': [resolution, resolution, resolution],
            'field_value_count': len(field_values),
            'corner_signature': stable_signature([value for corner in corners for value in corner]),
            'mesh_signature': stable_signature(flattened_coords[:128]),
            'cell_count': len(coords),
            'corner_count': len(corners),
            'occupied_cell_count': len(voxel_coords),
            'occupied_grid_cells': sum(1 for value in occupancy_field if value >= 0.5),
            'grid_resolution': resolution,
        }


class FlashVDMVolumeDecoding(VanillaVDMVolumeDecoding):
    pass


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    return VanillaVDMVolumeDecoding().decode(decoded_latents)
