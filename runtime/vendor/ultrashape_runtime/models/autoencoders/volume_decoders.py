"""Portable upstream-shaped volume decoder subset for the vendored runtime."""

from __future__ import annotations

from ...utils import clamp_unit, stable_signature

CORNER_DIRECTIONS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)


def _numeric_values(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def _field_grid(decoded_latents: dict[str, object]) -> tuple[list[list[list[float]]], int]:
    raw_grid = decoded_latents.get('field_grid')
    if isinstance(raw_grid, list) and raw_grid and isinstance(raw_grid[0], list):
        resolution = len(raw_grid)
        normalized: list[list[list[float]]] = []
        for plane in raw_grid:
            plane_rows: list[list[float]] = []
            if not isinstance(plane, list):
                continue
            for row in plane:
                if not isinstance(row, list):
                    continue
                plane_rows.append([round(float(value), 6) for value in row if isinstance(value, (int, float))])
            if plane_rows:
                normalized.append(plane_rows)
        if normalized:
            return normalized, resolution

    values = _numeric_values(decoded_latents.get('field_logits')) or _numeric_values(decoded_latents.get('occupancy_field'))
    resolution = int(round(max(len(values), 1) ** (1 / 3)))
    resolution = max(resolution, 2)
    cursor = 0
    grid: list[list[list[float]]] = []
    for _z_axis in range(resolution):
        plane: list[list[float]] = []
        for _y_axis in range(resolution):
            row: list[float] = []
            for _x_axis in range(resolution):
                row.append(float(values[cursor % len(values)]) if values else 0.0)
                cursor += 1
            plane.append(row)
        grid.append(plane)
    return grid, resolution


def generate_dense_grid_points(*, bounds: float = 1.01, octree_resolution: int) -> tuple[list[list[list[tuple[float, float, float]]]], list[int], list[float]]:
    if isinstance(bounds, (int, float)):
        minimum = -float(bounds)
        maximum = float(bounds)
    else:
        minimum = -1.01
        maximum = 1.01
    steps = max(int(octree_resolution), 1)
    stride = (maximum - minimum) / steps
    points: list[list[list[tuple[float, float, float]]]] = []
    for z_axis in range(steps + 1):
        plane: list[list[tuple[float, float, float]]] = []
        for y_axis in range(steps + 1):
            row: list[tuple[float, float, float]] = []
            for x_axis in range(steps + 1):
                row.append((minimum + (x_axis * stride), minimum + (y_axis * stride), minimum + (z_axis * stride)))
            plane.append(row)
        points.append(plane)
    return points, [steps + 1, steps + 1, steps + 1], [maximum - minimum] * 3


def get_sparse_valid_voxels(grid_logits: object) -> tuple[list[tuple[int, int, int]], list[tuple[float, float, float, float, float, float, float, float]]]:
    if not isinstance(grid_logits, list) or not grid_logits or not isinstance(grid_logits[0], list):
        return [], []

    coords: list[tuple[int, int, int]] = []
    corners: list[tuple[float, float, float, float, float, float, float, float]] = []
    depth = len(grid_logits)
    height = len(grid_logits[0]) if depth else 0
    width = len(grid_logits[0][0]) if height and isinstance(grid_logits[0][0], list) else 0
    for z_axis in range(max(depth - 1, 0)):
        for y_axis in range(max(height - 1, 0)):
            for x_axis in range(max(width - 1, 0)):
                cube = []
                for dx, dy, dz in CORNER_DIRECTIONS:
                    cube.append(round(float(grid_logits[z_axis + dz][y_axis + dy][x_axis + dx]), 6))
                if min(cube) <= 0.0 <= max(cube):
                    coords.append((x_axis, y_axis, z_axis))
                    corners.append(tuple(cube[:8]))
    return coords, corners


def _occupancy_field(field_values: list[float]) -> list[float]:
    return [clamp_unit((value + 1.0) / 2.0) for value in field_values]


class VanillaVolumeDecoder:
    hierarchy = 'dense-grid'

    def decode(self, decoded_latents: dict[str, object]) -> dict[str, object]:
        field_grid, resolution = _field_grid(decoded_latents)
        field_values = [float(value) for plane in field_grid for row in plane for value in row]
        coords, corners = get_sparse_valid_voxels(field_grid)
        occupancy_field = _occupancy_field(field_values)
        mesh_signature = stable_signature([float(axis) / max(resolution - 1, 1) for point in coords for axis in point])
        return {
            'decoder': self.__class__.__name__,
            'authority': 'geo_decoder(query_grid)',
            'hierarchy': self.hierarchy,
            'grid_logits': field_grid,
            'field_grid': field_grid,
            'coords': coords,
            'corners': corners,
            'iso': 0.0,
            'field_density': clamp_unit(sum(occupancy_field) / len(occupancy_field) if occupancy_field else 0.0),
            'field_signature': int(decoded_latents.get('field_signature')) if isinstance(decoded_latents.get('field_signature'), int) else stable_signature(field_values),
            'field_grid_signature': int(decoded_latents.get('field_grid_signature')) if isinstance(decoded_latents.get('field_grid_signature'), int) else stable_signature(field_values),
            'field_grid_shape': [resolution, resolution, resolution],
            'field_value_count': len(field_values),
            'corner_signature': stable_signature([value for corner in corners for value in corner]),
            'mesh_signature': mesh_signature,
            'cell_count': len(coords),
            'corner_count': len(corners),
            'occupied_cell_count': len(coords),
            'occupied_grid_cells': sum(1 for value in occupancy_field if value >= 0.5),
            'grid_resolution': resolution,
        }


class HierarchicalVolumeDecoding(VanillaVolumeDecoder):
    hierarchy = 'octree-near-surface'


class VanillaVDMVolumeDecoding(VanillaVolumeDecoder):
    pass


class FlashVDMVolumeDecoding(HierarchicalVolumeDecoding):
    pass


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    return VanillaVolumeDecoder().decode(decoded_latents)
