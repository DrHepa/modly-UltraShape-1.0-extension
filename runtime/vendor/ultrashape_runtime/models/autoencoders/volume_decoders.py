"""Volume decoder subset for the vendored runtime."""

from __future__ import annotations

from ...utils.tensors import clamp_unit, stable_signature


def _dense_field(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * 27

    dense_field: list[float] = []
    for index in range(27):
        value = values[index % len(values)]
        dense_field.append(clamp_unit((value * 0.88) + (((index % 9) + 1) / 200.0)))
    return dense_field


def _mc_coords(values: list[float]) -> list[tuple[float, float, float]]:
    if not values:
        return []

    coords: list[tuple[float, float, float]] = []
    lattice = [
        (-1.0, -0.5, -0.5),
        (0.0, -0.5, -0.5),
        (1.0, -0.5, -0.5),
        (-1.0, 0.5, -0.5),
        (0.0, 0.5, -0.5),
        (1.0, 0.5, -0.5),
        (-1.0, -0.5, 0.5),
        (0.0, -0.5, 0.5),
        (1.0, -0.5, 0.5),
        (-1.0, 0.5, 0.5),
        (0.0, 0.5, 0.5),
        (1.0, 0.5, 0.5),
    ]
    for index, (base_x, base_y, base_z) in enumerate(lattice):
        coords.append(
            (
                round(base_x + ((values[index % len(values)] - 0.5) * 0.18), 6),
                round(base_y + ((values[(index + 4) % len(values)] - 0.5) * 0.18), 6),
                round(base_z + ((values[(index + 8) % len(values)] - 0.5) * 0.18), 6),
            )
        )
    return coords


def _mc_corners(values: list[float], cell_count: int) -> list[tuple[float, float, float, float, float, float, float, float]]:
    if not values or cell_count <= 0:
        return []

    corners: list[tuple[float, float, float, float, float, float, float, float]] = []
    for cell_index in range(cell_count):
        signed_corner_values: list[float] = []
        for corner_index in range(8):
            raw_value = values[(cell_index + corner_index) % len(values)]
            signed_value = round(raw_value - 0.5 + (((corner_index % 4) - 1.5) / 20.0), 6)
            signed_corner_values.append(signed_value)

        if all(value <= 0.0 for value in signed_corner_values):
            signed_corner_values[-1] = abs(signed_corner_values[-1]) + 0.1
        elif all(value >= 0.0 for value in signed_corner_values):
            signed_corner_values[0] = -(abs(signed_corner_values[0]) + 0.1)

        corners.append(tuple(signed_corner_values[:8]))
    return corners


class VanillaVDMVolumeDecoding:
    def decode(self, decoded_latents: dict[str, object]) -> dict[str, object]:
        values = decoded_latents.get('mesh_seed') if isinstance(decoded_latents.get('mesh_seed'), list) else []
        if not values:
            values = decoded_latents.get('decoded_latents') if isinstance(decoded_latents.get('decoded_latents'), list) else []

        dense_field = _dense_field([float(value) for value in values if isinstance(value, (int, float))])
        coords = _mc_coords(dense_field)
        corners = _mc_corners(dense_field, len(coords))
        return {
            'decoder': self.__class__.__name__,
            'coords': coords,
            'corners': corners,
            'iso': 0.0,
            'field_density': clamp_unit(sum(dense_field) / len(dense_field) if dense_field else 0.0),
            'field_signature': stable_signature(dense_field),
            'mesh_signature': stable_signature([axis for point in coords for axis in point]),
        }


class FlashVDMVolumeDecoding(VanillaVDMVolumeDecoding):
    pass


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    return VanillaVDMVolumeDecoding().decode(decoded_latents)
