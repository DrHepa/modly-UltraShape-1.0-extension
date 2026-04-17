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


def _mesh_points(values: list[float]) -> list[tuple[float, float, float]]:
    if not values:
        return []

    points: list[tuple[float, float, float]] = []
    for index in range(0, min(len(values), 12)):
        x = (values[index % len(values)] * 2.0) - 1.0
        y = (values[(index + 4) % len(values)] * 2.0) - 1.0
        z = (values[(index + 8) % len(values)] * 2.0) - 1.0
        points.append((round(x, 6), round(y, 6), round(z, 6)))
    return points


class VanillaVDMVolumeDecoding:
    def decode(self, decoded_latents: dict[str, object]) -> dict[str, object]:
        values = decoded_latents.get('mesh_seed') if isinstance(decoded_latents.get('mesh_seed'), list) else []
        if not values:
            values = decoded_latents.get('decoded_latents') if isinstance(decoded_latents.get('decoded_latents'), list) else []

        dense_field = _dense_field([float(value) for value in values if isinstance(value, (int, float))])
        mesh_points = _mesh_points(dense_field)
        return {
            'decoder': self.__class__.__name__,
            'dense_field': dense_field,
            'mesh_points': mesh_points,
            'field_density': clamp_unit(sum(dense_field) / len(dense_field) if dense_field else 0.0),
            'field_signature': stable_signature(dense_field),
            'mesh_signature': stable_signature([axis for point in mesh_points for axis in point]),
        }


class FlashVDMVolumeDecoding(VanillaVDMVolumeDecoding):
    pass


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    return VanillaVDMVolumeDecoding().decode(decoded_latents)
