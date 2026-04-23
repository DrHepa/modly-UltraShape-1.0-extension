"""Utility exports for the vendored UltraShape runtime subset."""


def bytes_to_unit_floats(payload: bytes, *, length: int = 16) -> list[float]:
    if length <= 0:
        return []
    if not payload:
        return [0.0] * length

    values = list(payload[:length])
    while len(values) < length:
        values.extend(values[: length - len(values)])

    return [round(value / 255.0, 6) for value in values[:length]]


def stable_signature(values: list[float]) -> int:
    accumulator = 0
    for index, value in enumerate(values, start=1):
        accumulator += int(round(value * 1000)) * index
    return accumulator


def blend_sequences(*sequences: list[float]) -> list[float]:
    usable = [sequence for sequence in sequences if sequence]
    if not usable:
        return []

    length = max(len(sequence) for sequence in usable)
    blended: list[float] = []
    for index in range(length):
        total = 0.0
        count = 0
        for sequence in usable:
            total += sequence[index % len(sequence)]
            count += 1
        blended.append(round(total / count, 6))
    return blended


def clamp_unit(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


from .checkpoint import expected_checkpoint_name, expected_checkpoint_subtrees
from .misc import get_obj_from_str, instantiate_from_config, load_omega_config


__all__ = [
    'blend_sequences',
    'bytes_to_unit_floats',
    'clamp_unit',
    'expected_checkpoint_name',
    'expected_checkpoint_subtrees',
    'get_obj_from_str',
    'instantiate_from_config',
    'load_omega_config',
    'stable_signature',
]
