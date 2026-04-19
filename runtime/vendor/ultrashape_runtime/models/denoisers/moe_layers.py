"""Tiny MoE helpers for the vendored DiT subset."""

from __future__ import annotations


def _numeric_list(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def mix_expert_sequences(*sequences: list[float]) -> list[float]:
    usable = [sequence for sequence in sequences if sequence]
    if not usable:
        return []

    mixed: list[float] = []
    target_length = max(len(sequence) for sequence in usable)
    for index in range(target_length):
        total = 0.0
        weight_total = 0.0
        for expert_index, sequence in enumerate(usable, start=1):
            weight = float(expert_index)
            total += sequence[index % len(sequence)] * weight
            weight_total += weight
        mixed.append(round(total / max(weight_total, 1.0), 6))
    return mixed


def voxel_cond_signal(voxel_cond: dict[str, object], *, limit: int = 16) -> list[float]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    occupancies = _numeric_list(voxel_cond.get('occupancies'))
    resolution = int(voxel_cond.get('resolution', 0)) if isinstance(voxel_cond.get('resolution'), int) else 0
    scale = float(max(resolution, 1))
    signal: list[float] = []

    for index, coord in enumerate(coords[:limit]):
        if isinstance(coord, (list, tuple)):
            signal.extend(float(axis) / scale for axis in coord[:3] if isinstance(axis, (int, float)))
        if index < len(occupancies):
            signal.append(float(occupancies[index]))
        if len(signal) >= limit:
            break

    return [round(value, 6) for value in signal[:limit]]


def route_denoise_experts(
    *,
    latents: list[float],
    context: list[float],
    context_mask: list[float],
    voxel_cond: dict[str, object],
    checkpoint_signal: list[float],
    timestep: float,
) -> list[float]:
    timestep_signal = [float(timestep)] if latents or context or checkpoint_signal else []
    return mix_expert_sequences(
        latents,
        context,
        context_mask,
        voxel_cond_signal(voxel_cond),
        checkpoint_signal,
        timestep_signal,
    )


def moe_enabled() -> bool:
    return True
