"""Tiny MoE helpers for the vendored DiT subset."""

from __future__ import annotations


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


def moe_enabled() -> bool:
    return True
