"""Attention block helpers for the vendored autoencoder subset."""

from __future__ import annotations


def attention_backend(prefer_flash: bool = False) -> str:
    return 'flash_attn' if prefer_flash else 'sdpa'


def attention_heads(hidden_size: int, *, head_dim: int = 64) -> int:
    if hidden_size <= 0:
        return 1
    return max(hidden_size // max(head_dim, 1), 1)
