"""Attention processor helpers for the vendored autoencoder subset."""

from __future__ import annotations

from .attention_blocks import attention_backend


def processor_name(prefer_flash: bool = False) -> str:
    return f'{attention_backend(prefer_flash)}-processor'
