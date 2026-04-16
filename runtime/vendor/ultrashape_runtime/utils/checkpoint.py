"""Checkpoint metadata helpers."""

from __future__ import annotations

from pathlib import Path


class CheckpointResolutionError(Exception):
    code = 'WEIGHTS_MISSING'


def expected_checkpoint_name() -> str:
    return 'ultrashape_v1.pt'


def resolve_checkpoint(checkpoint: str | None, primary_weight: str | None, ext_dir: str) -> str:
    if checkpoint not in (None, ''):
        resolved = Path(checkpoint)
    else:
        relative_weight = primary_weight or f'models/ultrashape/{expected_checkpoint_name()}'
        resolved = Path(ext_dir) / relative_weight

    if not resolved.is_file():
        raise CheckpointResolutionError(f'Required checkpoint is not readable: {resolved}.')

    return str(resolved)
