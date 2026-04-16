"""Reference asset preprocessing for the local mc-only runner."""

from __future__ import annotations

from pathlib import Path


class ReferencePreprocessError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def normalize_reference_asset(path: str) -> dict[str, object]:
    asset_path = Path(path)
    if not asset_path.is_file():
        raise ReferencePreprocessError(f'reference_image is not readable: {path}.')

    content = asset_path.read_bytes()
    return {
        'path': str(asset_path),
        'byte_length': len(content),
        'has_content': bool(content),
    }
