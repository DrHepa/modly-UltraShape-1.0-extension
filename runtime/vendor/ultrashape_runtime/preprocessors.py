"""Reference asset preprocessing for the local mc-only runner."""

from __future__ import annotations

from pathlib import Path

from .rembg import maybe_apply_cutout, payload_has_cutout_alpha
from .utils.tensors import bytes_to_unit_floats, clamp_unit, stable_signature


class ReferencePreprocessError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def normalize_reference_asset(path: str) -> dict[str, object]:
    asset_path = Path(path)
    if not asset_path.is_file():
        raise ReferencePreprocessError(f'reference_image is not readable: {path}.')

    content = asset_path.read_bytes()
    processed_bytes, cutout_applied = maybe_apply_cutout(content, require_cutout='conditional')
    normalized = bytes_to_unit_floats(processed_bytes, length=16)

    return {
        'path': str(asset_path),
        'byte_length': len(processed_bytes),
        'has_content': bool(processed_bytes),
        'normalized_channels': 4,
        'tokens': normalized,
        'signature': stable_signature(normalized),
        'mean_intensity': clamp_unit(sum(normalized) / len(normalized) if normalized else 0.0),
        'cutout_applied': cutout_applied,
        'had_cutout_alpha': payload_has_cutout_alpha(content),
    }
