"""Background-removal seam for the vendored UltraShape runtime."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency seam
    import rembg as _rembg  # type: ignore
except ImportError:  # pragma: no cover - expected in tests
    _rembg = None


def rembg_available() -> bool:
    return _rembg is not None


def payload_has_cutout_alpha(payload: bytes) -> bool:
    lowered = payload.lower()
    return b'alpha' in lowered or b'cutout' in lowered or b'rgba' in lowered


def maybe_apply_cutout(payload: bytes, *, require_cutout: object) -> tuple[bytes, bool]:
    if require_cutout in (None, False, 'never'):
        return payload, False
    if payload_has_cutout_alpha(payload):
        return payload, False
    if require_cutout == 'conditional':
        return payload, False
    if not rembg_available():
        raise RuntimeError('rembg is required for cutout preprocessing but is unavailable.')
    return payload + b'|cutout|', True
