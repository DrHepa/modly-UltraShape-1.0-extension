"""Background-removal seam for the vendored UltraShape runtime."""

from __future__ import annotations

import struct
import zlib

try:  # pragma: no cover - optional dependency seam
    import rembg as _rembg  # type: ignore
except ImportError:  # pragma: no cover - expected in tests
    _rembg = None


def rembg_available() -> bool:
    return _rembg is not None


def payload_has_cutout_alpha(payload: bytes) -> bool:
    if len(payload) < 8 or payload[:8] != b'\x89PNG\r\n\x1a\n':
        return False

    offset = 8
    color_type = None
    idat_chunks: list[bytes] = []
    while offset + 8 <= len(payload):
        length = struct.unpack_from('>I', payload, offset)[0]
        offset += 4
        chunk_type = payload[offset : offset + 4]
        offset += 4
        chunk_data = payload[offset : offset + length]
        offset += length + 4
        if chunk_type == b'IHDR' and len(chunk_data) >= 10:
            color_type = chunk_data[9]
        elif chunk_type == b'IDAT':
            idat_chunks.append(chunk_data)
        elif chunk_type == b'IEND':
            break

    if color_type != 6 or not idat_chunks:
        return False

    try:
        decoded = zlib.decompress(b''.join(idat_chunks))
    except zlib.error:
        return False

    stride = 2 * 4 + 1
    for offset in range(0, len(decoded), stride):
        row = decoded[offset : offset + stride]
        if len(row) < stride:
            break
        for alpha_index in range(4, len(row), 4):
            if row[alpha_index] < 255:
                return True
    return False


def maybe_apply_cutout(payload: bytes, *, require_cutout: object) -> tuple[bytes, bool]:
    if require_cutout in (None, False, 'never'):
        return payload, False
    if payload_has_cutout_alpha(payload):
        return payload, False
    if require_cutout == 'conditional':
        return payload, False
    if not rembg_available():
        raise RuntimeError('rembg is required for cutout preprocessing but is unavailable.')
    remover = getattr(_rembg, 'remove', None)
    if not callable(remover):
        raise RuntimeError('rembg.remove is unavailable for cutout preprocessing.')
    result = remover(payload)
    if not isinstance(result, (bytes, bytearray)):
        raise RuntimeError('rembg.remove returned an unsupported payload type.')
    return bytes(result), True
