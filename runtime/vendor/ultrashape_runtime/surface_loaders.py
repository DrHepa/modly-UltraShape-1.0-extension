"""Surface loader seam for mc-only local refinement."""

from __future__ import annotations

from pathlib import Path


GLB_MAGIC = b'glTF'


class SurfaceLoadError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def _read_glb_payload(surface_path: Path) -> dict[str, object]:
    payload = surface_path.read_bytes()
    is_binary_glb = len(payload) >= 12 and payload[:4] == GLB_MAGIC

    return {
        'kind': 'glb-bytes',
        'path': str(surface_path),
        'bytes': payload,
        'byte_length': len(payload),
        'is_binary_glb': is_binary_glb,
    }


def load_coarse_surface(path: str) -> dict[str, object]:
    surface_path = Path(path)
    if not surface_path.is_file():
        raise SurfaceLoadError(f'coarse_mesh is not readable: {path}.')

    suffix = surface_path.suffix.lower()
    if suffix != '.glb':
        raise SurfaceLoadError(f'UltraShape local runner accepts only coarse .glb meshes in this MVP, received {suffix or "<none>"}.')

    return {
        'path': str(surface_path),
        'suffix': suffix,
        'mesh': _read_glb_payload(surface_path),
    }
