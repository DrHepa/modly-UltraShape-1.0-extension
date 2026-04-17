"""Surface loader seam for mc-only local refinement."""

from __future__ import annotations

from pathlib import Path

from .utils.tensors import bytes_to_unit_floats, stable_signature
from .utils.voxelize import sample_surface_points, voxelize_from_point


GLB_MAGIC = b'glTF'


class SurfaceLoadError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def _read_glb_payload(surface_path: Path) -> dict[str, object]:
    payload = surface_path.read_bytes()
    is_binary_glb = len(payload) >= 12 and payload[:4] == GLB_MAGIC
    surface_tokens = bytes_to_unit_floats(payload, length=16)
    surface_points = sample_surface_points({'bytes': payload})

    return {
        'kind': 'glb-bytes',
        'path': str(surface_path),
        'bytes': payload,
        'byte_length': len(payload),
        'is_binary_glb': is_binary_glb,
        'tokens': surface_tokens,
        'signature': stable_signature(surface_tokens),
        'surface_points': surface_points,
        'surface_point_count': len(surface_points),
    }


class SharpEdgeSurfaceLoader:
    def load(self, path: str) -> dict[str, object]:
        surface_path = Path(path)
        if not surface_path.is_file():
            raise SurfaceLoadError(f'coarse_mesh is not readable: {path}.')

        suffix = surface_path.suffix.lower()
        if suffix != '.glb':
            raise SurfaceLoadError(
                f'UltraShape local runner accepts only coarse .glb meshes in this MVP, received {suffix or "<none>"}.'
            )

        mesh = _read_glb_payload(surface_path)
        voxels = voxelize_from_point(mesh)
        return {
            'path': str(surface_path),
            'suffix': suffix,
            'loader': self.__class__.__name__,
            'mesh': mesh,
            'voxels': voxels,
        }


def load_coarse_surface(path: str) -> dict[str, object]:
    return SharpEdgeSurfaceLoader().load(path)
