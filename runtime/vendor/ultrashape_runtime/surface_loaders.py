"""Surface loader seam for mc-only local refinement."""

from __future__ import annotations

from pathlib import Path

from .utils.tensors import stable_signature
from .utils.voxelize import mesh_geometry_from_glb, sample_surface_points, voxelize_from_point


GLB_MAGIC = b'glTF'


class SurfaceLoadError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def _read_glb_payload(surface_path: Path) -> dict[str, object]:
    payload = surface_path.read_bytes()
    is_binary_glb = len(payload) >= 12 and payload[:4] == GLB_MAGIC
    geometry = mesh_geometry_from_glb(payload)
    vertices = geometry.get('vertices') if isinstance(geometry.get('vertices'), list) else []
    faces = geometry.get('faces') if isinstance(geometry.get('faces'), list) else []
    if not vertices or not faces:
        raise SurfaceLoadError(f'coarse_mesh must be a readable binary glb with parseable geometry: {surface_path}.')

    surface_points = sample_surface_points({'bytes': payload, 'vertices': vertices, 'faces': faces})
    bounds = geometry.get('bounds') if isinstance(geometry.get('bounds'), dict) else {}
    extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
    geometry_tokens = geometry.get('tokens') if isinstance(geometry.get('tokens'), list) else []
    surface_signature = geometry.get('signature') if isinstance(geometry.get('signature'), int) else stable_signature(geometry_tokens)

    return {
        'kind': 'coarse-glb-mesh',
        'path': str(surface_path),
        'bytes': payload,
        'byte_length': len(payload),
        'is_binary_glb': is_binary_glb,
        'vertices': vertices,
        'faces': faces,
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'bounds': bounds,
        'extents': extents,
        'tokens': geometry_tokens,
        'signature': surface_signature,
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
