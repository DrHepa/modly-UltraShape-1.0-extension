"""Surface loader seam for mc-only local refinement."""

from __future__ import annotations

from pathlib import Path

from .utils import stable_signature
from .utils.voxelize import mesh_geometry_from_glb, sample_surface_points, voxelize_from_point


GLB_MAGIC = b'glTF'


class _CompatState(dict):
    def __init__(self, *args, compat: dict[str, object] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compat = compat or {}

    def _resolve(self, key: str):
        value = self._compat[key]
        return value(self) if callable(value) else value

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if key in self._compat:
            return self._resolve(key)
        return default

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._compat:
            return self._resolve(key)
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self._compat


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

    return _CompatState({
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
        'sampled_surface_points': surface_points,
        'surface_point_count': len(surface_points),
        'evidence': {
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'surface_point_count': len(surface_points),
        },
    }, compat={
        'tokens': lambda state: list(geometry_tokens),
        'signature': lambda state: surface_signature,
        'surface_points': lambda state: list(state['sampled_surface_points']),
    })


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
        voxel_cond = voxelize_from_point(mesh)
        return _CompatState({
            'path': str(surface_path),
            'suffix': suffix,
            'loader': self.__class__.__name__,
            'mesh': mesh,
            'sampled_surface_points': mesh['sampled_surface_points'],
            'voxel_cond': voxel_cond,
        }, compat={
            'voxels': lambda state: state['voxel_cond'],
        })


def load_coarse_surface(path: str) -> dict[str, object]:
    return SharpEdgeSurfaceLoader().load(path)
