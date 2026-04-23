"""Surface loader aligned with upstream coarse-mesh normalization."""

from __future__ import annotations

from pathlib import Path

from .utils import stable_signature
from .utils.voxelize import mesh_geometry_from_glb, sample_surface_points, voxelize_from_point


GLB_MAGIC = b'glTF'
DEFAULT_NORMALIZE_SCALE = 0.9999


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


def _bounds(vertices: list[tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
    if not vertices:
        zero = (0.0, 0.0, 0.0)
        return {'min': zero, 'max': zero, 'extents': zero, 'center': zero}

    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    minimum = (min(xs), min(ys), min(zs))
    maximum = (max(xs), max(ys), max(zs))
    extents = tuple(round(maximum[index] - minimum[index], 6) for index in range(3))
    center = tuple(round((minimum[index] + maximum[index]) / 2.0, 6) for index in range(3))
    return {'min': minimum, 'max': maximum, 'extents': extents, 'center': center}


def _normalize_vertices(
    vertices: list[tuple[float, float, float]], *, normalize_scale: float = DEFAULT_NORMALIZE_SCALE
) -> tuple[list[tuple[float, float, float]], dict[str, object]]:
    bounds = _bounds(vertices)
    center = bounds['center']
    max_extent = max(*(float(axis) for axis in bounds['extents']), 1e-6)
    scale_factor = round((2.0 * float(normalize_scale)) / max_extent, 6)
    normalized = [
        (
            round((vertex[0] - center[0]) * scale_factor, 6),
            round((vertex[1] - center[1]) * scale_factor, 6),
            round((vertex[2] - center[2]) * scale_factor, 6),
        )
        for vertex in vertices
    ]
    return normalized, {
        'center': [round(float(axis), 6) for axis in center],
        'max_extent': round(float(max_extent), 6),
        'scale_factor': scale_factor,
        'normalize_scale': round(float(normalize_scale), 6),
    }


def _mesh_state(
    *,
    path: Path,
    payload: bytes,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    sampled_surface_points: list[tuple[float, float, float]],
    kind: str,
    source_signature: int,
    is_binary_glb: bool,
) -> dict[str, object]:
    bounds = _bounds(vertices)
    return _CompatState(
        {
            'kind': kind,
            'path': str(path),
            'bytes': payload,
            'byte_length': len(payload),
            'is_binary_glb': is_binary_glb,
            'vertices': vertices,
            'faces': faces,
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'bounds': bounds,
            'extents': bounds['extents'],
            'sampled_surface_points': sampled_surface_points,
            'surface_point_count': len(sampled_surface_points),
            'evidence': {
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'surface_point_count': len(sampled_surface_points),
            },
        },
        compat={
            'tokens': lambda state: [axis for vertex in state['vertices'] for axis in vertex[:3]][:24],
            'signature': lambda state: source_signature,
            'surface_points': lambda state: list(state['sampled_surface_points']),
        },
    )


def _read_glb_payload(surface_path: Path, *, normalize_scale: float = DEFAULT_NORMALIZE_SCALE) -> dict[str, object]:
    payload = surface_path.read_bytes()
    is_binary_glb = len(payload) >= 12 and payload[:4] == GLB_MAGIC
    geometry = mesh_geometry_from_glb(payload)
    vertices = geometry.get('vertices') if isinstance(geometry.get('vertices'), list) else []
    faces = geometry.get('faces') if isinstance(geometry.get('faces'), list) else []
    if not vertices or not faces:
        raise SurfaceLoadError(f'coarse_mesh must be a readable binary glb with parseable geometry: {surface_path}.')

    original_vertices = [tuple(float(axis) for axis in vertex[:3]) for vertex in vertices]
    original_faces = [tuple(int(index) for index in face[:3]) for face in faces]
    original_points = sample_surface_points({'bytes': payload, 'vertices': original_vertices, 'faces': original_faces})
    source_signature = geometry.get('signature') if isinstance(geometry.get('signature'), int) else stable_signature([])

    normalized_vertices, normalization_transform = _normalize_vertices(original_vertices, normalize_scale=normalize_scale)
    normalized_points = sample_surface_points({'vertices': normalized_vertices, 'faces': original_faces})

    original_mesh = _mesh_state(
        path=surface_path,
        payload=payload,
        vertices=original_vertices,
        faces=original_faces,
        sampled_surface_points=original_points,
        kind='coarse-glb-mesh',
        source_signature=source_signature,
        is_binary_glb=is_binary_glb,
    )
    normalized_mesh = _mesh_state(
        path=surface_path,
        payload=payload,
        vertices=normalized_vertices,
        faces=original_faces,
        sampled_surface_points=normalized_points,
        kind='normalized-coarse-glb-mesh',
        source_signature=source_signature,
        is_binary_glb=is_binary_glb,
    )

    return {
        'path': str(surface_path),
        'byte_length': len(payload),
        'is_binary_glb': is_binary_glb,
        'mesh': normalized_mesh,
        'original_mesh': original_mesh,
        'normalization_transform': normalization_transform,
    }


class SharpEdgeSurfaceLoader:
    def __init__(self, normalize_scale: float = DEFAULT_NORMALIZE_SCALE):
        self.normalize_scale = normalize_scale

    def load(self, path: str) -> dict[str, object]:
        surface_path = Path(path)
        if not surface_path.is_file():
            raise SurfaceLoadError(f'coarse_mesh is not readable: {path}.')

        suffix = surface_path.suffix.lower()
        if suffix != '.glb':
            raise SurfaceLoadError(
                f'UltraShape local runner accepts only coarse .glb meshes in this MVP, received {suffix or "<none>"}.'
            )

        surface_state = _read_glb_payload(surface_path, normalize_scale=self.normalize_scale)
        mesh = surface_state['mesh']
        voxel_cond = voxelize_from_point(mesh)
        return _CompatState(
            {
                'path': str(surface_path),
                'suffix': suffix,
                'loader': self.__class__.__name__,
                'mesh': mesh,
                'original_mesh': surface_state['original_mesh'],
                'normalization_transform': surface_state['normalization_transform'],
                'sampled_surface_points': mesh['sampled_surface_points'],
                'voxel_cond': voxel_cond,
            },
            compat={
                'voxels': lambda state: state['voxel_cond'],
            },
        )


def load_coarse_surface(path: str) -> dict[str, object]:
    return SharpEdgeSurfaceLoader().load(path)
