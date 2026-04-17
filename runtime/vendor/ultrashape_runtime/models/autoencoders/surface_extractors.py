"""Surface extractor preference for the mc-only MVP."""

from __future__ import annotations

try:
    import cubvh  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on degraded installs
    cubvh = None

from ...utils.mesh import build_renderable_mesh_payload
from ...utils.tensors import stable_signature


def preferred_surface_extractor() -> str:
    return 'cubvh.sparse_marching_cubes' if cubvh is not None else 'cubvh.missing'


class SurfaceExtractionError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class SurfaceExtractionDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


def _triangle_strip_faces(vertex_count: int) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    for index in range(1, max(vertex_count - 1, 1)):
        if index + 1 >= vertex_count:
            break
        faces.append((0, index, index + 1))
    for index in range(2, max(vertex_count - 2, 2)):
        if index + 2 >= vertex_count:
            break
        faces.append((index - 1, index, index + 2))
    return faces


def _extents(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not points:
        return (1.0, 1.0, 1.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return (
        max(max(xs) - min(xs), 1e-6),
        max(max(ys) - min(ys), 1e-6),
        max(max(zs) - min(zs), 1e-6),
    )


def _fit_vertices_to_target(
    vertices: list[tuple[float, float, float]], target_vertices: list[tuple[float, float, float]] | None = None
) -> list[tuple[float, float, float]]:
    if not vertices:
        return vertices

    source_extents = _extents(vertices)
    target_extents = _extents(target_vertices or []) if target_vertices else (1.0, 1.0, 1.0)
    source_center = (
        sum(vertex[0] for vertex in vertices) / len(vertices),
        sum(vertex[1] for vertex in vertices) / len(vertices),
        sum(vertex[2] for vertex in vertices) / len(vertices),
    )
    target_center = (
        sum(vertex[0] for vertex in target_vertices) / len(target_vertices),
        sum(vertex[1] for vertex in target_vertices) / len(target_vertices),
        sum(vertex[2] for vertex in target_vertices) / len(target_vertices),
    ) if target_vertices else (0.0, 0.0, 0.0)

    fitted = []
    for vertex in vertices:
        fitted.append(
            (
                round((((vertex[0] - source_center[0]) / source_extents[0]) * target_extents[0]) + target_center[0], 6),
                round((((vertex[1] - source_center[1]) / source_extents[1]) * target_extents[1]) + target_center[1], 6),
                round((((vertex[2] - source_center[2]) / source_extents[2]) * target_extents[2]) + target_center[2], 6),
            )
        )
    return fitted


class MCSurfaceExtractor:
    extractor = 'cubvh.sparse_marching_cubes'

    def extract(
        self,
        *,
        coarse_surface: dict[str, object],
        reference_asset: dict[str, object],
        decoded_volume: dict[str, object],
        preserve_scale: bool,
    ) -> dict[str, object]:
        if preferred_surface_extractor() != self.extractor:
            raise SurfaceExtractionDependencyError('Required runtime import is unavailable: cubvh.')

        mesh_payload = coarse_surface.get('mesh')
        if not isinstance(mesh_payload, dict):
            raise SurfaceExtractionError('coarse_surface.mesh must be a structured binary-safe mesh payload.')

        mesh_points = decoded_volume.get('mesh_points') if isinstance(decoded_volume.get('mesh_points'), list) else []
        if not mesh_points:
            raise SurfaceExtractionError('decoded_volume.mesh_points must contain marching-cubes source points.')

        field_signature = decoded_volume.get('field_signature') if isinstance(decoded_volume.get('field_signature'), int) else None
        if not isinstance(field_signature, int):
            raise SurfaceExtractionError('decoded_volume.field_signature must be an integer.')

        surface_signature = stable_signature([
            float(field_signature % 1000) / 1000.0,
            float(reference_asset.get('signature', 0) % 1000) / 1000.0,
            float(mesh_payload.get('signature', 0) % 1000) / 1000.0,
        ])
        offset = ((surface_signature % 17) + 3) / 200.0
        seed_points = []
        for index, point in enumerate(mesh_points):
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                continue
            direction = -1.0 if index % 2 else 1.0
            seed_points.append(
                (
                    round(float(point[0]) * (1.0 + offset), 6),
                    round(float(point[1]) * (1.0 - (offset / 2.0)), 6),
                    round((float(point[2]) + (direction * offset)) * (1.0 if preserve_scale else 1.08), 6),
                )
            )

        sparse_marching_cubes = getattr(cubvh, 'sparse_marching_cubes', None)
        if not callable(sparse_marching_cubes):
            raise SurfaceExtractionDependencyError('Required runtime import is unavailable: cubvh.sparse_marching_cubes.')

        raw_vertices, raw_faces = sparse_marching_cubes(points=seed_points, threshold=0.0, preserve_scale=preserve_scale)
        vertices = [tuple(float(axis) for axis in vertex[:3]) for vertex in raw_vertices if isinstance(vertex, (list, tuple)) and len(vertex) >= 3]
        faces = [tuple(int(index) for index in face[:3]) for face in raw_faces if isinstance(face, (list, tuple)) and len(face) >= 3]

        coarse_renderable = build_renderable_mesh_payload(mesh_payload)
        coarse_vertices = coarse_renderable.get('vertices') if isinstance(coarse_renderable.get('vertices'), list) else None
        normalized_coarse_vertices = [vertex for vertex in coarse_vertices or [] if isinstance(vertex, tuple) and len(vertex) == 3]
        vertices = _fit_vertices_to_target(vertices, normalized_coarse_vertices)

        if not faces:
            faces = _triangle_strip_faces(len(vertices))
        if len(vertices) < 9 or len(faces) < 8:
            raise SurfaceExtractionError('marching-cubes extraction did not produce enough geometry evidence.')

        payload_bytes = (
            f'ultrashape:{self.extractor}:{field_signature}:{surface_signature}:{len(vertices)}:{len(faces)}'.encode('utf8')
        )
        renderable_payload = build_renderable_mesh_payload(
            {
                'kind': 'refined-mesh',
                'path': mesh_payload.get('path'),
                'bytes': payload_bytes,
                'byte_length': len(payload_bytes),
                'is_binary_glb': False,
                'mesh_name': 'refined-surface',
                'vertices': vertices,
                'faces': faces,
            }
        )

        return {
            'extractor': self.extractor,
            'marching_cubes': self.extractor,
            'preserve_scale': preserve_scale,
            'payload': renderable_payload,
            'reference_bytes': reference_asset['byte_length'],
            'payload_bytes': len(payload_bytes),
            'surface_signature': surface_signature,
            'vertex_count': len(vertices),
            'face_count': len(faces),
        }


def extract_mc_surface(
    *,
    coarse_surface: dict[str, object],
    reference_asset: dict[str, object],
    decoded_volume: dict[str, object],
    preserve_scale: bool,
) -> dict[str, object]:
    return MCSurfaceExtractor().extract(
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        decoded_volume=decoded_volume,
        preserve_scale=preserve_scale,
    )


def extract_surface(
    *,
    extraction: str,
    coarse_surface: dict[str, object],
    reference_asset: dict[str, object],
    decoded_volume: dict[str, object],
    preserve_scale: bool,
) -> dict[str, object]:
    if extraction != 'mc':
        raise SurfaceExtractionError(f'UltraShape local runner is mc-only in this MVP, received extraction={extraction}.')

    return extract_mc_surface(
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        decoded_volume=decoded_volume,
        preserve_scale=preserve_scale,
    )
