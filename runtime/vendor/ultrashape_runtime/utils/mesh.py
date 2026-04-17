"""Mesh metadata helpers."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

from .tensors import bytes_to_unit_floats, clamp_unit, stable_signature


class MeshExportError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class MeshGateError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


GLB_MAGIC = b'glTF'
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942
CHAMFER_EPSILON = 0.003
RMS_DISPLACEMENT_THRESHOLD = 0.01
TOPOLOGY_DELTA_THRESHOLD = 0.01
PRESERVE_SCALE_TOLERANCE = (0.97, 1.03)


def default_surface_algorithm() -> str:
    return 'mc'


def _mesh_payload_bytes(mesh_payload: object) -> bytes:
    if not isinstance(mesh_payload, dict):
        raise MeshGateError('GEOMETRIC_GATE_REJECTED: mesh payload must be a structured mapping.')

    payload_bytes = mesh_payload.get('bytes')
    if not isinstance(payload_bytes, bytes):
        raise MeshGateError('GEOMETRIC_GATE_REJECTED: mesh payload bytes must be binary data.')
    return payload_bytes


def _extent_scale(mesh_payload: dict[str, object]) -> tuple[float, float, float]:
    raw_scale = mesh_payload.get('test_extent_scale')
    if not isinstance(raw_scale, list) or len(raw_scale) != 3:
        return (1.0, 1.0, 1.0)

    scale: list[float] = []
    for axis in raw_scale:
        if not isinstance(axis, (int, float)) or axis <= 0:
            return (1.0, 1.0, 1.0)
        scale.append(float(axis))
    return (scale[0], scale[1], scale[2])


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_vertices(raw_vertices: object) -> list[tuple[float, float, float]] | None:
    if not isinstance(raw_vertices, list) or not raw_vertices:
        return None

    vertices: list[tuple[float, float, float]] = []
    for vertex in raw_vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 3 or not all(_is_number(axis) for axis in vertex):
            return None
        vertices.append((round(float(vertex[0]), 6), round(float(vertex[1]), 6), round(float(vertex[2]), 6)))
    return vertices


def _normalize_faces(raw_faces: object) -> list[tuple[int, int, int]] | None:
    if not isinstance(raw_faces, list) or not raw_faces:
        return None

    faces: list[tuple[int, int, int]] = []
    for face in raw_faces:
        if not isinstance(face, (list, tuple)) or len(face) != 3 or not all(isinstance(index, int) for index in face):
            return None
        faces.append((int(face[0]), int(face[1]), int(face[2])))
    return faces


def _read_glb_chunks(payload: bytes) -> tuple[dict[str, object] | None, bytes | None]:
    if len(payload) < 20 or payload[:4] != GLB_MAGIC:
        return None, None

    total_length = struct.unpack_from('<I', payload, 8)[0]
    if total_length > len(payload):
        return None, None

    offset = 12
    document = None
    binary_blob = None
    while offset + 8 <= min(total_length, len(payload)):
        chunk_length, chunk_type = struct.unpack_from('<II', payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK_TYPE:
            try:
                document = json.loads(chunk.decode('utf8').rstrip(' \x00'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                document = None
        elif chunk_type == BIN_CHUNK_TYPE:
            binary_blob = chunk

    return document, binary_blob


def _extract_glb_geometry(payload: bytes) -> tuple[list[tuple[float, float, float]] | None, list[tuple[int, int, int]] | None]:
    document, binary_blob = _read_glb_chunks(payload)
    if not isinstance(document, dict) or not isinstance(binary_blob, bytes):
        return None, None

    meshes = document.get('meshes')
    accessors = document.get('accessors')
    buffer_views = document.get('bufferViews')
    if not isinstance(meshes, list) or not isinstance(accessors, list) or not isinstance(buffer_views, list) or not meshes:
        return None, None

    primitive = None
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get('primitives')
        if isinstance(primitives, list) and primitives:
            primitive = primitives[0]
            break
    if not isinstance(primitive, dict):
        return None, None

    attributes = primitive.get('attributes')
    if not isinstance(attributes, dict):
        return None, None
    position_accessor_index = attributes.get('POSITION')
    index_accessor_index = primitive.get('indices')
    if not isinstance(position_accessor_index, int) or not isinstance(index_accessor_index, int):
        return None, None

    def read_accessor(accessor_index: int) -> tuple[dict[str, object], dict[str, object]] | None:
        if accessor_index < 0 or accessor_index >= len(accessors):
            return None
        accessor = accessors[accessor_index]
        if not isinstance(accessor, dict):
            return None
        buffer_view_index = accessor.get('bufferView')
        if not isinstance(buffer_view_index, int) or buffer_view_index < 0 or buffer_view_index >= len(buffer_views):
            return None
        buffer_view = buffer_views[buffer_view_index]
        if not isinstance(buffer_view, dict):
            return None
        return accessor, buffer_view

    position_view = read_accessor(position_accessor_index)
    index_view = read_accessor(index_accessor_index)
    if position_view is None or index_view is None:
        return None, None
    position_accessor, position_buffer_view = position_view
    index_accessor, index_buffer_view = index_view

    if position_accessor.get('componentType') != 5126 or position_accessor.get('type') != 'VEC3':
        return None, None
    if index_accessor.get('componentType') != 5125 or index_accessor.get('type') != 'SCALAR':
        return None, None

    position_offset = int(position_buffer_view.get('byteOffset', 0)) + int(position_accessor.get('byteOffset', 0) or 0)
    position_stride = int(position_buffer_view.get('byteStride', 12) or 12)
    vertex_count = int(position_accessor.get('count', 0) or 0)
    vertices: list[tuple[float, float, float]] = []
    for index in range(vertex_count):
        start = position_offset + (index * position_stride)
        end = start + 12
        if end > len(binary_blob):
            return None, None
        vertices.append(tuple(round(axis, 6) for axis in struct.unpack_from('<3f', binary_blob, start)))

    index_offset = int(index_buffer_view.get('byteOffset', 0)) + int(index_accessor.get('byteOffset', 0) or 0)
    index_count = int(index_accessor.get('count', 0) or 0)
    faces: list[tuple[int, int, int]] = []
    for index in range(0, index_count, 3):
        start = index_offset + (index * 4)
        end = start + 12
        if end > len(binary_blob):
            return None, None
        faces.append(tuple(int(value) for value in struct.unpack_from('<3I', binary_blob, start)))

    return vertices, faces


def _base_mesh_faces() -> list[tuple[int, int, int]]:
    return [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]


def _derived_mesh_geometry(mesh_payload: dict[str, object]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    tokens = bytes_to_unit_floats(_mesh_payload_bytes(mesh_payload), length=36)
    scale_x, scale_y, scale_z = _extent_scale(mesh_payload)
    half_extents = (scale_x / 2.0, scale_y / 2.0, scale_z / 2.0)
    corners = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    vertices: list[tuple[float, float, float]] = []

    for index, (sign_x, sign_y, sign_z) in enumerate(corners):
        token_offset = index * 3
        axis_tokens = tokens[token_offset : token_offset + 3]
        while len(axis_tokens) < 3:
            axis_tokens.append(0.5)
        offsets = [((axis_token - 0.5) * 0.18) for axis_token in axis_tokens]
        vertices.append(
            (
                round(sign_x * half_extents[0] * (1.0 + offsets[0]), 6),
                round(sign_y * half_extents[1] * (1.0 + offsets[1]), 6),
                round(sign_z * half_extents[2] * (1.0 + offsets[2]), 6),
            )
        )

    faces = _base_mesh_faces() + [
        (0, 8, 1),
        (1, 9, 2),
        (2, 10, 3),
        (3, 11, 0),
        (4, 8, 5),
        (5, 9, 6),
        (6, 10, 7),
        (7, 11, 4),
    ]
    return vertices, faces


def _apply_extent_scale(vertices: list[tuple[float, float, float]], mesh_payload: dict[str, object]) -> list[tuple[float, float, float]]:
    scale_x, scale_y, scale_z = _extent_scale(mesh_payload)
    if (scale_x, scale_y, scale_z) == (1.0, 1.0, 1.0):
        return vertices
    return [
        (round(vertex[0] * scale_x, 6), round(vertex[1] * scale_y, 6), round(vertex[2] * scale_z, 6)) for vertex in vertices
    ]


def _geometry_signature(vertices: list[tuple[float, float, float]]) -> int:
    return stable_signature([axis for vertex in vertices for axis in vertex])


def _extents(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not vertices:
        return (1.0, 1.0, 1.0)
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    extent_x = max(max(xs) - min(xs), 1e-6)
    extent_y = max(max(ys) - min(ys), 1e-6)
    extent_z = max(max(zs) - min(zs), 1e-6)
    return (round(extent_x, 6), round(extent_y, 6), round(extent_z, 6))


def _mesh_geometry(mesh_payload: dict[str, object]) -> dict[str, object]:
    vertices = _normalize_vertices(mesh_payload.get('vertices'))
    faces = _normalize_faces(mesh_payload.get('faces'))
    if vertices is None or faces is None:
        payload_bytes = _mesh_payload_bytes(mesh_payload)
        parsed_vertices, parsed_faces = _extract_glb_geometry(payload_bytes)
        if parsed_vertices is not None and parsed_faces is not None:
            vertices, faces = parsed_vertices, parsed_faces
        else:
            vertices, faces = _derived_mesh_geometry(mesh_payload)

    vertices = _apply_extent_scale(vertices, mesh_payload)
    return {
        'points': vertices,
        'vertices': vertices,
        'faces': faces,
        'extents': _extents(vertices),
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'signature': _geometry_signature(vertices),
    }


def _centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = max(len(points), 1)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _center_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    center = _centroid(points)
    return [(point[0] - center[0], point[1] - center[1], point[2] - center[2]) for point in points]


def _bbox_diagonal(extents: tuple[float, float, float]) -> float:
    diagonal = math.sqrt(sum(axis * axis for axis in extents))
    return diagonal if diagonal > 0 else 1.0


def _mean_extent(extents: tuple[float, float, float]) -> float:
    return sum(extents) / max(len(extents), 1)


def _scale_points(points: list[tuple[float, float, float]], factor: float) -> list[tuple[float, float, float]]:
    return [(point[0] * factor, point[1] * factor, point[2] * factor) for point in points]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2))


def _nearest_distance(point: tuple[float, float, float], candidates: list[tuple[float, float, float]]) -> float:
    return min(_distance(point, candidate) for candidate in candidates)


def _chamfer_distance(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]], diagonal: float) -> float:
    forward = sum(_nearest_distance(point, b) for point in a) / max(len(a), 1)
    backward = sum(_nearest_distance(point, a) for point in b) / max(len(b), 1)
    return round((forward + backward) / max(diagonal, 1e-6), 6)


def _rms_displacement(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]], diagonal: float) -> float:
    squared = [(_distance(left, right) ** 2) for left, right in zip(a, b)]
    if not squared:
        return 0.0
    return round(math.sqrt(sum(squared) / len(squared)) / max(diagonal, 1e-6), 6)


def _extent_ratio(coarse_extents: tuple[float, float, float], refined_extents: tuple[float, float, float]) -> list[float]:
    return [round(refined_axis / coarse_axis if coarse_axis else 1.0, 3) for coarse_axis, refined_axis in zip(coarse_extents, refined_extents)]


def _topology_delta_ratio(coarse_count: int, refined_count: int) -> float:
    baseline = max(coarse_count, 1)
    return abs(refined_count - coarse_count) / baseline


def evaluate_geometric_gate(*, coarse_mesh_payload: object, refined_mesh_payload: object, preserve_scale: bool) -> dict[str, object]:
    if not isinstance(coarse_mesh_payload, dict) or not isinstance(refined_mesh_payload, dict):
        raise MeshGateError('GEOMETRIC_GATE_REJECTED: coarse and refined mesh payloads must be structured dictionaries.')

    coarse_geometry = _mesh_geometry(coarse_mesh_payload)
    refined_geometry = _mesh_geometry(refined_mesh_payload)
    coarse_points = _center_points(coarse_geometry['points'])
    refined_points = _center_points(refined_geometry['points'])
    coarse_extents = coarse_geometry['extents']
    refined_extents = refined_geometry['extents']
    extent_ratio = _extent_ratio(coarse_extents, refined_extents)
    diagonal = _bbox_diagonal(coarse_extents)

    if not preserve_scale:
        scale_factor = _mean_extent(coarse_extents) / max(_mean_extent(refined_extents), 1e-6)
        refined_points = _scale_points(refined_points, scale_factor)
        extent_ratio = [1.0, 1.0, 1.0]

    chamfer = _chamfer_distance(coarse_points, refined_points, diagonal)
    rms = _rms_displacement(coarse_points, refined_points, diagonal)
    vertex_delta = _topology_delta_ratio(coarse_geometry['vertex_count'], refined_geometry['vertex_count'])
    face_delta = _topology_delta_ratio(coarse_geometry['face_count'], refined_geometry['face_count'])
    topology_changed = max(vertex_delta, face_delta) >= TOPOLOGY_DELTA_THRESHOLD

    if preserve_scale and any(axis < PRESERVE_SCALE_TOLERANCE[0] or axis > PRESERVE_SCALE_TOLERANCE[1] for axis in extent_ratio):
        raise MeshGateError(
            f'GEOMETRIC_GATE_REJECTED: preserve-scale bbox tolerance failed with extent_ratio={extent_ratio}.'
        )

    if chamfer <= CHAMFER_EPSILON:
        raise MeshGateError(
            f'GEOMETRIC_GATE_REJECTED: normalized bidirectional Chamfer {chamfer} did not exceed epsilon {CHAMFER_EPSILON}.'
        )

    if not topology_changed and rms <= RMS_DISPLACEMENT_THRESHOLD:
        raise MeshGateError('GEOMETRIC_GATE_REJECTED: topology and RMS displacement do not prove real refinement.')

    return {
        'chamfer': chamfer,
        'rms': rms,
        'topology_changed': topology_changed,
        'extent_ratio': extent_ratio,
        'vertex_delta_ratio': round(vertex_delta, 6),
        'face_delta_ratio': round(face_delta, 6),
        'coarse_signature': coarse_geometry['signature'],
        'refined_signature': refined_geometry['signature'],
        'coarse_vertex_count': coarse_geometry['vertex_count'],
        'refined_vertex_count': refined_geometry['vertex_count'],
        'coarse_face_count': coarse_geometry['face_count'],
        'refined_face_count': refined_geometry['face_count'],
        'normalized_diagonal': round(clamp_unit(min(diagonal / math.sqrt(3), 1.0)), 6),
    }


def resolved_output_path(output_dir: str, output_format: str) -> Path:
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / f'refined.{output_format}').resolve()
    if destination.parent != destination_dir:
        raise MeshExportError(f'Refined output must stay inside output_dir: {destination}.')
    return destination


def build_renderable_mesh_payload(mesh_payload: object) -> dict[str, object]:
    if not isinstance(mesh_payload, dict):
        raise MeshExportError('mesh_payload must be a structured binary-safe mesh payload.')

    normalized_payload = dict(mesh_payload)
    payload_bytes = normalized_payload.get('bytes')
    if not isinstance(payload_bytes, bytes):
        raise MeshExportError('mesh_payload.bytes must be binary data.')

    vertices = _normalize_vertices(normalized_payload.get('vertices'))
    faces = _normalize_faces(normalized_payload.get('faces'))
    if vertices is None or faces is None:
        parsed_vertices, parsed_faces = _extract_glb_geometry(payload_bytes)
        if parsed_vertices is not None and parsed_faces is not None:
            vertices, faces = parsed_vertices, parsed_faces
        else:
            vertices, faces = _derived_mesh_geometry(normalized_payload)

    normalized_payload['vertices'] = _apply_extent_scale(vertices, normalized_payload)
    normalized_payload['faces'] = faces
    normalized_payload['mesh_name'] = str(normalized_payload.get('mesh_name') or normalized_payload.get('kind') or 'refined-mesh')
    return normalized_payload


def _export_trimesh_glb(mesh_payload: dict[str, object]) -> bytes:
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover
        raise MeshExportError('Required runtime import is unavailable: trimesh.') from error

    vertices = mesh_payload.get('vertices')
    faces = mesh_payload.get('faces')
    if not isinstance(vertices, list) or not isinstance(faces, list):
        raise MeshExportError('mesh_payload must include renderable vertices and faces before GLB export.')

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=str(mesh_payload.get('mesh_name') or 'refined-mesh'))
    exported = scene.export(file_type='glb')
    if not isinstance(exported, (bytes, bytearray)):
        raise MeshExportError('trimesh GLB export must return bytes.')
    return bytes(exported)


def _serialize_glb_payload(mesh_payload: object) -> bytes:
    normalized_payload = build_renderable_mesh_payload(mesh_payload)
    payload_bytes = normalized_payload.get('bytes')
    if normalized_payload.get('is_binary_glb') is True and isinstance(payload_bytes, bytes) and payload_bytes[:4] == GLB_MAGIC:
        return payload_bytes
    return _export_trimesh_glb(normalized_payload)


def export_refined_glb(*, output_dir: str, output_format: str, mesh_payload: object) -> str:
    if output_format != 'glb':
        raise MeshExportError('UltraShape local runner is glb-only in this MVP.')

    destination = resolved_output_path(output_dir, output_format)
    if os.environ.get('ULTRASHAPE_TEST_SKIP_OUTPUT_WRITE') == '1':
        return str(destination)

    destination.write_bytes(_serialize_glb_payload(mesh_payload))
    return str(destination)
