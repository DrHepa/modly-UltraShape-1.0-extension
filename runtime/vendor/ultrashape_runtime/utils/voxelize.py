"""Geometry-backed voxel conditioning helpers for the vendored runtime."""

from __future__ import annotations

import json
import struct

from . import clamp_unit, stable_signature


GLB_MAGIC = b'glTF'
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942


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


def _read_accessor(document: dict[str, object], accessor_index: int) -> tuple[dict[str, object], dict[str, object]] | None:
    accessors = document.get('accessors')
    buffer_views = document.get('bufferViews')
    if not isinstance(accessors, list) or not isinstance(buffer_views, list):
        return None
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


def _extract_glb_vertices(payload: bytes) -> list[tuple[float, float, float]]:
    document, binary_blob = _read_glb_chunks(payload)
    if not isinstance(document, dict) or not isinstance(binary_blob, bytes):
        return []

    meshes = document.get('meshes')
    if not isinstance(meshes, list) or not meshes:
        return []

    primitive = None
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get('primitives')
        if isinstance(primitives, list) and primitives:
            primitive = primitives[0]
            break
    if not isinstance(primitive, dict):
        return []

    attributes = primitive.get('attributes')
    if not isinstance(attributes, dict):
        return []
    accessor_index = attributes.get('POSITION')
    if not isinstance(accessor_index, int) or accessor_index < 0:
        return []

    accessor_view = _read_accessor(document, accessor_index)
    if accessor_view is None:
        return []
    accessor, buffer_view = accessor_view
    count = accessor.get('count')
    component_type = accessor.get('componentType')
    value_type = accessor.get('type')
    if not isinstance(count, int) or component_type != 5126 or value_type != 'VEC3':
        return []

    byte_offset = int(buffer_view.get('byteOffset', 0)) + int(accessor.get('byteOffset', 0) or 0)
    byte_length = int(buffer_view.get('byteLength', 0))
    stride = int(buffer_view.get('byteStride', 12) or 12)
    if stride < 12 or byte_length <= 0 or byte_offset < 0:
        return []

    vertices: list[tuple[float, float, float]] = []
    for index in range(count):
        start = byte_offset + (index * stride)
        end = start + 12
        if end > len(binary_blob):
            break
        vertices.append(tuple(round(axis, 6) for axis in struct.unpack_from('<3f', binary_blob, start)))
    return vertices


def _extract_glb_faces(payload: bytes) -> list[tuple[int, int, int]]:
    document, binary_blob = _read_glb_chunks(payload)
    if not isinstance(document, dict) or not isinstance(binary_blob, bytes):
        return []

    meshes = document.get('meshes')
    if not isinstance(meshes, list) or not meshes:
        return []

    primitive = None
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get('primitives')
        if isinstance(primitives, list) and primitives:
            primitive = primitives[0]
            break
    if not isinstance(primitive, dict):
        return []

    accessor_index = primitive.get('indices')
    if not isinstance(accessor_index, int):
        return []

    accessor_view = _read_accessor(document, accessor_index)
    if accessor_view is None:
        return []

    accessor, buffer_view = accessor_view
    if accessor.get('componentType') != 5125 or accessor.get('type') != 'SCALAR':
        return []

    count = int(accessor.get('count', 0) or 0)
    byte_offset = int(buffer_view.get('byteOffset', 0)) + int(accessor.get('byteOffset', 0) or 0)
    faces: list[tuple[int, int, int]] = []
    for index in range(0, count, 3):
        start = byte_offset + (index * 4)
        end = start + 12
        if end > len(binary_blob):
            break
        faces.append(tuple(int(value) for value in struct.unpack_from('<3I', binary_blob, start)))
    return faces


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


def mesh_geometry_from_glb(payload: bytes) -> dict[str, object]:
    vertices = _extract_glb_vertices(payload)
    faces = _extract_glb_faces(payload)
    if not vertices or not faces:
        return {}

    bounds = _bounds(vertices)
    point_values = [axis for vertex in vertices for axis in vertex]
    geometry_tokens = [clamp_unit((axis + 1.5) / 3.0) for axis in point_values[:24]]
    return {
        'vertices': vertices,
        'faces': faces,
        'bounds': bounds,
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'tokens': geometry_tokens,
        'signature': stable_signature(geometry_tokens),
    }


def _triangle_centroid(
    vertices: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> tuple[float, float, float] | None:
    try:
        points = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    except IndexError:
        return None
    return (
        round(sum(point[0] for point in points) / 3.0, 6),
        round(sum(point[1] for point in points) / 3.0, 6),
        round(sum(point[2] for point in points) / 3.0, 6),
    )


def sample_surface_points(mesh_payload: dict[str, object]) -> list[tuple[float, float, float]]:
    vertices = mesh_payload.get('vertices') if isinstance(mesh_payload.get('vertices'), list) else []
    faces = mesh_payload.get('faces') if isinstance(mesh_payload.get('faces'), list) else []
    normalized_vertices = [tuple(float(axis) for axis in vertex[:3]) for vertex in vertices if isinstance(vertex, (list, tuple)) and len(vertex) >= 3]
    normalized_faces = [tuple(int(index) for index in face[:3]) for face in faces if isinstance(face, (list, tuple)) and len(face) >= 3]
    if not normalized_vertices:
        payload = mesh_payload.get('bytes')
        if not isinstance(payload, bytes):
            return []
        geometry = mesh_geometry_from_glb(payload)
        normalized_vertices = geometry.get('vertices') if isinstance(geometry.get('vertices'), list) else []
        normalized_faces = geometry.get('faces') if isinstance(geometry.get('faces'), list) else []

    points = list(normalized_vertices)
    for face in normalized_faces:
        centroid = _triangle_centroid(normalized_vertices, face)
        if centroid is not None:
            points.append(centroid)
    return points


def voxelize_from_point(mesh_payload: dict[str, object], *, resolution: int = 12) -> dict[str, object]:
    points = sample_surface_points(mesh_payload)
    vertices = mesh_payload.get('vertices') if isinstance(mesh_payload.get('vertices'), list) else []
    normalized_vertices = [tuple(float(axis) for axis in vertex[:3]) for vertex in vertices if isinstance(vertex, (list, tuple)) and len(vertex) >= 3]
    bounds = _bounds(normalized_vertices or points)
    minimum = bounds['min']
    extents = tuple(max(axis, 1e-6) for axis in bounds['extents'])
    occupied: dict[tuple[int, int, int], int] = {}

    for point in points:
        voxel = []
        for axis_index, axis in enumerate(point):
            normalized = max(0.0, min(0.999999, (float(axis) - minimum[axis_index]) / extents[axis_index]))
            voxel.append(int(normalized * resolution))
        key = (voxel[0], voxel[1], voxel[2])
        occupied[key] = occupied.get(key, 0) + 1

    sorted_voxels = sorted(occupied.items())
    voxel_values = [count for _, count in sorted_voxels]
    voxel_coords = [coords for coords, _ in sorted_voxels]
    voxel_tokens: list[float] = []
    for coords, count in sorted_voxels:
        voxel_tokens.extend([coord / max(resolution, 1) for coord in coords])
        voxel_tokens.append(count / max(len(points), 1))
    voxel_signature = stable_signature([round(value, 6) for value in voxel_tokens[:24]])
    return {
        'voxelizer': 'voxelize_from_point',
        'resolution': resolution,
        'surface_point_count': len(points),
        'voxel_coords': voxel_coords,
        'voxel_values': voxel_values,
        'voxel_count': len(occupied),
        'bounds': bounds,
        'tokens': [round(value, 6) for value in voxel_tokens[:8]],
        'voxel_signature': voxel_signature,
        'occupied_ratio': clamp_unit(len(occupied) / max(len(points), 1)),
    }
