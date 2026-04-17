"""Lightweight voxel conditioning helpers for the vendored runtime."""

from __future__ import annotations

import json
import struct

from .tensors import bytes_to_unit_floats, stable_signature


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


def _extract_glb_vertices(payload: bytes) -> list[tuple[float, float, float]]:
    document, binary_blob = _read_glb_chunks(payload)
    if not isinstance(document, dict) or not isinstance(binary_blob, bytes):
        return []

    meshes = document.get('meshes')
    accessors = document.get('accessors')
    buffer_views = document.get('bufferViews')
    if not isinstance(meshes, list) or not isinstance(accessors, list) or not isinstance(buffer_views, list) or not meshes:
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
    if not isinstance(accessor_index, int) or accessor_index < 0 or accessor_index >= len(accessors):
        return []

    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict):
        return []
    buffer_view_index = accessor.get('bufferView')
    count = accessor.get('count')
    component_type = accessor.get('componentType')
    value_type = accessor.get('type')
    if (
        not isinstance(buffer_view_index, int)
        or not isinstance(count, int)
        or component_type != 5126
        or value_type != 'VEC3'
        or buffer_view_index < 0
        or buffer_view_index >= len(buffer_views)
    ):
        return []

    buffer_view = buffer_views[buffer_view_index]
    if not isinstance(buffer_view, dict):
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


def sample_surface_points(mesh_payload: dict[str, object]) -> list[tuple[float, float, float]]:
    payload = mesh_payload.get('bytes')
    if not isinstance(payload, bytes):
        return []

    vertices = _extract_glb_vertices(payload)
    if vertices:
        return vertices

    tokens = bytes_to_unit_floats(payload, length=18)
    points: list[tuple[float, float, float]] = []
    for index in range(0, len(tokens), 3):
        triplet = tokens[index : index + 3]
        if len(triplet) < 3:
            break
        points.append(tuple(round((value * 2.0) - 1.0, 6) for value in triplet))
    return points


def voxelize_from_point(mesh_payload: dict[str, object], *, resolution: int = 12) -> dict[str, object]:
    points = sample_surface_points(mesh_payload)
    occupied: dict[tuple[int, int, int], int] = {}

    for point in points:
        voxel = []
        for axis in point:
            normalized = max(0.0, min(0.999999, (float(axis) + 1.0) / 2.0))
            voxel.append(int(normalized * resolution))
        key = (voxel[0], voxel[1], voxel[2])
        occupied[key] = occupied.get(key, 0) + 1

    voxel_values = [count for _, count in sorted(occupied.items())]
    voxel_signature = stable_signature([round(value / max(resolution, 1), 6) for value in voxel_values])
    return {
        'voxelizer': 'voxelize_from_point',
        'resolution': resolution,
        'surface_point_count': len(points),
        'voxel_values': voxel_values,
        'voxel_count': len(occupied),
        'voxel_signature': voxel_signature,
    }
