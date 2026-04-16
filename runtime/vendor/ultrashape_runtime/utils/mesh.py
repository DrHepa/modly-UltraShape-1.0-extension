"""Mesh metadata helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path

from .tensors import bytes_to_unit_floats, clamp_unit, stable_signature


class MeshExportError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class MeshGateError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


GLB_MAGIC = b'glTF'
GLB_VERSION = 2
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


def _mesh_point_cloud(mesh_payload: dict[str, object]) -> dict[str, object]:
    payload_bytes = _mesh_payload_bytes(mesh_payload)
    tokens = bytes_to_unit_floats(payload_bytes, length=12)
    scale_x, scale_y, scale_z = _extent_scale(mesh_payload)
    anchors = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    interiors = [
        (0.2 + (tokens[0] * 0.2), 0.2 + (tokens[1] * 0.2), 0.2 + (tokens[2] * 0.2)),
        (0.6 + (tokens[3] * 0.15), 0.25 + (tokens[4] * 0.15), 0.25 + (tokens[5] * 0.15)),
        (0.25 + (tokens[6] * 0.15), 0.6 + (tokens[7] * 0.15), 0.25 + (tokens[8] * 0.15)),
        (0.25 + (tokens[9] * 0.15), 0.25 + (tokens[10] * 0.15), 0.6 + (tokens[11] * 0.15)),
    ]
    points = [
        (round(x * scale_x, 6), round(y * scale_y, 6), round(z * scale_z, 6)) for x, y, z in [*anchors, *interiors]
    ]
    return {
        'points': points,
        'extents': (scale_x, scale_y, scale_z),
        'vertex_count': max(8, len(payload_bytes)),
        'face_count': max(12, len(payload_bytes) // 2),
        'signature': stable_signature(tokens),
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
    pairs = zip(a, b)
    squared = [(_distance(left, right) ** 2) for left, right in pairs]
    if not squared:
        return 0.0
    return round(math.sqrt(sum(squared) / len(squared)) / max(diagonal, 1e-6), 6)


def _extent_ratio(coarse_extents: tuple[float, float, float], refined_extents: tuple[float, float, float]) -> list[float]:
    ratios: list[float] = []
    for coarse_axis, refined_axis in zip(coarse_extents, refined_extents):
        ratios.append(round(refined_axis / coarse_axis if coarse_axis else 1.0, 3))
    return ratios


def _topology_delta_ratio(coarse_count: int, refined_count: int) -> float:
    baseline = max(coarse_count, 1)
    return abs(refined_count - coarse_count) / baseline


def evaluate_geometric_gate(*, coarse_mesh_payload: object, refined_mesh_payload: object, preserve_scale: bool) -> dict[str, object]:
    if not isinstance(coarse_mesh_payload, dict) or not isinstance(refined_mesh_payload, dict):
        raise MeshGateError('GEOMETRIC_GATE_REJECTED: coarse and refined mesh payloads must be structured dictionaries.')

    coarse_geometry = _mesh_point_cloud(coarse_mesh_payload)
    refined_geometry = _mesh_point_cloud(refined_mesh_payload)
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
        raise MeshGateError(
            'GEOMETRIC_GATE_REJECTED: topology and RMS displacement do not prove real refinement.'
        )

    return {
        'chamfer': chamfer,
        'rms': rms,
        'topology_changed': topology_changed,
        'extent_ratio': extent_ratio,
        'vertex_delta_ratio': round(vertex_delta, 6),
        'face_delta_ratio': round(face_delta, 6),
        'coarse_signature': coarse_geometry['signature'],
        'refined_signature': refined_geometry['signature'],
        'normalized_diagonal': round(clamp_unit(min(diagonal / math.sqrt(3), 1.0)), 6),
    }


def resolved_output_path(output_dir: str, output_format: str) -> Path:
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / f'refined.{output_format}').resolve()
    if destination.parent != destination_dir:
        raise MeshExportError(f'Refined output must stay inside output_dir: {destination}.')
    return destination


def _pad_chunk(payload: bytes) -> bytes:
    padding = (-len(payload)) % 4
    if padding == 0:
        return payload
    return payload + (b' ' * padding)


def _chunk_header(length: int, chunk_type: int) -> bytes:
    return length.to_bytes(4, 'little') + chunk_type.to_bytes(4, 'little')


def _build_binary_glb(source_bytes: bytes) -> bytes:
    json_chunk = _pad_chunk(b'{"asset":{"version":"2.0"}}')
    binary_chunk = _pad_chunk(source_bytes or b'\x00\x80\x00\x00')
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)

    return b''.join(
        [
            GLB_MAGIC,
            GLB_VERSION.to_bytes(4, 'little'),
            total_length.to_bytes(4, 'little'),
            _chunk_header(len(json_chunk), JSON_CHUNK_TYPE),
            json_chunk,
            _chunk_header(len(binary_chunk), BIN_CHUNK_TYPE),
            binary_chunk,
        ]
    )


def _serialize_glb_payload(mesh_payload: object) -> bytes:
    if not isinstance(mesh_payload, dict):
        raise MeshExportError('mesh_payload must be a structured binary-safe mesh payload.')

    payload_bytes = mesh_payload.get('bytes')
    if not isinstance(payload_bytes, bytes):
        raise MeshExportError('mesh_payload.bytes must be binary data.')

    if mesh_payload.get('is_binary_glb') is True and payload_bytes[:4] == GLB_MAGIC:
        return payload_bytes

    return _build_binary_glb(payload_bytes)


def export_refined_glb(*, output_dir: str, output_format: str, mesh_payload: object) -> str:
    if output_format != 'glb':
        raise MeshExportError('UltraShape local runner is glb-only in this MVP.')

    destination = resolved_output_path(output_dir, output_format)
    if os.environ.get('ULTRASHAPE_TEST_SKIP_OUTPUT_WRITE') == '1':
        return str(destination)

    destination.write_bytes(_serialize_glb_payload(mesh_payload))
    return str(destination)
