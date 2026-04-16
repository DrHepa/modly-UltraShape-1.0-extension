"""Mesh metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path


class MeshExportError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


GLB_MAGIC = b'glTF'
GLB_VERSION = 2
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942


def default_surface_algorithm() -> str:
    return 'mc'


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
