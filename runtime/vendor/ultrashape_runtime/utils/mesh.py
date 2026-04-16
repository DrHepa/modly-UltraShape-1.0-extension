"""Mesh metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path


class MeshExportError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def default_surface_algorithm() -> str:
    return 'mc'


def resolved_output_path(output_dir: str, output_format: str) -> Path:
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / f'refined.{output_format}').resolve()
    if destination.parent != destination_dir:
        raise MeshExportError(f'Refined output must stay inside output_dir: {destination}.')
    return destination


def export_refined_glb(*, output_dir: str, output_format: str, mesh_payload: str) -> str:
    if output_format != 'glb':
        raise MeshExportError('UltraShape local runner is glb-only in this MVP.')

    destination = resolved_output_path(output_dir, output_format)
    if os.environ.get('ULTRASHAPE_TEST_SKIP_OUTPUT_WRITE') == '1':
        return str(destination)

    destination.write_text(mesh_payload, encoding='utf8')
    return str(destination)
