"""Surface loader seam for mc-only local refinement."""

from __future__ import annotations

from pathlib import Path


class SurfaceLoadError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def load_coarse_surface(path: str) -> dict[str, str]:
    surface_path = Path(path)
    if not surface_path.is_file():
        raise SurfaceLoadError(f'coarse_mesh is not readable: {path}.')

    return {
        'path': str(surface_path),
        'content': surface_path.read_text(encoding='utf8'),
        'suffix': surface_path.suffix.lower(),
    }
