"""Executable mc-only pipeline for the vendored UltraShape runtime."""

from __future__ import annotations

import importlib
from pathlib import Path

from .models.autoencoders.surface_extractors import extract_surface
from .preprocessors import normalize_reference_asset
from .surface_loaders import load_coarse_surface
from .utils.checkpoint import resolve_checkpoint
from .utils.mesh import export_refined_glb


class PipelineDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


class PipelineUnavailableError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def build_refine_pipeline() -> dict[str, str]:
    return {'name': 'ultrashape-refine', 'scope': 'mc-only'}


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered in {'null', 'none'}:
        return None
    return value


def _next_meaningful_line(lines: list[str], start: int) -> tuple[int, str] | None:
    for index in range(start, len(lines)):
        candidate = lines[index]
        if candidate.strip() and not candidate.lstrip().startswith('#'):
            return index, candidate
    return None


def _parse_block(lines: list[str], start: int, indent: int):
    probe = _next_meaningful_line(lines, start)
    if probe is None:
        return {}, len(lines)

    _, first_line = probe
    if len(first_line) - len(first_line.lstrip(' ')) < indent:
        return {}, start

    if first_line.strip().startswith('- '):
        items: list[object] = []
        index = start
        while True:
            probe = _next_meaningful_line(lines, index)
            if probe is None:
                return items, len(lines)
            index, line = probe
            line_indent = len(line) - len(line.lstrip(' '))
            if line_indent < indent:
                return items, index
            if line_indent != indent or not line.strip().startswith('- '):
                raise PipelineUnavailableError(f'Invalid config structure near: {line.strip()}')
            items.append(_parse_scalar(line.strip()[2:].strip()))
            index += 1

    mapping: dict[str, object] = {}
    index = start
    while True:
        probe = _next_meaningful_line(lines, index)
        if probe is None:
            return mapping, len(lines)
        index, line = probe
        line_indent = len(line) - len(line.lstrip(' '))
        if line_indent < indent:
            return mapping, index
        if line_indent != indent:
            raise PipelineUnavailableError(f'Invalid config indentation near: {line.strip()}')

        stripped = line.strip()
        if ':' not in stripped:
            raise PipelineUnavailableError(f'Invalid config line: {stripped}')
        key, raw_value = stripped.split(':', 1)
        value = raw_value.strip()
        index += 1
        if value:
            mapping[key.strip()] = _parse_scalar(value)
            continue

        child, index = _parse_block(lines, index, indent + 2)
        mapping[key.strip()] = child


def load_runtime_config(config_path: str) -> dict[str, object]:
    path = Path(config_path)
    if not path.is_file():
        raise PipelineUnavailableError(f'config_path is not readable: {config_path}.')

    config, _ = _parse_block(path.read_text(encoding='utf8').splitlines(), 0, 0)
    if not isinstance(config, dict):
        raise PipelineUnavailableError('Runtime config root must be a mapping.')
    return config


def require_imports(config: dict[str, object]) -> None:
    dependencies = config.get('dependencies')
    required_imports = []
    if isinstance(dependencies, dict):
        raw_required = dependencies.get('required_imports')
        if isinstance(raw_required, list):
            required_imports = [entry for entry in raw_required if isinstance(entry, str)]

    for module_name in required_imports:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise PipelineDependencyError(f'Required runtime import is unavailable: {module_name}.') from error


def validate_mvp_scope(config: dict[str, object], backend: str, output_format: str) -> tuple[str, str]:
    model = config.get('model')
    runtime = config.get('runtime')
    surface = config.get('surface')

    scope = model.get('scope') if isinstance(model, dict) else config.get('scope')
    configured_backend = runtime.get('backend') if isinstance(runtime, dict) else None
    extraction = surface.get('extraction') if isinstance(surface, dict) else None

    if scope != 'mc-only':
        raise PipelineUnavailableError(f'UltraShape local runner is mc-only in this MVP, received scope={scope}.')
    if configured_backend not in (None, 'local'):
        raise PipelineUnavailableError(
            f'UltraShape local runner is local-only in this MVP, received config backend={configured_backend}.'
        )
    if backend != 'local':
        raise PipelineUnavailableError('UltraShape local runner is local-only in this MVP.')
    if output_format != 'glb':
        raise PipelineUnavailableError('UltraShape local runner is glb-only in this MVP.')

    return 'mc', str(extraction or 'mc')


def run_refine_pipeline(
    *,
    reference_image: str,
    coarse_mesh: str,
    output_dir: str,
    output_format: str,
    checkpoint: str | None,
    config_path: str,
    ext_dir: str,
    backend: str,
    preserve_scale: bool,
) -> dict[str, object]:
    config = load_runtime_config(config_path)
    require_imports(config)
    _, extraction = validate_mvp_scope(config, backend, output_format)

    weights = config.get('weights') if isinstance(config.get('weights'), dict) else {}
    reference_asset = normalize_reference_asset(reference_image)
    coarse_surface = load_coarse_surface(coarse_mesh)
    resolved_checkpoint = resolve_checkpoint(checkpoint, weights.get('primary') if isinstance(weights, dict) else None, ext_dir)
    refined_surface = extract_surface(
        extraction=extraction,
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        preserve_scale=preserve_scale,
    )
    output_path = export_refined_glb(
        output_dir=output_dir,
        output_format=output_format,
        mesh_payload=refined_surface['payload'],
    )

    if not Path(output_path).is_file():
        raise PipelineUnavailableError(f'Expected refined.glb output was not generated: {output_path}.')

    return {
        'file_path': output_path,
        'format': 'glb',
        'backend': 'local',
        'warnings': [],
        'checkpoint': resolved_checkpoint,
    }
