"""Executable mc-only pipeline for the vendored UltraShape runtime."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from .models.autoencoders.model import ShapeVAE
from .models.autoencoders.surface_extractors import extract_surface
from .models.autoencoders.volume_decoders import decode_volume
from .models.conditioner_mask import SingleImageEncoder
from .models.denoisers.dit_mask import RefineDiT
from .preprocessors import ImageProcessorV2, normalize_reference_asset
from .schedulers import build_flow_matching_schedule
from .surface_loaders import SharpEdgeSurfaceLoader, load_coarse_surface
from .utils.checkpoint import apply_checkpoint_state, load_checkpoint_subtrees
from .utils.mesh import evaluate_geometric_gate, export_refined_glb


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
        required_group = dependencies.get('required')
        if isinstance(required_group, dict):
            nested_imports = required_group.get('imports')
            if isinstance(nested_imports, list):
                required_imports = [entry for entry in nested_imports if isinstance(entry, str)]

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
    steps: int,
    guidance_scale: float,
    seed: int | None,
    preserve_scale: bool,
) -> dict[str, object]:
    config = load_runtime_config(config_path)
    require_imports(config)
    _, extraction = validate_mvp_scope(config, backend, output_format)

    checkpoint_config = config.get('checkpoint') if isinstance(config.get('checkpoint'), dict) else {}
    weights = config.get('weights') if isinstance(config.get('weights'), dict) else {}
    image_processor = ImageProcessorV2()
    surface_loader = SharpEdgeSurfaceLoader()
    reference_asset = normalize_reference_asset(reference_image)
    coarse_surface = load_coarse_surface(coarse_mesh)
    checkpoint_bundle = load_checkpoint_subtrees(
        checkpoint,
        weights.get('primary') if isinstance(weights, dict) else None,
        ext_dir,
        checkpoint_config.get('required_subtrees') if isinstance(checkpoint_config.get('required_subtrees'), list) else None,
    )
    checkpoint_state = checkpoint_bundle['bundle'] if isinstance(checkpoint_bundle.get('bundle'), dict) else {}
    conditioner = SingleImageEncoder()
    conditioner_hydration = apply_checkpoint_state(conditioner, checkpoint_state.get('conditioner'))
    conditioning = conditioner.build(
        reference_asset=reference_asset,
        coarse_surface=coarse_surface,
    )
    schedule = build_flow_matching_schedule(steps=steps, guidance_scale=guidance_scale)
    denoiser = RefineDiT()
    denoiser_hydration = apply_checkpoint_state(denoiser, checkpoint_state.get('dit'))
    denoised = denoiser.denoise(
        conditioning=conditioning,
        schedule=schedule,
        seed=seed,
    )
    vae = ShapeVAE()
    vae_hydration = apply_checkpoint_state(vae, checkpoint_state.get('vae'))
    decoded_latents = vae.decode_latents(denoised, reference_asset)
    decoded_volume = decode_volume(decoded_latents)
    refined_surface = extract_surface(
        extraction=extraction,
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        decoded_volume=decoded_volume,
        preserve_scale=preserve_scale,
    )
    refined_payload = dict(refined_surface['payload'])

    if os.environ.get('ULTRASHAPE_TEST_FORCE_PASSTHROUGH') == '1':
        refined_payload = {
            **coarse_surface['mesh'],
            'kind': 'refined-mesh',
        }
    elif os.environ.get('ULTRASHAPE_TEST_FORCE_SCALE_DRIFT') == '1':
        refined_payload['test_extent_scale'] = [1.2, 1.2, 1.2]

    gate_metrics = evaluate_geometric_gate(
        coarse_mesh_payload=coarse_surface['mesh'],
        refined_mesh_payload=refined_payload,
        preserve_scale=preserve_scale,
    )
    output_path = export_refined_glb(
        output_dir=output_dir,
        output_format=output_format,
        mesh_payload=refined_payload,
    )

    if not Path(output_path).is_file():
        raise PipelineUnavailableError(f'Expected refined.glb output was not generated: {output_path}.')

    return {
        'file_path': output_path,
        'format': 'glb',
        'backend': 'local',
        'warnings': [],
        'metrics': {
            'chamfer': gate_metrics['chamfer'],
            'rms': gate_metrics['rms'],
            'topology_changed': gate_metrics['topology_changed'],
            'extent_ratio': gate_metrics['extent_ratio'],
            'execution_trace': ['preprocess', 'conditioning', 'scheduler', 'denoise', 'decode', 'extract'],
            'checkpoint': {
                **(checkpoint_bundle.get('summary') if isinstance(checkpoint_bundle.get('summary'), dict) else {}),
                'load_style': 'load_state_dict',
                'hydrated_modules': ['conditioner', 'dit', 'vae'],
                'strict': False,
                'hydration': [conditioner_hydration, denoiser_hydration, vae_hydration],
            },
            'preprocess': {
                'processor': image_processor.__class__.__name__,
                'byte_length': reference_asset['byte_length'],
                'normalized_channels': reference_asset['normalized_channels'],
                'image_signature': reference_asset['image_signature'],
                'mask_signature': reference_asset['mask_signature'],
                'signature': reference_asset['signature'],
            },
            'conditioning': {
                'surface_loader': surface_loader.__class__.__name__,
                'encoder': conditioning['encoder'],
                'voxelizer': coarse_surface['voxels']['voxelizer'],
                'voxel_count': conditioning['voxel_count'],
                'mask_tokens': conditioning['mask_tokens'],
                'surface_signature': coarse_surface['mesh']['signature'],
                'voxel_signature': coarse_surface['voxels']['voxel_signature'],
                'image_token_signature': conditioning['image_token_signature'],
                'mask_token_signature': conditioning['mask_token_signature'],
                'checkpoint_signature': conditioning['checkpoint_signature'],
                'conditioning_signature': conditioning['conditioning_signature'],
                'state_hydrated': conditioning['state_hydrated'],
                'signature': conditioning['signature'],
            },
            'scheduler': {
                'family': schedule['family'],
                'target': schedule['target'],
                'step_count': schedule['step_count'],
                'guidance_scale': schedule['guidance_scale'],
                'timestep_signature': schedule['timestep_signature'],
            },
            'denoise': {
                'model': denoised['model'],
                'attention': denoised['attention'],
                'checkpoint_signature': denoised['checkpoint_signature'],
                'scheduler_signature': denoised['scheduler_signature'],
                'latent_signature': denoised['latent_signature'],
                'state_hydrated': denoised['state_hydrated'],
            },
            'decode': {
                'vae': decoded_latents['vae'],
                'decoder': decoded_volume['decoder'],
                'mesh_signature': decoded_volume['mesh_signature'],
                'field_density': decoded_volume['field_density'],
                'field_signature': decoded_volume['field_signature'],
                'state_hydrated': decoded_latents['state_hydrated'],
            },
            'extract': {
                'extractor': refined_surface['extractor'],
                'marching_cubes': refined_surface['marching_cubes'],
                'vertex_count': refined_surface['vertex_count'],
                'face_count': refined_surface['face_count'],
                'payload_bytes': len(refined_payload['bytes']),
                'surface_signature': refined_surface['surface_signature'],
            },
            'gate': {
                'coarse_vertex_count': gate_metrics['coarse_vertex_count'],
                'refined_vertex_count': gate_metrics['refined_vertex_count'],
                'coarse_face_count': gate_metrics['coarse_face_count'],
                'refined_face_count': gate_metrics['refined_face_count'],
            },
        },
        'fallbacks': ['flash_attn->sdpa'] if denoised['attention'] == 'sdpa' else [],
        'subtrees_loaded': checkpoint_bundle['subtrees_loaded'],
        'checkpoint': checkpoint_bundle['path'],
        'execution': {
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
        },
    }
