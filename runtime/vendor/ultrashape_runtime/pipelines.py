"""Executable mc-only pipeline for the vendored UltraShape runtime."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path

from .models.autoencoders.model import ShapeVAE
from .models.autoencoders.surface_extractors import evaluate_geometric_gate, export_refined_glb, extract_surface
from .models.autoencoders.volume_decoders import decode_volume
from .models.conditioner_mask import SingleImageEncoder
from .models.denoisers.dit_mask import RefineDiT
from .preprocessors import ImageProcessorV2
from .real_mode import REAL_MODE_ADAPTER, describe_real_readiness, run_real_refine_pipeline
from .schedulers import build_flow_matching_schedule
from .surface_loaders import SharpEdgeSurfaceLoader
from .utils.checkpoint import apply_checkpoint_state, load_checkpoint_subtrees
from .utils import blend_sequences, clamp_unit, stable_signature
from .utils.misc import instantiate_from_config, load_omega_config


class PipelineDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


class PipelineUnavailableError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def build_refine_pipeline() -> dict[str, str]:
    return {
        'name': 'ultrashape-refine',
        'scope': 'mc-only',
        'entrypoint': 'scripts.infer_dit_refine.run_inference',
        'loader': 'load_models',
        'class': 'UltraShapePipeline',
    }


def _requested_runtime_mode() -> str:
    candidate = os.environ.get('ULTRASHAPE_RUNTIME_MODE', 'auto').strip().lower()
    return candidate if candidate in {'auto', 'real', 'portable'} else 'auto'


def resolve_runtime_mode(*, checkpoint: str | None = None, config_path: str | None = None, upstream_config_path: str | None = None) -> dict[str, object]:
    requested = _requested_runtime_mode()
    real = describe_real_readiness(
        checkpoint=checkpoint,
        runtime_config_path=config_path,
        upstream_config_path=upstream_config_path,
    ) if requested in {'auto', 'real'} else {
        'available': False,
        'adapter': REAL_MODE_ADAPTER,
        'source': None,
        'entrypoint': 'scripts.infer_dit_refine.run_inference',
        'blockers': [],
        'reason': 'Real readiness was bypassed because portable mode was forced.',
    }
    portable = {
        'available': True,
        'authoritative': False,
        'reason': 'Portable fallback is active for reduced environments; it is not the authoritative upstream closure.',
    }

    if requested == 'real':
        active = 'real' if bool(real.get('available')) else None
    elif requested == 'portable':
        active = 'portable'
    else:
        active = 'real' if bool(real.get('available')) else 'portable'

    selection = 'real-available' if active == 'real' else 'portable-only'
    if active is None:
        selection = 'blocked'
    return {
        'selection': selection,
        'requested': requested,
        'active': active,
        'real': real,
        'portable': portable,
    }


def retrieve_timesteps(schedule: dict[str, object]) -> tuple[list[float], int]:
    timesteps = _numeric_list(schedule.get('consumed_timesteps'))
    if not timesteps:
        timesteps = _numeric_list(schedule.get('timesteps'))
    return timesteps, len(timesteps)


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

    try:
        config = _normalize_runtime_config(load_omega_config(str(path)))
        _validate_exact_closure_config(config)
        return config
    except Exception:
        config, _ = _parse_block(path.read_text(encoding='utf8').splitlines(), 0, 0)
    if not isinstance(config, dict):
        raise PipelineUnavailableError('Runtime config root must be a mapping.')
    config = _normalize_runtime_config(config)
    _validate_exact_closure_config(config)
    return config


def describe_upstream_closure_markers(config_path: str) -> dict[str, object]:
    config = load_runtime_config(config_path)
    checkpoint = config.get('checkpoint') if isinstance(config.get('checkpoint'), dict) else {}
    return {
        'config_loader': 'OmegaConf.load',
        'checkpoint_hydration': 'strict',
        'preserve_scale_restore': 'coarse-normalization-inverse',
        'required_subtrees': list(checkpoint.get('required_subtrees', ['vae', 'dit', 'conditioner'])),
    }


def _mapping(candidate: object) -> dict[str, object]:
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _numeric_triplet(candidate: object) -> tuple[float, float, float] | None:
    if not isinstance(candidate, list) or len(candidate) != 3:
        return None
    values: list[float] = []
    for axis in candidate:
        if not isinstance(axis, (int, float)):
            return None
        values.append(float(axis))
    return (values[0], values[1], values[2])


def _restore_original_scale(mesh_payload: dict[str, object], normalization_transform: dict[str, object]) -> dict[str, object]:
    vertices = mesh_payload.get('vertices') if isinstance(mesh_payload.get('vertices'), list) else []
    center = _numeric_triplet(normalization_transform.get('center'))
    scale_factor = normalization_transform.get('scale_factor')
    if center is None or not isinstance(scale_factor, (int, float)) or float(scale_factor) == 0.0:
        return dict(mesh_payload)

    restored_vertices = []
    for vertex in vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) < 3:
            continue
        restored_vertices.append(
            (
                round((float(vertex[0]) / float(scale_factor)) + center[0], 6),
                round((float(vertex[1]) / float(scale_factor)) + center[1], 6),
                round((float(vertex[2]) / float(scale_factor)) + center[2], 6),
            )
        )

    return {
        **mesh_payload,
        'vertices': restored_vertices,
        'bytes': None,
        'is_binary_glb': False,
        'restored_from_normalized': True,
    }


def _normalize_runtime_config(config: dict[str, object]) -> dict[str, object]:
    normalized = dict(config)
    model = _mapping(normalized.get('model'))
    params = _mapping(model.get('params'))

    if params:
        for source_key, target_key in {
            'vae_config': 'vae_config',
            'dit_cfg': 'dit_cfg',
            'conditioner_config': 'conditioner_config',
            'scheduler_cfg': 'scheduler_cfg',
            'image_processor_cfg': 'image_processor_cfg',
            'surface_loader_cfg': 'surface_loader_cfg',
        }.items():
            if source_key in params and target_key not in normalized:
                normalized[target_key] = params[source_key]

    scheduler = _mapping(normalized.get('scheduler'))
    scheduler_cfg = _mapping(normalized.get('scheduler_cfg'))
    if 'target' not in scheduler and isinstance(scheduler_cfg.get('target'), str):
        scheduler['target'] = scheduler_cfg['target']
    if scheduler:
        normalized['scheduler'] = scheduler

    surface = _mapping(normalized.get('surface'))
    surface_loader_cfg = _mapping(normalized.get('surface_loader_cfg'))
    if 'target' not in surface and isinstance(surface_loader_cfg.get('target'), str):
        surface['target'] = surface_loader_cfg['target']
    if surface:
        normalized['surface'] = surface

    return normalized


def _validate_exact_closure_config(config: dict[str, object]) -> None:
    runtime = config.get('runtime') if isinstance(config.get('runtime'), dict) else None
    if not isinstance(runtime, dict) or runtime.get('requires_exact_closure') is not True:
        raise PipelineUnavailableError('Runtime config requires_exact_closure: true for the upstream closure path.')

    required_sections = {
        'checkpoint': 'checkpoint subtree loading',
        'preprocess': 'reference preprocessing',
        'conditioning': 'surface + voxel conditioning',
        'scheduler': 'scheduler wiring',
        'decoder': 'volume decode wiring',
        'surface': 'surface extraction wiring',
        'gate': 'geometric gate wiring',
        'export': 'refined mesh export wiring',
    }

    missing = [name for name in required_sections if not isinstance(config.get(name), dict)]
    if missing:
        raise PipelineUnavailableError(
            'Runtime config requires_exact_closure: true but is missing closure sections: ' + ', '.join(sorted(missing)) + '.'
        )


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


def _instantiate_runtime_component(config: dict[str, object], key: str, fallback_cls, /, **kwargs):
    component_config = config.get(key)
    if isinstance(component_config, Mapping) and isinstance(component_config.get('target'), str):
        return instantiate_from_config(component_config, **kwargs)
    return fallback_cls(**kwargs)


def _numeric_list(candidate: object) -> list[float]:
    if not isinstance(candidate, list):
        return []
    return [float(value) for value in candidate if isinstance(value, (int, float))]


def _reference_asset_from_cond_inputs(cond_inputs: dict[str, object]) -> dict[str, object]:
    image_tensor = cond_inputs.get('image')
    mask_tensor = cond_inputs.get('mask')
    image_values = _flatten_numeric(image_tensor)
    mask_values = _flatten_numeric(mask_tensor)
    return {
        'processor': 'ImageProcessorV2',
        'image_tensor': image_tensor,
        'mask_tensor': mask_tensor,
        'image_tensor_shape': getattr(image_tensor, 'shape', None),
        'mask_tensor_shape': getattr(mask_tensor, 'shape', None),
        'image_features': image_values,
        'mask_features': mask_values,
        'mean_intensity': clamp_unit(sum(image_values) / len(image_values) if image_values else 0.0),
        'mask_coverage': clamp_unit(sum(mask_values) / len(mask_values) if mask_values else 1.0),
        'cutout_applied': False,
        'normalized_channels': 4,
        'byte_length': len(image_values),
        'source_format': 'tensor',
        'pixel_count': max(len(image_values) // 4, 1),
        'image_signature': stable_signature(image_values),
        'mask_signature': stable_signature(mask_values),
        'signature': stable_signature(image_values + mask_values),
    }


class DiTPipeline:
    def __init__(self, *, vae, model, scheduler, conditioner, image_processor):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor

    def prepare_image(self, image, mask=None) -> dict[str, object]:
        if isinstance(image, dict):
            return dict(image)
        if hasattr(self.image_processor, 'process') and isinstance(image, str):
            processed = self.image_processor.process(image)
            return {'image': processed.get('image_tensor'), 'mask': processed.get('mask_tensor'), 'reference_asset': processed}
        processed = self.image_processor(image, mask=mask) if callable(self.image_processor) else {'image': image, 'mask': mask}
        if not isinstance(processed, dict):
            processed = {'image': image, 'mask': mask}
        processed.setdefault('image', image)
        processed.setdefault('mask', mask)
        return processed

    def encode_cond(self, *, reference_asset: dict[str, object], coarse_surface: dict[str, object]) -> dict[str, object]:
        return self.conditioner.build(reference_asset=reference_asset, coarse_surface=coarse_surface)

    def prepare_latents(self, *, conditioning: dict[str, object], coarse_surface: dict[str, object]) -> list[float]:
        return _initialize_latents(conditioning=conditioning, coarse_surface=coarse_surface)


class UltraShapePipeline(DiTPipeline):
    def __call__(
        self,
        *,
        image,
        voxel_cond: dict[str, object] | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int | None = None,
        coarse_surface: dict[str, object] | None = None,
        reference_asset: dict[str, object] | None = None,
        preserve_scale: bool = False,
        output_type: str = 'mesh',
        **kwargs,
    ):
        del kwargs
        cond_inputs = self.prepare_image(image)
        runtime_reference_asset = reference_asset or cond_inputs.get('reference_asset')
        if not isinstance(runtime_reference_asset, dict):
            runtime_reference_asset = _reference_asset_from_cond_inputs(cond_inputs)
        if not isinstance(coarse_surface, dict):
            raise PipelineUnavailableError('UltraShapePipeline requires a structured coarse_surface payload for faithful conditioning.')
        runtime_coarse_surface = coarse_surface
        conditioning = self.encode_cond(reference_asset=runtime_reference_asset, coarse_surface=runtime_coarse_surface)
        schedule = self.scheduler if isinstance(self.scheduler, dict) else build_flow_matching_schedule(steps=num_inference_steps, guidance_scale=guidance_scale)
        timesteps, _ = retrieve_timesteps(schedule)
        latents = self.prepare_latents(conditioning=conditioning, coarse_surface=runtime_coarse_surface)
        denoised = self.model.denoise(
            latents=latents,
            timesteps=timesteps,
            context=_numeric_list(conditioning.get('context')),
            context_mask=_numeric_list(conditioning.get('context_mask')),
            voxel_cond=runtime_coarse_surface.get('voxel_cond') if isinstance(runtime_coarse_surface.get('voxel_cond'), dict) else {},
            guidance_scale=guidance_scale,
            schedule=schedule,
            seed=seed,
        )
        decoded_latents = self.vae.decode_latents(
            denoised,
            runtime_reference_asset,
            conditioning=conditioning,
            coarse_surface=runtime_coarse_surface,
        )
        decoded_volume = decode_volume(decoded_latents)
        execution_trace = ['prepare_image', 'encode_cond', 'prepare_latents', 'diffusion_sampling', 'decode', 'extract']
        if output_type == 'latent':
            return {
                **decoded_volume,
                'authority': 'UltraShapePipeline._export',
                'execution_trace': execution_trace,
            }, denoised
        refined_surface = extract_surface(
            extraction='mc',
            coarse_surface=runtime_coarse_surface,
            reference_asset=runtime_reference_asset,
            decoded_volume=decoded_volume,
            preserve_scale=preserve_scale,
        )
        refined_surface = {
            **refined_surface,
            'authority': 'UltraShapePipeline._export',
            'execution_trace': execution_trace,
        }
        return refined_surface, denoised


def load_runtime_components(
    *,
    config: dict[str, object],
    checkpoint: str | None,
    ext_dir: str,
) -> tuple[dict[str, object], dict[str, object], object, object, object, object, object, dict[str, object], dict[str, object], dict[str, object]]:
    checkpoint_config = config.get('checkpoint') if isinstance(config.get('checkpoint'), dict) else {}
    weights = config.get('weights') if isinstance(config.get('weights'), dict) else {}
    image_processor = _instantiate_runtime_component(config, 'image_processor_cfg', ImageProcessorV2)
    surface_loader = _instantiate_runtime_component(config, 'surface_loader_cfg', SharpEdgeSurfaceLoader)
    checkpoint_bundle = load_checkpoint_subtrees(
        checkpoint,
        weights.get('primary') if isinstance(weights, dict) else None,
        ext_dir,
        checkpoint_config.get('required_subtrees') if isinstance(checkpoint_config.get('required_subtrees'), list) else None,
    )
    checkpoint_state = checkpoint_bundle['bundle'] if isinstance(checkpoint_bundle.get('bundle'), dict) else {}
    conditioner = _instantiate_runtime_component(
        config,
        'conditioner_config',
        SingleImageEncoder,
        checkpoint_state=checkpoint_state.get('conditioner'),
    )
    conditioner_hydration = apply_checkpoint_state(conditioner, checkpoint_state.get('conditioner'), strict=True)
    conditioner.hydration = conditioner_hydration
    denoiser = _instantiate_runtime_component(config, 'dit_cfg', RefineDiT, checkpoint_state=checkpoint_state.get('dit'))
    denoiser_hydration = apply_checkpoint_state(denoiser, checkpoint_state.get('dit'), strict=True)
    denoiser.hydration = denoiser_hydration
    vae = _instantiate_runtime_component(config, 'vae_config', ShapeVAE, checkpoint_state=checkpoint_state.get('vae'))
    vae_hydration = apply_checkpoint_state(vae, checkpoint_state.get('vae'), strict=True)
    return (
        checkpoint_bundle,
        checkpoint_state,
        image_processor,
        surface_loader,
        conditioner,
        denoiser,
        vae,
        conditioner_hydration,
        denoiser_hydration,
        vae_hydration,
    )


def _voxel_seed(voxel_cond: dict[str, object], *, limit: int = 8) -> list[float]:
    coords = voxel_cond.get('coords') if isinstance(voxel_cond.get('coords'), list) else []
    occupancies = _numeric_list(voxel_cond.get('occupancies'))
    resolution = int(voxel_cond.get('resolution', 0)) if isinstance(voxel_cond.get('resolution'), int) else 0
    signal: list[float] = []
    scale = float(max(resolution, 1))

    for index, coord in enumerate(coords[:limit]):
        if isinstance(coord, (list, tuple)):
            signal.extend(
                clamp_unit(float(axis) / scale)
                for axis in coord[:3]
                if isinstance(axis, (int, float))
            )
        if index < len(occupancies):
            signal.append(clamp_unit(float(occupancies[index])))
        if len(signal) >= limit:
            break

    return signal[:limit]


def _flatten_numeric(candidate: object) -> list[float]:
    values: list[float] = []

    def visit(node: object) -> None:
        if isinstance(node, (int, float)):
            values.append(float(node))
            return
        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(candidate)
    return values


def _surface_seed(coarse_surface: dict[str, object], *, limit: int = 8) -> list[float]:
    mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
    bounds = mesh.get('bounds') if isinstance(mesh.get('bounds'), dict) else {}
    minimum = bounds.get('min') if isinstance(bounds.get('min'), tuple) else (0.0, 0.0, 0.0)
    extents = bounds.get('extents') if isinstance(bounds.get('extents'), tuple) else (0.0, 0.0, 0.0)
    mesh_scale = max(*(float(axis) for axis in extents), 1.0)
    sampled_surface_points = coarse_surface.get('sampled_surface_points') if isinstance(coarse_surface.get('sampled_surface_points'), list) else []

    signal = [
        clamp_unit(float(axis) / mesh_scale)
        for axis in extents[:3]
        if isinstance(axis, (int, float))
    ]

    for point in sampled_surface_points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        for axis_index, axis in enumerate(point[:3]):
            if not isinstance(axis, (int, float)):
                continue
            minimum_axis = float(minimum[axis_index]) if axis_index < len(minimum) else 0.0
            extent_axis = float(extents[axis_index]) if axis_index < len(extents) and float(extents[axis_index]) > 0.0 else 1.0
            signal.append(clamp_unit((float(axis) - minimum_axis) / extent_axis))
            if len(signal) >= limit:
                return signal[:limit]

    return signal[:limit]


def _initialize_latents(
    *,
    conditioning: dict[str, object],
    coarse_surface: dict[str, object],
) -> list[float]:
    context = _numeric_list(conditioning.get('context'))
    unconditional_context = _numeric_list(conditioning.get('unconditional_context'))
    context_vectors = _flatten_numeric(conditioning.get('context_vectors'))
    voxel_cond = coarse_surface.get('voxel_cond') if isinstance(coarse_surface.get('voxel_cond'), dict) else {}
    latent_basis = blend_sequences(
        context_vectors,
        context,
        unconditional_context,
        _voxel_seed(voxel_cond),
        _surface_seed(coarse_surface),
    )[:8]

    if not latent_basis:
        latent_basis = [0.0]

    return [clamp_unit(value) for value in latent_basis]


def _signature_parts(*candidates: object) -> list[float]:
    parts: list[float] = []
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            parts.append(float(candidate))
        elif isinstance(candidate, list):
            parts.extend(float(value) for value in candidate if isinstance(value, (int, float)))
    return parts


def _causality_metrics(
    *,
    reference_asset: dict[str, object],
    coarse_surface: dict[str, object],
    conditioning: dict[str, object],
    denoised: dict[str, object],
    decoded_volume: dict[str, object],
    refined_surface: dict[str, object],
    gate_metrics: dict[str, object],
) -> dict[str, object]:
    metadata = conditioning.get('metadata') if isinstance(conditioning.get('metadata'), dict) else {}
    mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
    voxels = coarse_surface.get('voxels') if isinstance(coarse_surface.get('voxels'), dict) else {}
    denoise_inputs = denoised.get('inputs') if isinstance(denoised.get('inputs'), dict) else {}
    denoise_context = denoise_inputs.get('context') if isinstance(denoise_inputs.get('context'), dict) else {}
    denoise_context_mask = denoise_inputs.get('context_mask') if isinstance(denoise_inputs.get('context_mask'), dict) else {}
    denoise_voxel = denoise_inputs.get('voxel_cond') if isinstance(denoise_inputs.get('voxel_cond'), dict) else {}

    image_fingerprint = int(reference_asset.get('image_signature', 0)) if isinstance(reference_asset.get('image_signature'), int) else 0
    mesh_fingerprint = stable_signature(
        _signature_parts(
            mesh.get('signature'),
            voxels.get('voxel_signature'),
            mesh.get('vertex_count'),
            mesh.get('face_count'),
            voxels.get('voxel_count'),
        )
    )
    checkpoint_fingerprint = stable_signature(
        _signature_parts(
            metadata.get('checkpoint_signature'),
            denoised.get('checkpoint_signature'),
            conditioning.get('checkpoint_signature'),
        )
    )
    context_fingerprint = stable_signature(
        _signature_parts(
            metadata.get('context_signature'),
            metadata.get('context_mask_signature'),
            denoise_context.get('signature'),
            denoise_context_mask.get('signature'),
            denoise_voxel.get('signature'),
        )
    )
    latent_fingerprint = stable_signature(
        _signature_parts(
            denoised.get('latent_signature'),
            denoised.get('per_step_signatures'),
        )
    )
    field_fingerprint = stable_signature(
        _signature_parts(
            decoded_volume.get('field_signature'),
            decoded_volume.get('corner_signature'),
            decoded_volume.get('occupied_cell_count'),
        )
    )
    geometry_fingerprint = stable_signature(
        _signature_parts(
            refined_surface.get('surface_signature'),
            gate_metrics.get('refined_signature'),
            refined_surface.get('vertex_count'),
            refined_surface.get('face_count'),
        )
    )

    return {
        'proof': 'fingerprint-causality',
        'image_fingerprint': image_fingerprint,
        'mesh_fingerprint': mesh_fingerprint,
        'checkpoint_fingerprint': checkpoint_fingerprint,
        'context_fingerprint': context_fingerprint,
        'latent_fingerprint': latent_fingerprint,
        'field_fingerprint': field_fingerprint,
        'geometry_fingerprint': geometry_fingerprint,
    }


def _stage_evidence(
    *,
    checkpoint_bundle: dict[str, object],
    reference_asset: dict[str, object],
    coarse_surface: dict[str, object],
    conditioning: dict[str, object],
    schedule: dict[str, object],
    denoised: dict[str, object],
    decoded_volume: dict[str, object],
    refined_surface: dict[str, object],
    exported_payload_bytes: int,
) -> dict[str, object]:
    checkpoint_path = checkpoint_bundle.get('path')
    checkpoint_summary = checkpoint_bundle.get('summary') if isinstance(checkpoint_bundle.get('summary'), dict) else {}
    mesh = coarse_surface.get('mesh') if isinstance(coarse_surface.get('mesh'), dict) else {}
    voxels = coarse_surface.get('voxels') if isinstance(coarse_surface.get('voxels'), dict) else {}
    metadata = conditioning.get('metadata') if isinstance(conditioning.get('metadata'), dict) else {}

    return {
        'preprocess': {
            'source': 'reference_image',
            'processor': reference_asset.get('processor', 'ImageProcessorV2'),
            'image_signature': reference_asset.get('image_signature'),
            'mask_signature': reference_asset.get('mask_signature'),
        },
        'checkpoint': {
            'source': Path(str(checkpoint_path)).name if checkpoint_path else 'ultrashape_v1.pt',
            'subtrees_loaded': checkpoint_bundle.get('subtrees_loaded', []),
            'signature': checkpoint_summary.get('signature'),
        },
        'conditioning': {
            'source': 'coarse_mesh+reference_image+checkpoint',
            'surface_signature': mesh.get('signature'),
            'voxel_signature': voxels.get('voxel_signature'),
            'checkpoint_signature': metadata.get('checkpoint_signature'),
        },
        'scheduler': {
            'source': schedule.get('target', 'diffusers.FlowMatchEulerDiscreteScheduler'),
            'timestep_signature': schedule.get('timestep_signature'),
        },
        'denoise': {
            'source': denoised.get('model', 'RefineDiT'),
            'latent_signature': denoised.get('latent_signature'),
            'checkpoint_signature': denoised.get('checkpoint_signature'),
        },
        'decode': {
            'source': f"{decoded_volume.get('vae', 'ShapeVAE')}+{decoded_volume.get('decoder', 'VanillaVDMVolumeDecoding')}",
            'field_signature': decoded_volume.get('field_signature'),
            'corner_signature': decoded_volume.get('corner_signature'),
        },
        'extract': {
            'source': refined_surface.get('marching_cubes', 'cubvh.sparse_marching_cubes'),
            'surface_signature': refined_surface.get('surface_signature'),
            'payload_bytes': exported_payload_bytes,
        },
    }


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
    upstream_config_path: str | None = None,
) -> dict[str, object]:
    config = load_runtime_config(config_path)
    require_imports(config)
    _, extraction = validate_mvp_scope(config, backend, output_format)
    runtime_mode = resolve_runtime_mode(checkpoint=checkpoint, config_path=config_path, upstream_config_path=upstream_config_path)

    if runtime_mode.get('active') == 'real':
        return run_real_refine_pipeline(
            reference_image=reference_image,
            coarse_mesh=coarse_mesh,
            output_dir=output_dir,
            output_format=output_format,
            checkpoint=checkpoint,
            config_path=config_path,
            upstream_config_path=upstream_config_path,
            ext_dir=ext_dir,
            backend=backend,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            preserve_scale=preserve_scale,
        )
    if runtime_mode.get('requested') == 'real':
        return run_real_refine_pipeline(
            reference_image=reference_image,
            coarse_mesh=coarse_mesh,
            output_dir=output_dir,
            output_format=output_format,
            checkpoint=checkpoint,
            config_path=config_path,
            upstream_config_path=upstream_config_path,
            ext_dir=ext_dir,
            backend=backend,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            preserve_scale=preserve_scale,
        )

    (
        checkpoint_bundle,
        checkpoint_state,
        image_processor,
        surface_loader,
        conditioner,
        denoiser,
        vae,
        conditioner_hydration,
        denoiser_hydration,
        vae_hydration,
    ) = load_runtime_components(config=config, checkpoint=checkpoint, ext_dir=ext_dir)
    reference_asset = image_processor.process(reference_image)
    coarse_surface = surface_loader.load(coarse_mesh)
    schedule = build_flow_matching_schedule(steps=steps, guidance_scale=guidance_scale)
    pipeline = UltraShapePipeline(
        vae=vae,
        model=denoiser,
        scheduler=schedule,
        conditioner=conditioner,
        image_processor=image_processor,
    )
    conditioning = conditioner.build(reference_asset=reference_asset, coarse_surface=coarse_surface)
    refined_surface, denoised = pipeline(
        image=reference_asset,
        voxel_cond=coarse_surface.get('voxel_cond') if isinstance(coarse_surface.get('voxel_cond'), dict) else {},
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        preserve_scale=preserve_scale,
    )
    decoded_latents = vae.decode_latents(denoised, reference_asset, conditioning=conditioning, coarse_surface=coarse_surface)
    decoded_volume = decode_volume(decoded_latents)
    normalization_transform = _mapping(coarse_surface.get('normalization_transform'))
    refined_payload = dict(refined_surface['payload'])
    if preserve_scale and normalization_transform:
        refined_payload = _restore_original_scale(refined_payload, normalization_transform)

    coarse_gate_payload = coarse_surface.get('original_mesh') if preserve_scale else coarse_surface.get('mesh')
    if not isinstance(coarse_gate_payload, dict):
        coarse_gate_payload = coarse_surface['mesh']

    gate_metrics = evaluate_geometric_gate(
        coarse_mesh_payload=coarse_gate_payload,
        refined_mesh_payload=refined_payload,
        preserve_scale=preserve_scale,
        reference_image_signature=reference_asset.get('image_signature'),
        checkpoint_signature=conditioning.get('checkpoint_signature'),
    )
    output_path = export_refined_glb(
        output_dir=output_dir,
        output_format=output_format,
        mesh_payload=refined_payload,
    )

    if not Path(output_path).is_file():
        raise PipelineUnavailableError(f'Expected refined.glb output was not generated: {output_path}.')

    exported_payload_bytes = Path(output_path).stat().st_size
    causality = _causality_metrics(
        reference_asset=reference_asset,
        coarse_surface=coarse_surface,
        conditioning=conditioning,
        denoised=denoised,
        decoded_volume=decoded_volume,
        refined_surface=refined_surface,
        gate_metrics=gate_metrics,
    )

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
                'strict': True,
                'hydration': [conditioner_hydration, denoiser_hydration, vae_hydration],
            },
            'pipeline': {
                'entrypoint': 'scripts.infer_dit_refine.run_inference',
                'loader': 'load_models',
                'class': pipeline.__class__.__name__,
                'returns_latents': True,
                'execution_trace': refined_surface.get('execution_trace', []),
            },
            'runtime_mode': {
                **runtime_mode,
                'real': {
                    **(runtime_mode.get('real') if isinstance(runtime_mode.get('real'), dict) else {}),
                    'adapter': REAL_MODE_ADAPTER,
                },
            },
            'preprocess': {
                'processor': image_processor.__class__.__name__,
                'byte_length': reference_asset['byte_length'],
                'normalized_channels': reference_asset['normalized_channels'],
                'source_format': reference_asset['source_format'],
                'pixel_count': reference_asset['pixel_count'],
                'image_tensor_shape': reference_asset['image_tensor_shape'],
                'image_feature_count': len(reference_asset['image_features']),
                'mask_feature_count': len(reference_asset['mask_features']),
                'mean_intensity': reference_asset['mean_intensity'],
                'mask_coverage': reference_asset['mask_coverage'],
                'cutout_applied': reference_asset['cutout_applied'],
                'image_signature': reference_asset['image_signature'],
                'mask_signature': reference_asset['mask_signature'],
                'signature': reference_asset['signature'],
            },
            'conditioning': {
                'surface_loader': surface_loader.__class__.__name__,
                'encoder': conditioning['encoder'],
                'voxelizer': coarse_surface['voxels']['voxelizer'],
                'surface_vertex_count': coarse_surface['mesh']['vertex_count'],
                'surface_face_count': coarse_surface['mesh']['face_count'],
                'surface_point_count': coarse_surface['mesh']['surface_point_count'],
                'surface_bounds': coarse_surface['mesh']['bounds'],
                'normalization_transform': normalization_transform,
                'voxel_count': conditioning['voxel_count'],
                'voxel_resolution': coarse_surface['voxels']['resolution'],
                'context_token_count': len(conditioning['context']),
                'context_mask_token_count': len(conditioning['context_mask']),
                'cfg_pairing': conditioning['cfg_pairing'],
                'conditioner_metadata': conditioning['metadata'],
                'surface_signature': coarse_surface['mesh']['signature'],
                'voxel_signature': coarse_surface['voxels']['voxel_signature'],
                'state_hydrated': conditioning['state_hydrated'],
                'hydration': conditioning['hydration'],
            },
            'scheduler': {
                'family': schedule['family'],
                'target': schedule['target'],
                'object_type': schedule['object_type'],
                'step_count': schedule['step_count'],
                'guidance_scale': schedule['guidance_scale'],
                'timestep_count': schedule['timestep_count'],
                'timestep_signature': schedule['timestep_signature'],
                'consumed_timesteps': schedule['consumed_timesteps'],
                'consumed_sigmas': schedule['consumed_sigmas'],
                'sigma_start': schedule['sigma_start'],
                'sigma_end': schedule['sigma_end'],
            },
            'denoise': {
                'model': denoised['model'],
                'attention': denoised['attention'],
                'checkpoint_signature': denoised['checkpoint_signature'],
                'inputs': denoised['inputs'],
                'latent_signature': denoised['latent_signature'],
                'latent_count': denoised['latent_count'],
                'latent_mean': denoised['latent_mean'],
                'timestep_count': denoised['timestep_count'],
                'state_hydrated': denoised['state_hydrated'],
                'hydration': denoised['hydration'],
            },
            'decode': {
                'vae': decoded_latents['vae'],
                'decoder': decoded_volume['decoder'],
                'authority': decoded_volume['authority'],
                'input_latent_signature': denoised['latent_signature'],
                'mesh_signature': decoded_volume['mesh_signature'],
                'field_density': decoded_volume['field_density'],
                'field_signature': decoded_volume['field_signature'],
                'field_grid_signature': decoded_volume['field_grid_signature'],
                'field_grid_shape': decoded_volume['field_grid_shape'],
                'field_value_count': decoded_volume['field_value_count'],
                'cell_count': decoded_volume['cell_count'],
                'corner_count': decoded_volume['corner_count'],
                'corner_signature': decoded_volume['corner_signature'],
                'occupied_cell_count': decoded_volume['occupied_cell_count'],
                'occupied_grid_cells': decoded_volume['occupied_grid_cells'],
                'grid_resolution': decoded_volume['grid_resolution'],
                'state_hydrated': decoded_latents['state_hydrated'],
            },
            'extract': {
                'extractor': refined_surface['extractor'],
                'marching_cubes': refined_surface['marching_cubes'],
                'authority': refined_surface['authority'],
                'vertex_count': refined_surface['vertex_count'],
                'face_count': refined_surface['face_count'],
                'payload_bytes': exported_payload_bytes,
                'source_cell_count': refined_surface['source_cell_count'],
                'source_field_signature': refined_surface['source_field_signature'],
                'source_corner_signature': refined_surface['source_corner_signature'],
                'surface_signature': refined_surface['surface_signature'],
            },
            'gate': {
                'coarse_signature': gate_metrics['coarse_signature'],
                'refined_signature': gate_metrics['refined_signature'],
                'reference_image_signature': gate_metrics['reference_image_signature'],
                'checkpoint_signature': gate_metrics['checkpoint_signature'],
                'preserve_scale': gate_metrics['preserve_scale'],
                'scale_fit_applied': gate_metrics['scale_fit_applied'],
                'scale_restore_applied': bool(preserve_scale and normalization_transform),
                'attribution_signature': gate_metrics['attribution_signature'],
                'coarse_vertex_count': gate_metrics['coarse_vertex_count'],
                'refined_vertex_count': gate_metrics['refined_vertex_count'],
                'coarse_face_count': gate_metrics['coarse_face_count'],
                'refined_face_count': gate_metrics['refined_face_count'],
            },
            'causality': causality,
            'stage_evidence': _stage_evidence(
                checkpoint_bundle=checkpoint_bundle,
                reference_asset=reference_asset,
                coarse_surface=coarse_surface,
                conditioning=conditioning,
                schedule=schedule,
                denoised=denoised,
                decoded_volume=decoded_volume,
                refined_surface=refined_surface,
                exported_payload_bytes=exported_payload_bytes,
            ),
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
