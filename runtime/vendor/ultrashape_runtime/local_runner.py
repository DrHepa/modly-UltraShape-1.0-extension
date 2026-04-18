"""Stable local runner seam for the UltraShape mc-only MVP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .pipelines import load_runtime_config, run_refine_pipeline


PUBLIC_ERROR_CODE = 'LOCAL_RUNTIME_UNAVAILABLE'
PUBLIC_ERROR_CODES = {'DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE'}
REQUIRED_FIELDS = {
    'reference_image',
    'coarse_mesh',
    'output_dir',
    'output_format',
    'checkpoint',
    'config_path',
    'ext_dir',
    'backend',
    'steps',
    'guidance_scale',
    'seed',
    'preserve_scale',
}
REQUIRED_CONFIG_MARKERS = ('vae_config', 'dit_cfg', 'conditioner_config', 'image_processor_cfg', 'scheduler_cfg')
REQUIRED_RUNTIME_IMPORTS = ('diffusers', 'cubvh')
REQUIRED_CHECKPOINT_SUBTREES = ('vae', 'dit', 'conditioner')
REQUIRED_PUBLIC_BACKEND_MODES = ('auto', 'local')
REQUIRED_PUBLIC_OUTPUT_FORMATS = ('glb',)
REQUIRED_PUBLIC_ERROR_CODES = ('DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE')


class LocalRunnerError(Exception):
    def __init__(self, message: str, code: str = PUBLIC_ERROR_CODE):
        super().__init__(message)
        self.code = code


def normalize_public_error_code(code: object) -> str:
    return str(code) if code in PUBLIC_ERROR_CODES else PUBLIC_ERROR_CODE


def read_job() -> dict[str, object]:
    line = sys.stdin.readline()
    if not line:
        raise LocalRunnerError('Missing job JSON on stdin.')

    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise LocalRunnerError('Runner job must be a JSON object.')

    missing = sorted(REQUIRED_FIELDS.difference(payload.keys()))
    if missing:
        raise LocalRunnerError(f'Missing required runner fields: {", ".join(missing)}.')

    return payload


def ensure_file(path_value: str, field: str) -> None:
    if not Path(path_value).is_file():
        raise LocalRunnerError(f'{field} is not readable: {path_value}.')


def _truth_error(missing: list[str]) -> LocalRunnerError:
    return LocalRunnerError(
        'Runtime config is missing exact runtime truth markers required for the real-geometry MVP: '
        f'{", ".join(missing)}.'
    )


def load_runtime_contract(config_path: str) -> dict[str, object]:
    config = load_runtime_config(config_path)
    missing: list[str] = []

    runtime = config.get('runtime') if isinstance(config.get('runtime'), dict) else {}
    model = config.get('model') if isinstance(config.get('model'), dict) else {}
    export = config.get('export') if isinstance(config.get('export'), dict) else {}
    checkpoint = config.get('checkpoint') if isinstance(config.get('checkpoint'), dict) else {}
    public_contract = config.get('public_contract') if isinstance(config.get('public_contract'), dict) else {}
    dependencies = config.get('dependencies') if isinstance(config.get('dependencies'), dict) else {}
    required = dependencies.get('required') if isinstance(dependencies.get('required'), dict) else {}
    required_imports = required.get('imports') if isinstance(required.get('imports'), list) else []
    required_subtrees = checkpoint.get('required_subtrees') if isinstance(checkpoint.get('required_subtrees'), list) else []
    public_backend_modes = public_contract.get('backend_modes') if isinstance(public_contract.get('backend_modes'), list) else []
    public_output_formats = public_contract.get('success_output_formats') if isinstance(public_contract.get('success_output_formats'), list) else []
    public_error_codes = public_contract.get('public_error_codes') if isinstance(public_contract.get('public_error_codes'), list) else []

    if runtime.get('requires_exact_closure') is not True:
        missing.append('runtime.requires_exact_closure=true')
    if model.get('scope') != 'mc-only':
        missing.append('model.scope=mc-only')
    if runtime.get('backend') != 'local':
        missing.append('runtime.backend=local')
    if export.get('format') != 'glb':
        missing.append('export.format=glb')
    if tuple(public_backend_modes) != REQUIRED_PUBLIC_BACKEND_MODES:
        missing.append('public_contract.backend_modes=auto,local')
    if tuple(public_output_formats) != REQUIRED_PUBLIC_OUTPUT_FORMATS:
        missing.append('public_contract.success_output_formats=glb')
    if tuple(public_error_codes) != REQUIRED_PUBLIC_ERROR_CODES:
        missing.append('public_contract.public_error_codes=DEPENDENCY_MISSING,WEIGHTS_MISSING,LOCAL_RUNTIME_UNAVAILABLE')

    for marker in REQUIRED_CONFIG_MARKERS:
        if not isinstance(config.get(marker), dict):
            missing.append(marker)

    for module_name in REQUIRED_RUNTIME_IMPORTS:
        if module_name not in required_imports:
            missing.append(f'dependencies.required.imports:{module_name}')

    if tuple(required_subtrees) != REQUIRED_CHECKPOINT_SUBTREES:
        missing.append('checkpoint.required_subtrees=vae,dit,conditioner')

    if missing:
        raise _truth_error(missing)

    return {
        'backend': 'local-only',
        'scope': 'mc-only',
        'output_format': 'glb-only',
        'requires_exact_closure': True,
        'checkpoint_subtrees': list(REQUIRED_CHECKPOINT_SUBTREES),
        'public_error_codes': list(REQUIRED_PUBLIC_ERROR_CODES),
    }


def run_refine_job(
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
    ensure_file(reference_image, 'reference_image')
    ensure_file(coarse_mesh, 'coarse_mesh')
    ensure_file(config_path, 'config_path')

    if backend != 'local':
        raise LocalRunnerError('UltraShape local runner is local-only in this MVP.')
    if output_format != 'glb':
        raise LocalRunnerError('UltraShape local runner is glb-only in this MVP.')
    if not isinstance(steps, int) or steps <= 0:
        raise LocalRunnerError('steps must be a positive integer.')
    if not isinstance(guidance_scale, (int, float)) or guidance_scale <= 0:
        raise LocalRunnerError('guidance_scale must be a positive number.')
    if seed is not None and not isinstance(seed, int):
        raise LocalRunnerError('seed must be an integer or null.')
    if not isinstance(preserve_scale, bool):
        raise LocalRunnerError('preserve_scale must be a boolean.')

    Path(ext_dir)
    runtime_contract = load_runtime_contract(config_path)
    try:
        pipeline_result = run_refine_pipeline(
            reference_image=reference_image,
            coarse_mesh=coarse_mesh,
            output_dir=output_dir,
            output_format=output_format,
            checkpoint=checkpoint,
            config_path=config_path,
            ext_dir=ext_dir,
            backend=backend,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            preserve_scale=preserve_scale,
        )
    except Exception as error:
        code = normalize_public_error_code(getattr(error, 'code', PUBLIC_ERROR_CODE))
        raise LocalRunnerError(str(error), code=code) from error

    return {
        'file_path': pipeline_result['file_path'],
        'format': pipeline_result['format'],
        'backend': pipeline_result['backend'],
        'metrics': pipeline_result['metrics'],
        'fallbacks': pipeline_result['fallbacks'],
        'subtrees_loaded': pipeline_result['subtrees_loaded'],
        'runtime_contract': runtime_contract,
        'warnings': pipeline_result.get('warnings', []),
    }


def main() -> int:
    try:
        job = read_job()
        result = run_refine_job(**job)
        sys.stdout.write(json.dumps({'ok': True, 'result': result}))
        sys.stdout.flush()
        return 0
    except Exception as error:
        code = normalize_public_error_code(getattr(error, 'code', PUBLIC_ERROR_CODE))
        sys.stdout.write(
            json.dumps(
                {
                    'ok': False,
                    'error_code': code,
                    'error_message': str(error),
                }
            )
        )
        sys.stdout.flush()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
