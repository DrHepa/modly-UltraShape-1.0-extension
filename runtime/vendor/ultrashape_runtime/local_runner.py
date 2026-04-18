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


class LocalRunnerError(Exception):
    def __init__(self, message: str, code: str = PUBLIC_ERROR_CODE):
        super().__init__(message)
        self.code = code


def normalize_public_error_code(code: object) -> str:
    return str(code) if code in PUBLIC_ERROR_CODES else PUBLIC_ERROR_CODE


def collapse_public_message(code: str, message: object) -> str:
    text = str(message).strip() or 'UltraShape local runner reported an execution failure.'

    for prefix in [
        'GEOMETRIC_GATE_REJECTED:',
        'PIPELINE_UNAVAILABLE:',
        'PIPELINE_DEPENDENCY_ERROR:',
        'SURFACE_EXTRACTION_ERROR:',
        'SURFACE_LOAD_ERROR:',
        'REFERENCE_PREPROCESS_ERROR:',
        'MESH_EXPORT_ERROR:',
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if text.startswith(f'{code}:'):
        return text

    return text


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
    load_runtime_config(config_path)
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
        raise LocalRunnerError(collapse_public_message(code, error), code=code) from error

    return {
        'file_path': pipeline_result['file_path'],
        'format': pipeline_result['format'],
        'backend': pipeline_result['backend'],
        'metrics': pipeline_result['metrics'],
        'fallbacks': pipeline_result['fallbacks'],
        'subtrees_loaded': pipeline_result['subtrees_loaded'],
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
                    'error_message': collapse_public_message(code, error),
                }
            )
        )
        sys.stdout.flush()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
