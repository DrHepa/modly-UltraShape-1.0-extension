"""Stable local runner seam for the UltraShape mc-only MVP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .pipelines import run_refine_pipeline


PUBLIC_ERROR_CODE = 'LOCAL_RUNTIME_UNAVAILABLE'
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
            preserve_scale=preserve_scale,
        )
    except Exception as error:
        code = getattr(error, 'code', PUBLIC_ERROR_CODE)
        raise LocalRunnerError(str(error), code=code) from error

    return {
        'file_path': pipeline_result['file_path'],
        'format': pipeline_result['format'],
        'backend': pipeline_result['backend'],
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
        code = getattr(error, 'code', PUBLIC_ERROR_CODE)
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
