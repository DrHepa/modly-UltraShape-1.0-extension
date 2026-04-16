#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PARAMS = {
    'checkpoint': None,
    'backend': 'auto',
    'steps': 30,
    'guidance_scale': 5.5,
    'seed': None,
    'preserve_scale': True,
    'output_format': 'glb',
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
MESH_EXTENSIONS = {'.glb', '.obj', '.fbx', '.ply'}
ALLOWED_BACKENDS = {'auto', 'local', 'remote', 'hybrid'}
ALLOWED_OUTPUTS = {'glb', 'obj', 'fbx', 'ply'}
PUBLIC_ERROR_CODES = {
    'INVALID_PARAMS',
    'MISSING_INPUT',
    'UNREADABLE_ASSET',
    'UNSUPPORTED_ASSET_TYPE',
    'DEPENDENCY_MISSING',
    'WEIGHTS_MISSING',
    'LOCAL_RUNTIME_UNAVAILABLE',
}


class ProcessorError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + '\n')
    sys.stdout.flush()


def emit_progress(percent: int, label: str) -> None:
    emit({'type': 'progress', 'percent': percent, 'label': label})


def emit_error(error: Exception) -> None:
    code = getattr(error, 'code', 'LOCAL_RUNTIME_UNAVAILABLE')
    if code not in PUBLIC_ERROR_CODES:
        code = 'LOCAL_RUNTIME_UNAVAILABLE'

    message = str(error)
    if not message.startswith(f'{code}:'):
        message = f'{code}: {message}'

    emit({'type': 'error', 'message': message, 'code': code})


def read_payload() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise ProcessorError('INVALID_PARAMS', 'Missing JSON payload line on stdin.')

    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ProcessorError('INVALID_PARAMS', 'Processor payload must be a JSON object.')
    return payload


def normalize_params(payload: dict) -> dict:
    raw_params = payload.get('params')
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise ProcessorError('INVALID_PARAMS', 'params must be an object when provided.')

    merged = {**DEFAULT_PARAMS, **raw_params}

    backend = merged['backend']
    if backend not in ALLOWED_BACKENDS:
        raise ProcessorError('INVALID_PARAMS', 'backend must be auto, local, remote, or hybrid.')
    if not isinstance(merged['steps'], int) or merged['steps'] <= 0:
        raise ProcessorError('INVALID_PARAMS', 'steps must be a positive integer.')
    if not isinstance(merged['guidance_scale'], (int, float)) or merged['guidance_scale'] <= 0:
        raise ProcessorError('INVALID_PARAMS', 'guidance_scale must be a positive number.')
    if merged['seed'] is not None and not isinstance(merged['seed'], int):
        raise ProcessorError('INVALID_PARAMS', 'seed must be an integer or null.')
    if not isinstance(merged['preserve_scale'], bool):
        raise ProcessorError('INVALID_PARAMS', 'preserve_scale must be a boolean.')
    if merged['output_format'] not in ALLOWED_OUTPUTS:
        raise ProcessorError('INVALID_PARAMS', 'output_format must be glb, obj, fbx, or ply.')
    checkpoint = merged.get('checkpoint')
    if checkpoint not in (None, ''):
        ensure_file(str(checkpoint), 'checkpoint', allowed_extensions=None)
        merged['checkpoint'] = str(checkpoint)
    else:
        merged['checkpoint'] = None

    return merged


def resolve_output_dir(payload: dict) -> str:
    for candidate in (
        payload.get('workspaceDir'),
        payload.get('tempDir'),
        nested_get(payload, 'context', 'workspaceDir'),
        nested_get(payload, 'context', 'tempDir'),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    raise ProcessorError(
        'MISSING_INPUT',
        'Processor payload must include workspaceDir or tempDir for output packaging.',
    )


def resolve_input_paths(payload: dict, params: dict) -> tuple[str, str]:
    input_payload = payload.get('input')
    if not isinstance(input_payload, dict):
        raise ProcessorError('MISSING_INPUT', 'Processor payload must include input.')

    named_reference = nested_get(input_payload, 'inputs', 'reference_image', 'filePath')
    named_coarse = nested_get(input_payload, 'inputs', 'coarse_mesh', 'filePath')
    if isinstance(named_reference, str) and named_reference.strip() and isinstance(named_coarse, str) and named_coarse.strip():
        return named_reference.strip(), named_coarse.strip()

    reference_path = input_payload.get('filePath')
    coarse_path = params.get('coarse_mesh')

    if not isinstance(reference_path, str) or not reference_path.strip():
        raise ProcessorError('MISSING_INPUT', 'Missing required field: reference_image.')
    if not isinstance(coarse_path, str) or not coarse_path.strip():
        raise ProcessorError('MISSING_INPUT', 'Missing required field: coarse_mesh.')

    return reference_path.strip(), coarse_path.strip()


def resolve_ext_dir(payload: dict) -> Path:
    for candidate in (
        os.environ.get('ULTRASHAPE_EXT_DIR'),
        payload.get('extDir'),
        nested_get(payload, 'context', 'extDir'),
        Path(__file__).resolve().parent,
        os.getcwd(),
    ):
        if isinstance(candidate, Path):
            return candidate
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate.strip())

    raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Unable to resolve the installed UltraShape runtime directory.')


def ensure_file(path_value: str, field: str, allowed_extensions: set[str] | None) -> None:
    path = Path(path_value)
    if not path.is_file():
        raise ProcessorError('UNREADABLE_ASSET', f'{field} is not readable: {path_value}.')
    if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
        raise ProcessorError('UNSUPPORTED_ASSET_TYPE', f'{field} must use an allowed format.')


def load_runtime_readiness(ext_dir: Path) -> dict:
    readiness_path = ext_dir / '.runtime-readiness.json'
    if not readiness_path.is_file():
        raise ProcessorError(
            'LOCAL_RUNTIME_UNAVAILABLE',
            f'Missing runtime readiness file: {readiness_path}.',
        )

    with readiness_path.open('r', encoding='utf8') as handle:
        readiness = json.load(handle)

    if not isinstance(readiness, dict):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Runtime readiness payload must be a JSON object.')

    return readiness


def select_backend(requested_backend: str) -> str:
    if requested_backend in {'remote', 'hybrid'}:
        raise ProcessorError(
            'LOCAL_RUNTIME_UNAVAILABLE',
            'Remote and hybrid execution are explicitly unsupported in the local-only MVP.',
        )

    return 'local'


def ensure_runtime_ready(readiness: dict) -> None:
    if readiness.get('required_imports_ok') is False:
        missing_required = ', '.join(as_string_list(readiness.get('missing_required')))
        suffix = f' Missing: {missing_required}.' if missing_required else ''
        raise ProcessorError('DEPENDENCY_MISSING', f'Required local UltraShape dependencies are unavailable.{suffix}')

    if readiness.get('status') == 'blocked':
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local UltraShape runtime is blocked for this installation.')

    if readiness.get('backend') not in (None, 'local'):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Installed readiness is not compatible with local UltraShape execution.')

    if readiness.get('status') not in {'ready', 'degraded'}:
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local UltraShape runtime readiness is invalid for execution.')

    if readiness.get('weights_ready') is False:
        missing_weights = ', '.join(as_string_list(readiness.get('missing_required')))
        suffix = f' Missing: {missing_weights}.' if missing_weights else ''
        if readiness.get('status') != 'ready':
            raise ProcessorError(
                'LOCAL_RUNTIME_UNAVAILABLE',
                f'Local UltraShape install readiness is not executable while required weights are absent.{suffix}',
            )
        raise ProcessorError('WEIGHTS_MISSING', f'Local UltraShape weights are not ready.{suffix}')


def ensure_mvp_execution_contract(params: dict, readiness: dict) -> None:
    requested_scope = params.get('mvp_scope') or params.get('scope')
    if requested_scope not in (None, 'mc-only'):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'UltraShape local execution is mc-only in this MVP.')

    if readiness.get('mvp_scope') not in (None, 'mc-only'):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Installed UltraShape runtime is not compatible with mc-only execution.')

    if params['output_format'] != 'glb':
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'UltraShape local execution is glb-only in this MVP.')


def resolve_runner_paths(ext_dir: Path) -> tuple[str, str]:
    python_path = ext_dir / 'venv' / 'bin' / 'python'
    config_path = ext_dir / 'runtime' / 'configs' / 'infer_dit_refine.yaml'

    if not python_path.is_file():
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', f'Missing local runtime Python executable: {python_path}.')
    if not config_path.is_file():
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', f'Missing local runtime config: {config_path}.')

    return str(python_path), str(config_path)


def resolve_checkpoint_path(ext_dir: Path, params: dict) -> str:
    configured = params.get('checkpoint')
    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    return str(ext_dir / 'models' / 'ultrashape' / 'ultrashape_v1.pt')


def run_local_runtime(
    *,
    ext_dir: Path,
    output_dir: str,
    reference_image: str,
    coarse_mesh: str,
    backend: str,
    params: dict,
) -> dict:
    python_path, config_path = resolve_runner_paths(ext_dir)
    runner_runtime_path = ext_dir / 'runtime'
    job = {
        'reference_image': reference_image,
        'coarse_mesh': coarse_mesh,
        'output_dir': output_dir,
        'output_format': params['output_format'],
        'checkpoint': resolve_checkpoint_path(ext_dir, params),
        'config_path': config_path,
        'ext_dir': str(ext_dir),
        'backend': backend,
        'steps': params['steps'],
        'guidance_scale': params['guidance_scale'],
        'seed': params['seed'],
        'preserve_scale': params['preserve_scale'],
    }

    env = dict(os.environ)
    env['PYTHONPATH'] = str(runner_runtime_path)

    try:
        outcome = subprocess.run(
            [python_path, '-m', 'ultrashape_runtime.local_runner'],
            input=json.dumps(job) + '\n',
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as error:
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', f'Unable to launch local runtime runner: {error}.') from error

    stdout = outcome.stdout.strip()
    if not stdout:
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner produced no JSON result on stdout.')

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', f'Local runtime runner returned invalid JSON: {error}.') from error

    if not isinstance(envelope, dict):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner stdout must be a JSON object.')

    if envelope.get('ok') is False:
        code = envelope.get('error_code')
        message = envelope.get('error_message')
        if code not in {'DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE'}:
            code = 'LOCAL_RUNTIME_UNAVAILABLE'
        if not isinstance(message, str) or not message.strip():
            message = 'Local runtime runner reported an unknown execution failure.'
        raise ProcessorError(code, message)

    result = envelope.get('result')
    if envelope.get('ok') is not True or not isinstance(result, dict):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner success envelope is malformed.')

    return validate_runner_result(result, output_dir)


def validate_runner_result(result: dict, output_dir: str) -> dict:
    file_path = result.get('file_path')
    output_format = result.get('format')
    backend = result.get('backend')
    metrics = result.get('metrics')
    fallbacks = result.get('fallbacks')
    subtrees_loaded = result.get('subtrees_loaded')

    if not isinstance(file_path, str) or not file_path.strip():
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner did not return a readable file path.')
    if output_format != 'glb':
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner returned a non-glb artifact.')
    if backend != 'local':
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner returned a non-local backend result.')
    if not isinstance(metrics, dict):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner must return checkpoint-backed execution metadata.')
    if not isinstance(fallbacks, list) or any(not isinstance(item, str) for item in fallbacks):
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', 'Local runtime runner must return fallback metadata as a string list.')
    if subtrees_loaded != ['vae', 'dit', 'conditioner']:
        raise ProcessorError(
            'LOCAL_RUNTIME_UNAVAILABLE',
            'Local runtime runner must report checkpoint-backed execution metadata for vae/dit/conditioner.',
        )

    resolved_output_dir = Path(output_dir).resolve()
    resolved_file = Path(file_path).resolve()

    try:
        resolved_file.relative_to(resolved_output_dir)
    except ValueError as error:
        raise ProcessorError(
            'LOCAL_RUNTIME_UNAVAILABLE',
            'Local runtime runner returned a file path outside the requested output directory.',
        ) from error

    if not resolved_file.is_file():
        raise ProcessorError('LOCAL_RUNTIME_UNAVAILABLE', f'Local runtime runner did not create the reported output file: {resolved_file}.')

    return {
        'file_path': str(resolved_file),
        'backend': backend,
        'metrics': metrics,
        'fallbacks': fallbacks,
        'subtrees_loaded': subtrees_loaded,
    }


def nested_get(payload: dict, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def main() -> int:
    try:
        emit_progress(5, 'validating')
        payload = read_payload()
        params = normalize_params(payload)
        output_dir = resolve_output_dir(payload)
        reference_image, coarse_mesh = resolve_input_paths(payload, params)
        ext_dir = resolve_ext_dir(payload)
        selected_backend = select_backend(params['backend'])
        readiness = load_runtime_readiness(ext_dir)

        ensure_file(reference_image, 'reference_image', IMAGE_EXTENSIONS)
        ensure_file(coarse_mesh, 'coarse_mesh', MESH_EXTENSIONS)

        emit(
            {
                'type': 'log',
                'message': json.dumps(
                    {
                        'reference_image': reference_image,
                        'coarse_mesh': coarse_mesh,
                        'backend': selected_backend,
                        'output_format': params['output_format'],
                        'output_dir': output_dir,
                        'ext_dir': str(ext_dir),
                        'node_id': payload.get('nodeId') or nested_get(payload, 'input', 'nodeId'),
                    }
                ),
            }
        )

        ensure_runtime_ready(readiness)
        ensure_mvp_execution_contract(params, readiness)

        emit_progress(20, f'preflight:{selected_backend}')
        emit_progress(60, f'running:{selected_backend}')
        emit_progress(90, 'packaging')

        runtime_result = run_local_runtime(
            ext_dir=ext_dir,
            output_dir=output_dir,
            reference_image=reference_image,
            coarse_mesh=coarse_mesh,
            backend=selected_backend,
            params=params,
        )
        emit(
            {
                'type': 'log',
                'message': json.dumps(
                    {
                        'backend': runtime_result['backend'],
                        'metrics': runtime_result['metrics'],
                        'fallbacks': runtime_result['fallbacks'],
                        'subtrees_loaded': runtime_result['subtrees_loaded'],
                    }
                ),
            }
        )
        emit({'type': 'done', 'result': {'filePath': runtime_result['file_path']}})
        return 0
    except Exception as error:
        emit_error(error)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
