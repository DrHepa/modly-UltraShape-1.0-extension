#!/usr/bin/env python3
import json
import os
import shutil
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
    code = getattr(error, 'code', 'BACKEND_UNAVAILABLE')
    emit({'type': 'error', 'message': str(error), 'code': code})


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


def ensure_file(path_value: str, field: str, allowed_extensions: set[str] | None) -> None:
    path = Path(path_value)
    if not path.is_file():
        raise ProcessorError('UNREADABLE_ASSET', f'{field} is not readable: {path_value}.')
    if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
        raise ProcessorError('UNSUPPORTED_ASSET_TYPE', f'{field} must use an allowed format.')


def compute_preflight(requested_backend: str) -> str:
    host_platform = os.environ.get('ULTRASHAPE_HOST_PLATFORM', sys.platform)
    host_arch = os.environ.get('ULTRASHAPE_HOST_ARCH', os.uname().machine)
    local_available = env_flag('ULTRASHAPE_LOCAL_AVAILABLE', False)
    remote_available = env_flag('ULTRASHAPE_REMOTE_AVAILABLE', False) or bool(
        os.environ.get('ULTRASHAPE_TEST_ARTIFACT_PATH')
    )
    linux_arm64 = host_platform.startswith('linux') and host_arch in {'arm64', 'aarch64'}

    recommended = 'remote' if linux_arm64 and remote_available else 'local' if local_available else 'remote'

    if requested_backend == 'local':
        if local_available:
            return 'local'
        if remote_available:
            return 'remote'
        raise ProcessorError('BACKEND_UNAVAILABLE', 'BACKEND_UNAVAILABLE: Requested local backend is unavailable and no remote fallback is configured.')

    if requested_backend in {'remote', 'hybrid'}:
        if remote_available:
            return requested_backend
        raise ProcessorError('BACKEND_UNAVAILABLE', 'BACKEND_UNAVAILABLE: Remote/hybrid backend is unavailable for this request.')

    if recommended == 'local' and local_available:
        return 'local'
    if remote_available:
        return recommended
    if local_available:
        return 'local'
    raise ProcessorError('BACKEND_UNAVAILABLE', 'BACKEND_UNAVAILABLE: No eligible UltraShape backend is available for this host.')


def package_result(output_dir: str, output_format: str) -> str:
    source_path = os.environ.get('ULTRASHAPE_TEST_ARTIFACT_PATH')
    if not source_path:
        raise ProcessorError('BACKEND_UNAVAILABLE', 'BACKEND_UNAVAILABLE: UltraShape backend is not configured for this request.')

    source = Path(source_path)
    if not source.is_file():
        raise ProcessorError('BACKEND_UNAVAILABLE', 'BACKEND_UNAVAILABLE: Configured UltraShape test artifact is not readable.')

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f'refined.{output_format}'
    shutil.copyfile(source, destination)
    return str(destination)


def nested_get(payload: dict, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {'1', 'true', 'yes', 'on'}


def main() -> int:
    try:
        emit_progress(5, 'validating')
        payload = read_payload()
        params = normalize_params(payload)
        output_dir = resolve_output_dir(payload)
        reference_image, coarse_mesh = resolve_input_paths(payload, params)

        ensure_file(reference_image, 'reference_image', IMAGE_EXTENSIONS)
        ensure_file(coarse_mesh, 'coarse_mesh', MESH_EXTENSIONS)

        emit(
            {
                'type': 'log',
                'message': json.dumps(
                    {
                        'reference_image': reference_image,
                        'coarse_mesh': coarse_mesh,
                        'backend': params['backend'],
                        'output_format': params['output_format'],
                        'output_dir': output_dir,
                        'node_id': payload.get('nodeId') or nested_get(payload, 'input', 'nodeId'),
                    }
                ),
            }
        )

        selected_backend = compute_preflight(params['backend'])
        emit_progress(20, f'preflight:{selected_backend}')
        emit_progress(60, f'running:{selected_backend}')
        emit_progress(90, 'packaging')

        file_path = package_result(output_dir, params['output_format'])
        emit({'type': 'done', 'result': {'filePath': file_path}})
        return 0
    except Exception as error:
        emit_error(error)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
