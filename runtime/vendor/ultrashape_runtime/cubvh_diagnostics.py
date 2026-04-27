"""Shared cubvh diagnostics helpers for setup and runtime extraction."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any


TINY_INPUT_CELL_THRESHOLD = 4096


def minimal_self_test_fixture() -> dict[str, object]:
    """Return the smallest crossing-case sparse marching-cubes fixture."""

    return {
        'coords': [(0, 0, 0)],
        'corners': [(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)],
        'iso': 0.0,
        'input': {
            'coords_shape': (1, 3),
            'corners_shape': (1, 8),
            'cell_count': 1,
            'iso': 0.0,
        },
    }


def detect_callables(cubvh_module: ModuleType | None) -> dict[str, bool]:
    return {
        'sparse_marching_cubes': callable(getattr(cubvh_module, 'sparse_marching_cubes', None)) if cubvh_module else False,
        'sparse_marching_cubes_cpu': callable(getattr(cubvh_module, 'sparse_marching_cubes_cpu', None)) if cubvh_module else False,
    }


def get_callable(cubvh_module: ModuleType | None, name: str) -> Callable[..., object] | None:
    candidate = getattr(cubvh_module, name, None) if cubvh_module else None
    return candidate if callable(candidate) else None


def normalize_rows(values: object) -> list[object]:
    if hasattr(values, 'cpu') and callable(values.cpu):
        values = values.cpu()
    if hasattr(values, 'tolist') and callable(values.tolist):
        values = values.tolist()
    return list(values) if isinstance(values, (list, tuple)) else []


def normalize_output(raw_vertices: object, raw_faces: object) -> dict[str, object]:
    vertex_rows = normalize_rows(raw_vertices)
    face_rows = normalize_rows(raw_faces)
    vertices = [tuple(float(axis) for axis in vertex[:3]) for vertex in vertex_rows if isinstance(vertex, (list, tuple)) and len(vertex) >= 3]
    faces = [tuple(int(index) for index in face[:3]) for face in face_rows if isinstance(face, (list, tuple)) and len(face) >= 3]
    return {
        'normalized': True,
        'vertices': vertices,
        'faces': faces,
        'vertex_count': len(vertices),
        'face_count': len(faces),
    }


def tensor_shape(rows: list[tuple[Any, ...]], tensor: object) -> tuple[int, ...]:
    raw_shape = getattr(tensor, 'shape', None)
    if isinstance(raw_shape, (list, tuple)) and raw_shape:
        return tuple(int(axis) for axis in raw_shape)
    column_count = len(rows[0]) if rows else 0
    return (len(rows), column_count)


def input_summary(
    *,
    coords: list[tuple[int, int, int]],
    corners: list[tuple[float, float, float, float, float, float, float, float]],
    coords_tensor: object,
    corners_tensor: object,
    iso: float,
) -> dict[str, object]:
    flat_coords = [float(value) for row in coords for value in row]
    flat_corners = [float(value) for row in corners for value in row]
    return {
        'coords_shape': tensor_shape(coords, coords_tensor),
        'coords_dtype': str(getattr(coords_tensor, 'dtype', None)),
        'coords_min': min(flat_coords) if flat_coords else 0.0,
        'coords_max': max(flat_coords) if flat_coords else 0.0,
        'corners_shape': tensor_shape(corners, corners_tensor),
        'corners_dtype': str(getattr(corners_tensor, 'dtype', None)),
        'corners_min': round(min(flat_corners), 6) if flat_corners else 0.0,
        'corners_max': round(max(flat_corners), 6) if flat_corners else 0.0,
        'cell_count': len(coords),
        'iso': round(float(iso), 6),
    }


def format_input_summary(summary: dict[str, object]) -> str:
    return ', '.join(f'{key}={summary[key]}' for key in [
        'coords_shape',
        'coords_dtype',
        'coords_min',
        'coords_max',
        'corners_shape',
        'corners_dtype',
        'corners_min',
        'corners_max',
        'cell_count',
        'iso',
    ])


def capture_metadata(cubvh_module: ModuleType | None = None, torch_module: ModuleType | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        'cubvh_module_path': getattr(cubvh_module, '__file__', None) if cubvh_module else None,
        'cubvh_callables': detect_callables(cubvh_module),
        'selected_cuda_home': os.environ.get('CUDA_HOME'),
        'env_torch_cuda_arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST'),
        'cuda_available': False,
    }
    if torch_module is None:
        metadata['torch_unavailable_reason'] = 'torch module unavailable'
        return metadata

    metadata['torch_version'] = getattr(torch_module, '__version__', None)
    torch_version = getattr(torch_module, 'version', None)
    metadata['torch_cuda_version'] = getattr(torch_version, 'cuda', None)
    metadata['cuda_runtime_hint'] = metadata['torch_cuda_version']
    metadata['default_nvcc_cuda_version'] = os.environ.get('CUDA_VERSION') or os.environ.get('NVCC_CUDA_VERSION')
    if not metadata['default_nvcc_cuda_version']:
        metadata['default_nvcc_cuda_version_unavailable_reason'] = 'CUDA_VERSION/NVCC_CUDA_VERSION environment variables not set; nvcc probing intentionally not executed'
    cuda = getattr(torch_module, 'cuda', None)
    try:
        metadata['cuda_available'] = bool(cuda is not None and callable(cuda.is_available) and cuda.is_available())
    except Exception as error:  # pragma: no cover - hardware/API dependent
        metadata['cuda_available'] = False
        metadata['cuda_unavailable_reason'] = str(error)
    if not metadata['cuda_available'] and 'cuda_unavailable_reason' not in metadata:
        metadata['cuda_unavailable_reason'] = 'torch.cuda.is_available() returned false'
    arch_list = _safe_call(cuda, 'get_arch_list') if cuda is not None else None
    if arch_list is not None:
        metadata['torch_arch_list'] = list(arch_list) if isinstance(arch_list, (list, tuple)) else arch_list
    driver_hint = _safe_call(cuda, 'driver_version') if cuda is not None else None
    if driver_hint is None:
        torch_c = getattr(torch_module, '_C', None)
        driver_hint = _safe_call(torch_c, '_cuda_getDriverVersion') if torch_c is not None else None
    if driver_hint is None:
        metadata['cuda_driver_hint_unavailable_reason'] = 'driver version not exposed by torch.cuda or torch._C without external shelling'
    else:
        metadata['cuda_driver_hint'] = driver_hint
    if metadata['cuda_available']:
        try:
            index = cuda.current_device() if hasattr(cuda, 'current_device') and callable(cuda.current_device) else 0
            metadata['device_name'] = cuda.get_device_name(index) if hasattr(cuda, 'get_device_name') else None
            metadata['device_capability'] = cuda.get_device_capability(index) if hasattr(cuda, 'get_device_capability') else None
        except Exception as error:  # pragma: no cover - hardware/API dependent
            metadata['cuda_metadata_error'] = str(error)
    return metadata


def _safe_call(target: object, method_name: str) -> object | None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:  # pragma: no cover - hardware/API dependent
        return None


def classify_error(
    error: BaseException | str,
    *,
    path: str,
    input_info: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> str:
    text = str(error).lower()
    cell_count = int((input_info or {}).get('cell_count', 0) or 0)
    if path == 'import':
        return 'import_failure'
    if path == 'cpu' and ('missing callable' in text or 'not callable' in text or 'unavailable' in text):
        return 'cpu_fallback_unavailable'
    if 'missing callable' in text or 'not callable' in text:
        return 'missing_callable'
    if path == 'cpu':
        return 'cpu_fallback_failure'
    if 'out of memory' in text or 'cuda oom' in text:
        if cell_count and cell_count <= TINY_INPUT_CELL_THRESHOLD:
            return 'cuda_oom_tiny_input_suspected_kernel_or_arch_mismatch'
        return 'cuda_oom'
    if _looks_like_arch_toolkit_mismatch(text, metadata):
        return 'suspected_arch_toolkit_mismatch'
    if 'cuda' in text or 'kernel' in text or 'launch' in text:
        return 'cuda_kernel_failure'
    return f'{path}_failure'


def _looks_like_arch_toolkit_mismatch(text: str, metadata: dict[str, object] | None) -> bool:
    mismatch_tokens = (
        'no kernel image is available',
        'invalid device function',
        'unsupported gpu architecture',
        'named symbol not found',
    )
    if any(token in text for token in mismatch_tokens):
        return True
    capability = (metadata or {}).get('device_capability')
    capability_text = '.'.join(str(item) for item in capability) if isinstance(capability, (list, tuple)) else str(capability or '')
    arch_list = str((metadata or {}).get('env_torch_cuda_arch_list') or '')
    if capability_text in {'12.1', 'sm_121'} and arch_list and '12.1' not in arch_list and '12+PTX' not in arch_list:
        return 'cuda' in text or 'kernel' in text or 'launch' in text
    return False


def rebuild_guidance(error_class: str, metadata: dict[str, object] | None = None) -> dict[str, object]:
    recommended = error_class in {
        'cuda_oom_tiny_input_suspected_kernel_or_arch_mismatch',
        'cuda_kernel_failure',
        'suspected_arch_toolkit_mismatch',
        'missing_callable',
        'cpu_fallback_unavailable',
    }
    fields = [
        'device_capability',
        'torch_cuda_version',
        'env_torch_cuda_arch_list',
        'selected_cuda_home',
        'cubvh_module_path',
    ]
    reason = 'force_source_compile_or_match_torch_cuda_arch' if recommended else 'not_applicable'
    return {
        'recommended': recommended,
        'reason': reason,
        'fields': {field: (metadata or {}).get(field) for field in fields},
        'commands_hint': 'Reinstall cubvh from source with a torch/CUDA toolkit and TORCH_CUDA_ARCH_LIST compatible with the target GPU; this is diagnostic guidance, not a guaranteed repair.',
    }


def self_test_result(
    cubvh_module: ModuleType | None,
    *,
    path: str,
    torch_module: ModuleType | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    fixture = minimal_self_test_fixture()
    input_info = fixture['input'] if isinstance(fixture['input'], dict) else {}
    callable_name = 'sparse_marching_cubes_cpu' if path == 'cpu' else 'sparse_marching_cubes'
    callable_fn = get_callable(cubvh_module, callable_name)
    if callable_fn is None:
        error = f'missing callable {callable_name}'
        return {
            'available': False,
            'executable': False,
            'skipped': path == 'cpu',
            'error_class': classify_error(error, path=path, input_info=input_info, metadata=metadata),
            'error_message': error,
            'elapsed_ms': 0,
            'input': input_info,
            'launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING') if path == 'cuda' else None,
        }

    started = time.monotonic()
    try:
        coords = _tensor_or_rows(torch_module, fixture['coords'], dtype_name='int32')
        corners = _tensor_or_rows(torch_module, fixture['corners'], dtype_name='float32')
        raw_vertices, raw_faces = callable_fn(coords, corners, fixture['iso'], ensure_consistency=False)
        output = normalize_output(raw_vertices, raw_faces)
        executable = bool(output['vertex_count']) and bool(output['face_count'])
        return {
            'available': True,
            'executable': executable,
            'skipped': False,
            'error_class': None if executable else f'{path}_empty_output',
            'error_message': None if executable else 'cubvh self-test returned empty or unparseable output',
            'elapsed_ms': int((time.monotonic() - started) * 1000),
            'input': input_info,
            'output': output,
            'launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING') if path == 'cuda' else None,
        }
    except Exception as error:  # pragma: no cover - exercised by subprocess contract tests
        return {
            'available': True,
            'executable': False,
            'skipped': False,
            'error_class': classify_error(error, path=path, input_info=input_info, metadata=metadata),
            'error_message': str(error),
            'elapsed_ms': int((time.monotonic() - started) * 1000),
            'input': input_info,
            'launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING') if path == 'cuda' else None,
        }


def _tensor_or_rows(torch_module: ModuleType | None, rows: object, *, dtype_name: str) -> object:
    tensor_fn = getattr(torch_module, 'tensor', None) if torch_module else None
    if callable(tensor_fn):
        dtype = getattr(torch_module, dtype_name, None)
        return tensor_fn(rows, dtype=dtype)
    return rows


def readiness_summary() -> str:
    return 'runtime-live-no-setup-self-test'


def failure_message(
    *,
    error_class: str,
    input_info: dict[str, object],
    readiness: str,
    guidance: dict[str, object],
    cuda_error: BaseException | None = None,
    cpu_error: BaseException | None = None,
) -> str:
    parts = [
        f'cubvh_error_class={error_class}',
        format_input_summary(input_info),
        f'readiness_summary={readiness}',
        f'rebuild_guidance={guidance.get("reason")}',
    ]
    if cuda_error is not None:
        parts.append(f'cuda_error={cuda_error}')
    if cpu_error is not None:
        parts.append(f'cpu_error={cpu_error}')
    return 'cubvh extraction failed: ' + '; '.join(parts)
