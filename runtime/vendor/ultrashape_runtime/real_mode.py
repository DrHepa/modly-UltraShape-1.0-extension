"""Authoritative real-mode adapter for an explicit upstream UltraShape checkout."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REAL_MODE_ADAPTER = 'ultrashape_runtime.real_mode.run_real_refine_pipeline'
REAL_MODE_ENTRYPOINT = 'scripts.infer_dit_refine.run_inference'
UPSTREAM_CHECKOUT_ENV = 'ULTRASHAPE_UPSTREAM_CHECKOUT'
UPSTREAM_CONFIG_ENV = 'ULTRASHAPE_UPSTREAM_CONFIG'

REQUIRED_CHECKOUT_MARKERS = (
    ('scripts/infer_dit_refine.py', 'file'),
    ('configs/infer_dit_refine.yaml', 'file'),
    ('ultrashape', 'dir'),
)
LICENSE_MARKERS = ('LICENSE', 'LICENSE.txt', 'NOTICE', 'Notice.txt')
REAL_REQUIRED_IMPORTS = ('torch', 'flash_attn', 'cubvh', 'omegaconf')


class RealModeUnavailableError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def _checkout_root(checkout_path: str | os.PathLike[str] | None = None) -> Path | None:
    candidate = str(checkout_path or os.environ.get(UPSTREAM_CHECKOUT_ENV, '')).strip()
    return Path(candidate).expanduser().resolve() if candidate else None


def _git_revision(checkout: Path) -> str:
    head = checkout / '.git' / 'HEAD'
    if not head.is_file():
        return 'unknown'
    raw_head = head.read_text(encoding='utf8').strip()
    if raw_head.startswith('ref:'):
        ref_path = checkout / '.git' / raw_head.split(':', 1)[1].strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding='utf8').strip() or 'unknown'
    return raw_head or 'unknown'


def _marker_blockers(checkout: Path) -> list[str]:
    blockers: list[str] = []
    if not checkout.is_dir():
        return [f'checkout-path:not-directory:{checkout}']

    for relative_path, marker_type in REQUIRED_CHECKOUT_MARKERS:
        marker = checkout / relative_path
        if marker_type == 'file' and not marker.is_file():
            blockers.append(f'checkout-marker:{relative_path}')
        if marker_type == 'dir' and not marker.is_dir():
            blockers.append(f'checkout-marker:{relative_path}')

    if not any((checkout / marker).is_file() for marker in LICENSE_MARKERS):
        blockers.append('checkout-marker:license')

    return blockers


def validate_upstream_checkout(checkout_path: str | os.PathLike[str] | None = None) -> dict[str, object]:
    checkout = _checkout_root(checkout_path)
    if checkout is None:
        return {
            'available': False,
            'adapter': REAL_MODE_ADAPTER,
            'source': None,
            'entrypoint': REAL_MODE_ENTRYPOINT,
            'blockers': [f'checkout-config:{UPSTREAM_CHECKOUT_ENV.lower()}'],
        }

    blockers = _marker_blockers(checkout)
    return {
        'available': not blockers,
        'adapter': REAL_MODE_ADAPTER,
        'source': 'checkout' if not blockers else None,
        'checkout_path': str(checkout),
        'revision': _git_revision(checkout) if not blockers else 'unknown',
        'entrypoint': REAL_MODE_ENTRYPOINT,
        'blockers': blockers,
    }


def describe_real_mode(checkout_path: str | os.PathLike[str] | None = None) -> dict[str, object]:
    description = validate_upstream_checkout(checkout_path)
    if description.get('available'):
        return _with_real_readiness(description, checkpoint=None, runtime_config_path=None, upstream_config_path=None)
    return {
        **description,
        'dependencies': _dependency_status([]),
        'reason': 'Authoritative real mode requires an explicit validated upstream UltraShape checkout.',
    }


def describe_real_readiness(
    *,
    checkout_path: str | os.PathLike[str] | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    runtime_config_path: str | os.PathLike[str] | None = None,
    upstream_config_path: str | os.PathLike[str] | None = None,
    python_exe: str | os.PathLike[str] | None = None,
    venv_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    description = validate_upstream_checkout(checkout_path)
    if description.get('available'):
        return _with_real_readiness(
            description,
            checkpoint=checkpoint,
            runtime_config_path=runtime_config_path or config_path,
            upstream_config_path=upstream_config_path,
            python_exe=python_exe,
            venv_dir=venv_dir,
        )
    return {
        **description,
        'runtime_config': {'available': False, 'path': str(runtime_config_path or config_path) if runtime_config_path or config_path else None},
        'upstream_config': {'available': False, 'path': None},
        'config': {'available': False, 'path': None},
        'checkpoint': {'available': False, 'path': str(checkpoint) if checkpoint else None},
        'dependencies': _dependency_status([]),
        'interpreter': _interpreter_status(python_exe=python_exe, venv_dir=venv_dir),
        'reason': 'Authoritative real mode requires an explicit validated upstream UltraShape checkout.',
    }


def _dependency_status(missing: list[str]) -> dict[str, object]:
    status = {
        name: {
            'available': name not in missing,
            'required': True,
            'degradable_for_portable': False,
        }
        for name in REAL_REQUIRED_IMPORTS
    }
    torch_status = status.get('torch')
    if isinstance(torch_status, dict) and torch_status.get('available') is True:
        torch_status['cuda_available'] = _torch_cuda_available()
    return status


def _torch_cuda_available() -> bool:
    try:
        torch = importlib.import_module('torch')
        cuda = getattr(torch, 'cuda', None)
        is_available = getattr(cuda, 'is_available', None)
        return bool(is_available()) if callable(is_available) else False
    except Exception:
        return False


def _missing_imports(module_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def _interpreter_status(
    *,
    python_exe: str | os.PathLike[str] | None = None,
    venv_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    selected_text = str(python_exe).strip() if python_exe else sys.executable
    selected = Path(selected_text).expanduser()
    recorded_venv = Path(str(venv_dir)).expanduser().resolve() if venv_dir else None
    matches_recorded_venv = True
    if recorded_venv is not None:
        try:
            selected.resolve().relative_to(recorded_venv)
        except ValueError:
            matches_recorded_venv = False
    return {
        'python_exe': selected_text,
        'runtime_python': sys.executable,
        'venv_dir': str(recorded_venv) if recorded_venv else None,
        'matches_recorded_venv': matches_recorded_venv,
    }


def _interpreter_blockers(
    *,
    python_exe: str | os.PathLike[str] | None = None,
    venv_dir: str | os.PathLike[str] | None = None,
) -> list[str]:
    status = _interpreter_status(python_exe=python_exe, venv_dir=venv_dir)
    if status.get('venv_dir') and status.get('matches_recorded_venv') is False:
        return [f"interpreter:venv-mismatch:{status['python_exe']}:not-under:{status['venv_dir']}"]
    return []


def _with_checkout_on_path(checkout: Path, callback):
    original_path = list(sys.path)
    stale_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == 'scripts'
        or name.startswith('scripts.')
        or name == 'ultrashape'
        or name.startswith('ultrashape.')
    }
    try:
        for name in stale_modules:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(checkout))
        return callback()
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == 'scripts' or name.startswith('scripts.') or name == 'ultrashape' or name.startswith('ultrashape.'):
                sys.modules.pop(name, None)
        sys.modules.update(stale_modules)


def _entrypoint_blockers(checkout: Path) -> list[str]:
    try:
        _with_checkout_on_path(checkout, lambda: _import_upstream_run_inference(checkout))
    except Exception as error:
        return [f'entrypoint:{REAL_MODE_ENTRYPOINT}:{error}']
    return []


def _with_real_readiness(
    checkout_description: dict[str, object],
    *,
    checkpoint: str | os.PathLike[str] | None,
    runtime_config_path: str | os.PathLike[str] | None,
    upstream_config_path: str | os.PathLike[str] | None,
    python_exe: str | os.PathLike[str] | None = None,
    venv_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    checkout = Path(str(checkout_description['checkout_path']))
    runtime_config = Path(runtime_config_path).expanduser().resolve() if runtime_config_path else None
    upstream_config = _resolve_upstream_config(checkout, upstream_config_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
    missing_deps = _missing_imports(REAL_REQUIRED_IMPORTS)
    blockers = list(checkout_description.get('blockers', []))
    blockers.extend(f'dependency:{name}' for name in missing_deps)
    if 'torch' not in missing_deps and not _torch_cuda_available():
        blockers.append('cuda:unavailable')
    if runtime_config is not None and not runtime_config.is_file():
        blockers.append(f'runtime_config:{runtime_config}')
    if not upstream_config.is_file():
        blockers.append(f'upstream_config:{upstream_config}')
    if checkpoint_path is None or not checkpoint_path.is_file():
        blockers.append(f'checkpoint:{checkpoint_path or "missing"}')
    blockers.extend(_interpreter_blockers(python_exe=python_exe, venv_dir=venv_dir))
    blockers.extend(_entrypoint_blockers(checkout))

    return {
        **checkout_description,
        'available': not blockers,
        'source': 'checkout' if not blockers else checkout_description.get('source'),
        'blockers': blockers,
        'dependencies': _dependency_status(missing_deps),
        'interpreter': _interpreter_status(python_exe=python_exe, venv_dir=venv_dir),
        'runtime_config': {'available': runtime_config is not None and runtime_config.is_file(), 'path': str(runtime_config) if runtime_config else None},
        'upstream_config': {'available': upstream_config.is_file(), 'path': str(upstream_config)},
        'config': {'available': upstream_config.is_file(), 'path': str(upstream_config)},
        'checkpoint': {'available': checkpoint_path is not None and checkpoint_path.is_file(), 'path': str(checkpoint_path) if checkpoint_path else None},
        'torch_cuda_profile': os.environ.get('CUDA_VERSION') or 'unknown',
    }


def _resolve_upstream_config(checkout: Path, upstream_config_path: str | os.PathLike[str] | None = None) -> Path:
    explicit = str(upstream_config_path or os.environ.get(UPSTREAM_CONFIG_ENV, '')).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return checkout / 'configs' / 'infer_dit_refine.yaml'


def _import_upstream_run_inference(checkout: Path):
    try:
        del checkout
        module = importlib.import_module('scripts.infer_dit_refine')
        run_inference = getattr(module, 'run_inference', None)
        if not callable(run_inference):
            raise RealModeUnavailableError(f'Upstream entrypoint is not callable: {REAL_MODE_ENTRYPOINT}.')
        return run_inference
    except Exception as error:
        if isinstance(error, RealModeUnavailableError):
            raise
        raise RealModeUnavailableError(f'Unable to import upstream real entrypoint {REAL_MODE_ENTRYPOINT}: {error}.') from error


def _build_upstream_args(
    *,
    checkout: Path,
    reference_image: str,
    coarse_mesh: str,
    output_dir: str,
    checkpoint: str | None,
    upstream_config_path: str | None,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    preserve_scale: bool,
) -> SimpleNamespace:
    upstream_config = str(_resolve_upstream_config(checkout, upstream_config_path))
    return SimpleNamespace(
        image=reference_image,
        reference_image=reference_image,
        mesh=coarse_mesh,
        coarse_mesh=coarse_mesh,
        ckpt=checkpoint,
        ckpt_path=checkpoint,
        checkpoint=checkpoint,
        config=upstream_config,
        config_path=upstream_config,
        output_dir=output_dir,
        num_inference_steps=steps,
        num_steps=steps,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        preserve_scale=preserve_scale,
        remove_bg=False,
        low_vram=False,
        scale=0.99,
        num_latents=32768,
        chunk_size=8000,
        octree_res=1024,
    )


def run_real_refine_pipeline(
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
    checkout_path: str | None = None,
    upstream_config_path: str | None = None,
    python_exe: str | None = None,
    venv_dir: str | None = None,
) -> dict[str, object]:
    del ext_dir
    if backend != 'local':
        raise RealModeUnavailableError('UltraShape real runner is local-only in this MVP.')
    if output_format != 'glb':
        raise RealModeUnavailableError('UltraShape real runner is glb-only in this MVP.')

    checkout_description = describe_real_readiness(
        checkout_path=checkout_path,
        checkpoint=checkpoint,
        runtime_config_path=config_path,
        upstream_config_path=upstream_config_path,
        python_exe=python_exe,
        venv_dir=venv_dir,
    )
    if not checkout_description.get('available'):
        blockers = ', '.join(str(blocker) for blocker in checkout_description.get('blockers', []))
        raise RealModeUnavailableError(f'Real runtime mode requested but unavailable: {blockers}.')

    checkout = Path(str(checkout_description['checkout_path']))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expected_name = f'{Path(reference_image).stem}_refined.glb'
    final_output = output_path / 'refined.glb'
    def execute_upstream() -> None:
        run_inference = _import_upstream_run_inference(checkout)
        with tempfile.TemporaryDirectory(prefix='ultrashape-upstream-real-') as upstream_output:
            args = _build_upstream_args(
                checkout=checkout,
                reference_image=reference_image,
                coarse_mesh=coarse_mesh,
                output_dir=upstream_output,
                checkpoint=checkpoint,
                upstream_config_path=upstream_config_path,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                preserve_scale=preserve_scale,
            )
            args.output_dir = upstream_output
            run_inference(args)

            upstream_refined = Path(upstream_output) / expected_name
            if not upstream_refined.is_file():
                raise RealModeUnavailableError(f'Upstream real mode did not produce expected refined output: {expected_name}.')
            shutil.copyfile(upstream_refined, final_output)

    _with_checkout_on_path(checkout, execute_upstream)

    return {
        'file_path': str(final_output),
        'format': 'glb',
        'backend': 'local',
        'warnings': [],
        'metrics': {
            'execution_trace': ['upstream-import', 'upstream-run_inference', 'normalize-output'],
            'pipeline': {
                'entrypoint': REAL_MODE_ENTRYPOINT,
                'class': 'upstream UltraShape-1.0',
                'returns_latents': False,
            },
            'runtime_mode': {
                'selection': 'real-available',
                'requested': os.environ.get('ULTRASHAPE_RUNTIME_MODE', 'auto').strip().lower() or 'auto',
                'active': 'real',
                'real': checkout_description,
                'portable': {
                    'available': True,
                    'authoritative': False,
                    'reason': 'Portable fallback was bypassed because real mode executed.',
                },
            },
            'upstream': {
                'source': 'checkout',
                'checkout_path': str(checkout),
                'revision': checkout_description.get('revision'),
                'entrypoint': REAL_MODE_ENTRYPOINT,
                'output_name': expected_name,
            },
        },
        'fallbacks': [],
        'subtrees_loaded': ['upstream-real'],
        'checkpoint': checkpoint,
        'execution': {
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
        },
    }
