#!/usr/bin/env python3
"""Truthful clean-room setup for the local UltraShape shell."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from shutil import copy2, copytree
import sys

REQUIRED_IMPORTS = [
    'torch',
    'torchvision',
    'numpy',
    'trimesh',
    'PIL',
    'cv2',
    'skimage',
    'yaml',
    'omegaconf',
    'einops',
    'transformers',
    'huggingface_hub',
    'accelerate',
    'diffusers',
    'cubvh',
    'safetensors',
    'tqdm',
]
CONDITIONAL_IMPORTS = ['rembg', 'onnxruntime']
DEGRADABLE_IMPORTS = ['flash_attn']
CHECKPOINT_RELATIVE = Path('models/ultrashape/ultrashape_v1.pt')
CONFIG_RELATIVE = Path('runtime/configs/infer_dit_refine.yaml')
VENDOR_RELATIVE = Path('runtime/vendor/ultrashape_runtime')
READINESS_FILE = '.runtime-readiness.json'
SUMMARY_FILE = '.setup-summary.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage the truthful local UltraShape shell')
    parser.add_argument('payload_json', nargs='?')
    parser.add_argument('--ext-dir', default=str(repo_root()))
    parser.add_argument('--python-exe', default=sys.executable)
    args = parser.parse_args()

    if args.payload_json:
        payload = json.loads(args.payload_json)
        if not isinstance(payload, dict):
            raise SystemExit('setup payload must be a JSON object')
        ext_dir = payload.get('ext_dir')
        python_exe = payload.get('python_exe')
        if isinstance(ext_dir, str) and ext_dir.strip() != '':
            args.ext_dir = ext_dir
        if isinstance(python_exe, str) and python_exe.strip() != '':
            args.python_exe = python_exe

    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def stage_runtime_assets(ext_dir: Path) -> tuple[bool, bool]:
    config_source = repo_root() / CONFIG_RELATIVE
    vendor_source = repo_root() / VENDOR_RELATIVE
    config_target = ext_dir / CONFIG_RELATIVE
    vendor_target = ext_dir / VENDOR_RELATIVE

    config_target.parent.mkdir(parents=True, exist_ok=True)
    vendor_target.parent.mkdir(parents=True, exist_ok=True)
    (ext_dir / CHECKPOINT_RELATIVE.parent).mkdir(parents=True, exist_ok=True)

    if config_source.is_file() and _same_path(config_source, config_target) is False:
        copy2(config_source, config_target)
    if vendor_source.is_dir() and _same_path(vendor_source, vendor_target) is False:
        copytree(vendor_source, vendor_target, dirs_exist_ok=True)

    return config_target.is_file(), vendor_target.is_dir()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return False


def detect_missing_imports(module_names: list[str]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def build_readiness(ext_dir: Path, *, config_ready: bool, vendor_ready: bool) -> dict[str, object]:
    checkpoint_path = ext_dir / CHECKPOINT_RELATIVE
    vendor_path = ext_dir / VENDOR_RELATIVE.parent
    missing_required: list[str] = []
    if not config_ready:
        missing_required.append(f'config:{CONFIG_RELATIVE.as_posix()}')
    if not vendor_ready:
        missing_required.append(f'vendor:{VENDOR_RELATIVE.as_posix()}')

    missing_required_imports = detect_missing_imports(REQUIRED_IMPORTS)
    missing_conditional = detect_missing_imports(CONDITIONAL_IMPORTS)
    missing_degradable = detect_missing_imports(DEGRADABLE_IMPORTS)

    missing_required.extend(f'import:{name}' for name in missing_required_imports)
    if not checkpoint_path.is_file():
        missing_required.append(f'weight:{CHECKPOINT_RELATIVE.as_posix()}')

    required_imports_ok = not missing_required_imports and config_ready and vendor_ready
    weights_ready = checkpoint_path.is_file()
    config_and_vendor_ready = config_ready and vendor_ready
    runtime_ready = required_imports_ok and weights_ready

    status = 'blocked'
    if runtime_ready and (missing_conditional or missing_degradable):
        status = 'degraded'
    elif runtime_ready:
        status = 'ready'

    return {
        'backend': 'local',
        'checkpoint': str(checkpoint_path),
        'config_path': str(ext_dir / CONFIG_RELATIVE),
        'config_ready': config_ready,
        'ext_dir': str(ext_dir),
        'missing_required': missing_required,
        'missing_conditional': [f'import:{name}' for name in missing_conditional],
        'missing_degradable': [f'import:{name}' for name in missing_degradable],
        'missing_optional': [f'import:{name}' for name in [*missing_conditional, *missing_degradable]],
        'required_imports_ok': required_imports_ok,
        'runtime_ready': runtime_ready,
        'status': status,
        'vendor_path': str(vendor_path),
        'vendor_ready': vendor_ready,
        'weights_ready': weights_ready,
        'required_checkpoint_subtrees': ['vae', 'dit', 'conditioner'],
        'install_success': not missing_required,
        'install_ready': not missing_required,
        'runtime_closure_ready': config_and_vendor_ready,
        'runtime_closure_reason': (
            'The clean-room vendored closure is staged for local refinement.' if config_and_vendor_ready else None
        ),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8')


def main() -> int:
    args = parse_args()
    ext_dir = Path(args.ext_dir).resolve()
    ext_dir.mkdir(parents=True, exist_ok=True)

    config_ready, vendor_ready = stage_runtime_assets(ext_dir)
    readiness = build_readiness(ext_dir, config_ready=config_ready, vendor_ready=vendor_ready)
    summary = {
        'python_exe': args.python_exe,
        'ext_dir': str(ext_dir),
        'config_ready': readiness['config_ready'],
        'vendor_ready': readiness['vendor_ready'],
        'required_imports_ok': readiness['required_imports_ok'],
        'weights_ready': readiness['weights_ready'],
        'runtime_ready': readiness['runtime_ready'],
        'runtime_closure_ready': readiness['runtime_closure_ready'],
        'runtime_closure_reason': readiness['runtime_closure_reason'],
        'install_success': readiness['install_success'],
        'install_ready': readiness['install_ready'],
        'status': readiness['status'],
        'vendor_path': readiness['vendor_path'],
        'missing_required': readiness['missing_required'],
        'missing_optional': readiness['missing_optional'],
        'missing_conditional': readiness['missing_conditional'],
        'missing_degradable': readiness['missing_degradable'],
        'checkpoint': readiness['checkpoint'],
        'config_path': readiness['config_path'],
    }

    write_json(ext_dir / READINESS_FILE, readiness)
    write_json(ext_dir / SUMMARY_FILE, summary)
    json.dump(summary, sys.stdout)
    sys.stdout.write('\n')
    return 0 if readiness['install_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
