#!/usr/bin/env python3
"""Installer for the UltraShape model shell."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import copy2, copytree, which
from typing import Any
from venv import EnvBuilder

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

CORE_PIP_DEPENDENCIES = [
    'torch==2.7.0+cu128',
    'torchvision==0.22.0',
    'numpy',
    'trimesh',
    'Pillow',
    'opencv-python-headless',
    'scikit-image',
    'PyYAML',
    'omegaconf',
    'einops',
    'transformers',
    'huggingface_hub',
    'accelerate',
    'diffusers',
    'safetensors',
    'tqdm',
    'rembg',
    'onnxruntime',
]
TORCH_EXTRA_INDEX_URL = 'https://download.pytorch.org/whl/cu128'
CUBVH_TORCH_CUDA_TOOLKITS = {
    'cu128': Path('/usr/local/cuda-12.8'),
}
CUBVH_PINNED_REF = '7855c000f95e43742081060d869702b2b2b33d1f'
CUBVH_SOURCE = f'git+https://github.com/ashawkey/cubvh@{CUBVH_PINNED_REF}'
FLASH_ATTN_PACKAGE = 'flash-attn'
DEFAULT_WEIGHT_REPO_ID = 'infinith/UltraShape'
WEIGHT_FILENAME = 'ultrashape_v1.pt'

CHECKPOINT_RELATIVE = Path('models/ultrashape/ultrashape_v1.pt')
CONFIG_RELATIVE = Path('runtime/configs/infer_dit_refine.yaml')
VENDOR_RELATIVE = Path('runtime/vendor/ultrashape_runtime')
READINESS_FILE = '.runtime-readiness.json'
SUMMARY_FILE = '.setup-summary.json'
REAL_MODE_ADAPTER = 'ultrashape_runtime.real_mode.run_real_refine_pipeline'
REAL_MODE_REASON = (
    'Authoritative real mode is optional and remains unavailable until the exact upstream '
    'torch module graph adapter is vendored and the required runtime dependencies are present.'
)
PORTABLE_MODE_REASON = 'Portable fallback is active for reduced environments; it is not the authoritative upstream closure.'

PACKAGE_STUBS = {
    'torch': (
        'import json\n\n'
        '__version__ = "0.0-test"\n\n'
        'def load(path, map_location=None):\n'
        '    del map_location\n'
        '    with open(path, "r", encoding="utf8") as handle:\n'
        '        return json.load(handle)\n'
    ),
    'torchvision': '__version__ = "0.0-test"\n',
    'numpy': '__version__ = "0.0-test"\n',
    'trimesh': '__version__ = "0.0-test"\n',
    'PIL': '__version__ = "0.0-test"\n',
    'cv2': '__version__ = "0.0-test"\n',
    'skimage': '__version__ = "0.0-test"\n',
    'yaml': '__version__ = "0.0-test"\n',
    'omegaconf': 'class OmegaConf:\n    pass\n',
    'einops': '__version__ = "0.0-test"\n',
    'transformers': '__version__ = "0.0-test"\n',
    'huggingface_hub': (
        'import json\n'
        'import os\n'
        'from pathlib import Path\n\n'
        'class GatedRepoError(Exception):\n'
        '    status_code = 401\n\n'
        'class HfHubHTTPError(Exception):\n'
        '    def __init__(self, message, status_code=None):\n'
        '        super().__init__(message)\n'
        '        self.status_code = status_code\n\n'
        'class RepositoryNotFoundError(Exception):\n'
        '    status_code = 404\n\n'
        'class RevisionNotFoundError(Exception):\n'
        '    status_code = 404\n\n'
        'class RemoteEntryNotFoundError(Exception):\n'
        '    status_code = 404\n\n'
        'class EntryNotFoundError(Exception):\n'
        '    status_code = 404\n\n'
        'class LocalEntryNotFoundError(Exception):\n'
        '    pass\n\n'
        'def hf_hub_download(repo_id, filename, revision=None, token=None, local_dir=None, local_dir_use_symlinks=False, **kwargs):\n'
        '    del repo_id, revision, token, local_dir_use_symlinks, kwargs\n'
        '    scenario = os.environ.get("ULTRASHAPE_SETUP_TEST_HF_SCENARIO")\n'
        '    if scenario == "auth":\n'
        '        raise GatedRepoError("401 Unauthorized from Hugging Face")\n'
        '    if scenario == "network":\n'
        '        raise LocalEntryNotFoundError("Connection error while downloading UltraShape weights")\n'
        '    if scenario == "not-found":\n'
        '        raise RemoteEntryNotFoundError("404 missing ultrashape_v1.pt")\n'
        '    if scenario == "other":\n'
        '        raise RuntimeError("Unexpected HF failure during UltraShape acquisition")\n'
        '    target_dir = Path(local_dir) if local_dir else Path.cwd()\n'
        '    target_dir.mkdir(parents=True, exist_ok=True)\n'
        '    target_path = target_dir / filename\n'
        '    default_checkpoint = json.dumps({\n'
        '        "vae": {"tensors": {"weights": [0.1, 0.2, 0.3, 0.4]}},\n'
        '        "dit": {"tensors": {"weights": [0.5, 0.6, 0.7, 0.8]}},\n'
        '        "conditioner": {"tensors": {"weights": [0.2, 0.4, 0.6, 0.8]}},\n'
        '    })\n'
        '    payload = os.environ.get("ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE", default_checkpoint)\n'
        '    if payload == "stub-weight":\n'
        '        payload = default_checkpoint\n'
        '    target_path.write_text(payload, encoding="utf8")\n'
        '    return str(target_path)\n'
    ),
    'accelerate': '__version__ = "0.0-test"\n',
    'diffusers': '__version__ = "0.0-test"\n',
    'cubvh': '__version__ = "0.0-test"\n',
    'safetensors': '__version__ = "0.0-test"\n',
    'tqdm': '__version__ = "0.0-test"\n',
    'rembg': 'def remove(payload):\n    return payload\n',
    'onnxruntime': '__version__ = "0.0-test"\n',
    'flash_attn': '__version__ = "0.0-test"\n',
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Install the UltraShape model shell runtime')
    parser.add_argument('payload_json', nargs='?')
    parser.add_argument('--ext-dir', default=str(repo_root()))
    parser.add_argument('--python-exe', default=sys.executable)
    parser.add_argument('--gpu-sm', dest='gpu_sm')
    parser.add_argument('--cuda-version', dest='cuda_version')
    args = parser.parse_args()

    payload: dict[str, Any] = {}
    if args.payload_json:
        payload = json.loads(args.payload_json)
        if not isinstance(payload, dict):
            raise SystemExit('setup payload must be a JSON object')

    ext_dir = payload.get('ext_dir')
    python_exe = payload.get('python_exe')
    gpu_sm = payload.get('gpu_sm')
    cuda_version = payload.get('cuda_version')

    if isinstance(ext_dir, str) and ext_dir.strip():
        args.ext_dir = ext_dir.strip()
    if isinstance(python_exe, str) and python_exe.strip():
        args.python_exe = python_exe.strip()
    args.gpu_sm = normalize_optional_scalar(gpu_sm, args.gpu_sm)
    args.cuda_version = normalize_optional_scalar(cuda_version, args.cuda_version)
    args.payload = payload
    return args


def normalize_optional_scalar(payload_value: object, fallback: str | None) -> str | None:
    if isinstance(payload_value, bool):
        return fallback
    if isinstance(payload_value, (int, float)):
        return str(int(payload_value)) if isinstance(payload_value, float) and payload_value.is_integer() else str(payload_value)
    if isinstance(payload_value, str) and payload_value.strip():
        return payload_value.strip()
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return None


def stage_runtime_assets(ext_dir: Path) -> tuple[bool, bool]:
    config_source = repo_root() / CONFIG_RELATIVE
    vendor_source = repo_root() / VENDOR_RELATIVE
    config_target = ext_dir / CONFIG_RELATIVE
    vendor_target = ext_dir / VENDOR_RELATIVE

    config_target.parent.mkdir(parents=True, exist_ok=True)
    vendor_target.parent.mkdir(parents=True, exist_ok=True)
    (ext_dir / CHECKPOINT_RELATIVE.parent).mkdir(parents=True, exist_ok=True)

    if config_source.is_file() and not same_path(config_source, config_target):
        copy2(config_source, config_target)
    if vendor_source.is_dir() and not same_path(vendor_source, vendor_target):
        copytree(vendor_source, vendor_target, dirs_exist_ok=True)

    return config_target.is_file(), vendor_target.is_dir()


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return False


def detect_host_facts() -> dict[str, Any]:
    platform_name = os.environ.get('ULTRASHAPE_SETUP_TEST_HOST_PLATFORM') or sys.platform
    machine = os.environ.get('ULTRASHAPE_SETUP_TEST_HOST_MACHINE') or os.uname().machine.lower()
    cuda_version = os.environ.get('ULTRASHAPE_SETUP_TEST_HOST_CUDA_VERSION') or detect_cuda_version()
    gpu_sm = os.environ.get('ULTRASHAPE_SETUP_TEST_HOST_GPU_SM') or detect_gpu_sm()
    return {
        'platform': platform_name,
        'machine': machine,
        'cuda_version': cuda_version,
        'gpu_sm': gpu_sm,
    }


def detect_cuda_version() -> str | None:
    env_cuda = os.environ.get('CUDA_VERSION')
    if env_cuda:
        return env_cuda.strip() or None
    nvcc = which('nvcc')
    if not nvcc:
        return None
    try:
        result = subprocess.run([nvcc, '--version'], capture_output=True, text=True, check=False)
    except OSError:
        return None
    match = re.search(r'release\s+(\d+\.\d+)', result.stdout + result.stderr)
    return match.group(1) if match else None


def detect_gpu_sm() -> str | None:
    nvidia_smi = which('nvidia-smi')
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), '')
    return first_line or None


def ensure_venv(ext_dir: Path) -> Path:
    venv_dir = ext_dir / 'venv'
    EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
    return venv_dir


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / 'bin' / 'python'


def venv_site_packages(venv_dir: Path) -> Path:
    result = subprocess.run(
        [str(venv_python(venv_dir)), '-c', 'import site; print(site.getsitepackages()[0])'],
        capture_output=True,
        text=True,
        env=child_env(),
        check=True,
    )
    return Path(result.stdout.strip())


def run_command(command: list[str]) -> str:
    subprocess.run(command, check=True, capture_output=True, text=True, env=child_env())
    return ' '.join(command)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    if extra:
        env.update(extra)
    return env


def detect_torch_cuda_profile() -> str | None:
    for dependency in CORE_PIP_DEPENDENCIES:
        if not dependency.startswith('torch=='):
            continue
        match = re.search(r'\+(cu\d+)$', dependency)
        if match:
            return match.group(1)

    match = re.search(r'/(cu\d+)$', TORCH_EXTRA_INDEX_URL)
    return match.group(1) if match else None


def prepend_env_path(current: str | None, value: str) -> str:
    entries = [value]
    if current:
        entries.extend(item for item in current.split(os.pathsep) if item and item != value)
    return os.pathsep.join(entries)


def resolve_cubvh_cuda_toolkit() -> dict[str, Any]:
    torch_cuda_profile = detect_torch_cuda_profile()
    expected_cuda_home = CUBVH_TORCH_CUDA_TOOLKITS.get(torch_cuda_profile)
    stub_mode = os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1'

    selected_cuda_home: Path | None = None
    if expected_cuda_home and (stub_mode or expected_cuda_home.is_dir()):
        selected_cuda_home = expected_cuda_home
    else:
        env_cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if env_cuda_home:
            selected_cuda_home = Path(env_cuda_home)
        else:
            nvcc = which('nvcc')
            if nvcc:
                selected_cuda_home = Path(nvcc).resolve().parent.parent

    env_overrides: dict[str, str] = {}
    if selected_cuda_home:
        selected_str = str(selected_cuda_home)
        env_overrides = {
            'CUDA_HOME': selected_str,
            'CUDA_PATH': selected_str,
            'PATH': prepend_env_path(os.environ.get('PATH'), str(selected_cuda_home / 'bin')),
            'LD_LIBRARY_PATH': prepend_env_path(os.environ.get('LD_LIBRARY_PATH'), str(selected_cuda_home / 'lib64')),
            'LIBRARY_PATH': prepend_env_path(os.environ.get('LIBRARY_PATH'), str(selected_cuda_home / 'lib64')),
        }

    return {
        'torch_cuda_profile': torch_cuda_profile,
        'expected_cuda_home': str(expected_cuda_home) if expected_cuda_home else None,
        'selected_cuda_home': str(selected_cuda_home) if selected_cuda_home else None,
        'toolkit_pinned': bool(expected_cuda_home and selected_cuda_home and expected_cuda_home == selected_cuda_home),
        'env_overrides': env_overrides,
    }


def bootstrap_packaging_tools(venv_dir: Path) -> dict[str, Any]:
    python_path = str(venv_python(venv_dir))
    commands = [run_command([python_path, '-m', 'ensurepip', '--upgrade'])]
    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        return {'mode': 'stub', 'commands': commands, 'packages': ['pip', 'setuptools', 'wheel']}
    commands.append(run_command([python_path, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel']))
    return {'mode': 'pip', 'commands': commands, 'packages': ['pip', 'setuptools', 'wheel']}


def create_stub_package(site_packages: Path, module_name: str) -> None:
    package_dir = site_packages / module_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / '__init__.py').write_text(PACKAGE_STUBS[module_name], encoding='utf8')


def install_core_dependencies(venv_dir: Path) -> dict[str, Any]:
    site_packages = venv_site_packages(venv_dir)
    site_packages.mkdir(parents=True, exist_ok=True)

    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        installed = []
        for module_name in [*REQUIRED_IMPORTS, *CONDITIONAL_IMPORTS, *DEGRADABLE_IMPORTS]:
            if module_name in {'cubvh', 'flash_attn'}:
                continue
            create_stub_package(site_packages, module_name)
            installed.append(module_name)
        return {'mode': 'stub', 'commands': ['stub-core-dependencies'], 'installed_modules': installed}

    command = [
        str(venv_python(venv_dir)),
        '-m',
        'pip',
        'install',
        '--extra-index-url',
        TORCH_EXTRA_INDEX_URL,
        *CORE_PIP_DEPENDENCIES,
    ]
    rendered = run_command(command)
    return {
        'mode': 'pip',
        'commands': [rendered],
        'installed_modules': [name for name in REQUIRED_IMPORTS if name != 'cubvh'] + CONDITIONAL_IMPORTS,
    }


def detect_cubvh_prerequisites(host_facts: dict[str, Any]) -> dict[str, Any]:
    missing_tokens = {
        token.strip()
        for token in os.environ.get('ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING', '').split(',')
        if token.strip()
    }
    machine = str(host_facts.get('machine') or '')
    platform_name = str(host_facts.get('platform') or '')
    is_linux_arm64 = platform_name.startswith('linux') and machine in {'arm64', 'aarch64'}
    cuda_toolkit = resolve_cubvh_cuda_toolkit()
    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        detected = {
            'host': f'{platform_name}-{machine}',
            'git': 'git' not in missing_tokens,
            'compiler': None if 'compiler' in missing_tokens else 'g++',
            'cuda': None if 'cuda' in missing_tokens else cuda_toolkit['selected_cuda_home'],
            'eigen': None if 'eigen' in missing_tokens else '/usr/include/eigen3',
        }
    else:
        compiler = next((candidate for candidate in ('c++', 'g++', 'clang++') if which(candidate)), None)
        eigen_candidates = [Path('/usr/include/eigen3'), Path('/usr/local/include/eigen3')]
        eigen_path = next((str(candidate) for candidate in eigen_candidates if (candidate / 'Eigen' / 'Core').is_file()), None)
        detected = {
            'host': f'{platform_name}-{machine}',
            'git': which('git') is not None,
            'compiler': compiler,
            'cuda': cuda_toolkit['selected_cuda_home'],
            'eigen': eigen_path,
        }

    missing = []
    if not is_linux_arm64:
        missing.append('linux-arm64')
    if not detected['git']:
        missing.append('git')
    if not detected['compiler']:
        missing.append('compiler-toolchain')
    if not detected['cuda']:
        missing.append('cuda-build-tooling')
    if not detected['eigen']:
        missing.append('eigen-headers')

    return {
        'ok': not missing,
        'detected_prerequisites': detected,
        'missing_prerequisites': missing,
        'failure_message': (
            None
            if not missing
            else 'cubvh source build requires Linux ARM64 with git, compiler toolchain, CUDA build tooling, and Eigen headers available.'
        ),
    }


def is_linux_arm64_host(host_facts: dict[str, Any]) -> bool:
    platform_name = str(host_facts.get('platform') or '')
    machine = str(host_facts.get('machine') or '')
    return platform_name.startswith('linux') and machine in {'arm64', 'aarch64'}


def install_cubvh_stage(venv_dir: Path, ext_dir: Path) -> tuple[dict[str, Any], list[str]]:
    command = [str(venv_python(venv_dir)), '-m', 'pip', 'install', '--no-build-isolation', CUBVH_SOURCE]
    rendered_command = ' '.join(command)
    cuda_toolkit = resolve_cubvh_cuda_toolkit()
    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        create_stub_package(venv_site_packages(venv_dir), 'cubvh')
    else:
        subprocess.run(command, check=True, env=child_env(cuda_toolkit['env_overrides']))

    missing = run_import_smoke(venv_dir, ext_dir, ['cubvh'])
    return (
        {
            'attempted': True,
            'required': True,
            'status': 'ready' if not missing else 'blocked',
            'commands': [rendered_command],
            'source': CUBVH_SOURCE,
            'pinned_ref': CUBVH_PINNED_REF,
            'torch_cuda_profile': cuda_toolkit['torch_cuda_profile'],
            'expected_cuda_home': cuda_toolkit['expected_cuda_home'],
            'selected_cuda_home': cuda_toolkit['selected_cuda_home'],
            'toolkit_pinned': cuda_toolkit['toolkit_pinned'],
            'env_overrides': cuda_toolkit['env_overrides'],
            'import_smoke_missing': missing,
            'failure_message': None if not missing else 'cubvh build completed but import smoke failed.',
        },
        missing,
    )


def install_flash_attn_stage(venv_dir: Path, ext_dir: Path, host_facts: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if is_linux_arm64_host(host_facts):
        missing = ['flash_attn']
        message = 'flash_attn stage skipped on Linux ARM64 host; continuing with degraded PyTorch SDPA fallback.'
        return (
            {
                'attempted': False,
                'required': False,
                'degradable': True,
                'status': 'degraded',
                'commands': [],
                'import_smoke_missing': missing,
                'failure_message': message,
                'skip_reason': message,
            },
            missing,
        )

    command = [str(venv_python(venv_dir)), '-m', 'pip', 'install', '--no-build-isolation', FLASH_ATTN_PACKAGE]
    rendered_command = ' '.join(command)
    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        if os.environ.get('ULTRASHAPE_SETUP_TEST_FLASH_ATTN_STAGE_FAIL') != 'install':
            create_stub_package(venv_site_packages(venv_dir), 'flash_attn')
    else:
        try:
            subprocess.run(command, check=True, env=child_env())
        except subprocess.CalledProcessError:
            missing = ['flash_attn']
            return (
                {
                    'attempted': True,
                    'required': False,
                    'degradable': True,
                    'status': 'degraded',
                    'commands': [rendered_command],
                    'import_smoke_missing': missing,
                    'failure_message': 'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
                },
                missing,
            )

    missing = run_import_smoke(venv_dir, ext_dir, ['flash_attn'])
    return (
        {
            'attempted': True,
            'required': False,
            'degradable': True,
            'status': 'ready' if not missing else 'degraded',
            'commands': [rendered_command],
            'import_smoke_missing': missing,
            'failure_message': None if not missing else 'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
        },
        missing,
    )


def run_import_smoke(venv_dir: Path, ext_dir: Path, modules: list[str]) -> list[str]:
    script = [
        'import importlib, json, sys',
        f'sys.path.insert(0, {str((ext_dir / VENDOR_RELATIVE.parent).resolve())!r})',
        f'modules = {modules!r}',
        'missing = []',
        'for module in modules:',
        '    try:',
        '        importlib.import_module(module)',
        '    except Exception:',
        '        missing.append(module)',
        'print(json.dumps(missing))',
    ]
    result = subprocess.run(
        [str(venv_python(venv_dir)), '-c', '\n'.join(script)],
        capture_output=True,
        text=True,
        env=child_env(),
        check=False,
    )
    if result.returncode != 0:
        return list(modules)
    payload = json.loads(result.stdout or '[]')
    return [item for item in payload if isinstance(item, str)]


def run_runtime_import_smoke(venv_dir: Path, ext_dir: Path) -> dict[str, Any]:
    script = [
        'import importlib, json, sys, types',
        f'sys.path.insert(0, {str((ext_dir / VENDOR_RELATIVE.parent).resolve())!r})',
        'pipelines_stub = types.ModuleType("ultrashape_runtime.pipelines")',
        'pipelines_stub.load_runtime_config = lambda path: {"config_path": path}',
        'pipelines_stub.run_refine_pipeline = lambda **kwargs: {"file_path": "", "format": kwargs.get("output_format", "glb"), "backend": kwargs.get("backend", "local"), "metrics": {}, "fallbacks": [], "subtrees_loaded": []}',
        'sys.modules["ultrashape_runtime.pipelines"] = pipelines_stub',
        'try:',
        '    importlib.import_module("ultrashape_runtime.local_runner")',
        '    print(json.dumps({"ok": True, "target": "ultrashape_runtime.local_runner"}))',
        'except Exception as error:',
        '    print(json.dumps({"ok": False, "target": "ultrashape_runtime.local_runner", "error_class": error.__class__.__name__, "error_message": str(error)}))',
        '    sys.exit(1)',
    ]
    result = subprocess.run(
        [str(venv_python(venv_dir)), '-c', '\n'.join(script)],
        capture_output=True,
        text=True,
        env=child_env(),
        check=False,
    )
    try:
        payload = json.loads(result.stdout or '{}')
    except json.JSONDecodeError:
        payload = {}

    ok = bool(payload.get('ok')) and result.returncode == 0
    return {
        'target': 'ultrashape_runtime.local_runner',
        'ok': ok,
        'status': 'ready' if ok else 'blocked',
        'error_class': payload.get('error_class') if isinstance(payload.get('error_class'), str) else None,
        'failure_message': None if ok else (payload.get('error_message') if isinstance(payload.get('error_message'), str) else (result.stderr.strip() or result.stdout.strip())),
    }


def _missing_checkpoint_subtree_tokens(message: str) -> list[str]:
    prefix = 'Required checkpoint subtrees are missing:'
    if not message.startswith(prefix):
        return []
    raw_names = message[len(prefix):].strip().rstrip('.')
    return [f'checkpoint-subtree:{name.strip()}' for name in raw_names.split(',') if name.strip()]


def run_checkpoint_smoke(venv_dir: Path, ext_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return {
            'ok': False,
            'status': 'blocked',
            'required_subtrees': ['vae', 'dit', 'conditioner'],
            'missing_required': [f'weight:{CHECKPOINT_RELATIVE.as_posix()}'],
            'failure_message': f'Required checkpoint is not readable: {checkpoint_path}.',
        }

    script = [
        'import json, sys',
        f'sys.path.insert(0, {str((ext_dir / VENDOR_RELATIVE.parent).resolve())!r})',
        'from ultrashape_runtime.utils.checkpoint import load_checkpoint_subtrees, expected_checkpoint_subtrees',
        f'checkpoint = {str(checkpoint_path)!r}',
        f'ext_dir = {str(ext_dir)!r}',
        'required = list(expected_checkpoint_subtrees())',
        'try:',
        '    bundle = load_checkpoint_subtrees(checkpoint, None, ext_dir, required)',
        '    print(json.dumps({"ok": True, "required_subtrees": required, "subtrees_loaded": bundle.get("subtrees_loaded", []), "summary": bundle.get("summary", {})}))',
        'except Exception as error:',
        '    print(json.dumps({"ok": False, "required_subtrees": required, "error_class": error.__class__.__name__, "error_code": getattr(error, "code", None), "error_message": str(error)}))',
        '    sys.exit(1)',
    ]
    result = subprocess.run(
        [str(venv_python(venv_dir)), '-c', '\n'.join(script)],
        capture_output=True,
        text=True,
        env=child_env(),
        check=False,
    )
    try:
        payload = json.loads(result.stdout or '{}')
    except json.JSONDecodeError:
        payload = {}

    ok = bool(payload.get('ok')) and result.returncode == 0
    required_subtrees = payload.get('required_subtrees') if isinstance(payload.get('required_subtrees'), list) else ['vae', 'dit', 'conditioner']
    failure_message = payload.get('error_message') if isinstance(payload.get('error_message'), str) else (result.stderr.strip() or result.stdout.strip())
    missing_required = [] if ok else _missing_checkpoint_subtree_tokens(failure_message)
    if not ok and not missing_required:
        missing_required = [f'weight:{CHECKPOINT_RELATIVE.as_posix()}']

    return {
        'ok': ok,
        'status': 'ready' if ok else 'blocked',
        'required_subtrees': [item for item in required_subtrees if isinstance(item, str)],
        'subtrees_loaded': payload.get('subtrees_loaded') if isinstance(payload.get('subtrees_loaded'), list) else [],
        'summary': payload.get('summary') if isinstance(payload.get('summary'), dict) else {},
        'error_class': payload.get('error_class') if isinstance(payload.get('error_class'), str) else None,
        'error_code': payload.get('error_code') if isinstance(payload.get('error_code'), str) else None,
        'failure_message': None if ok else failure_message,
        'missing_required': missing_required,
    }


def build_weight_source_descriptor(kind: str, source: Path | str) -> dict[str, Any]:
    source_value = str(Path(source).resolve()) if kind in {'ext-dir', 'required_weight_path', 'env-local', 'repo-local'} else str(source)
    return {'kind': kind, 'source': source_value}


def copy_required_weight(source_path: Path, target_path: Path) -> bool:
    if not source_path.is_file():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not same_path(source_path, target_path):
        copy2(source_path, target_path)
    return target_path.is_file()


def resolve_hf_source(payload: dict[str, Any]) -> dict[str, Any]:
    repo_id = payload.get('weight_repo_id') if isinstance(payload.get('weight_repo_id'), str) and payload.get('weight_repo_id').strip() else None
    revision = payload.get('weight_repo_revision') if isinstance(payload.get('weight_repo_revision'), str) and payload.get('weight_repo_revision').strip() else None
    token = payload.get('weight_hf_token') if isinstance(payload.get('weight_hf_token'), str) and payload.get('weight_hf_token').strip() else os.environ.get('ULTRASHAPE_WEIGHT_HF_TOKEN')
    chosen_repo = repo_id or os.environ.get('ULTRASHAPE_WEIGHT_REPO_ID') or DEFAULT_WEIGHT_REPO_ID
    chosen_revision = revision or os.environ.get('ULTRASHAPE_WEIGHT_REPO_REVISION')
    return {
        'kind': 'hf-default' if not (repo_id or revision) else 'hf-override',
        'repo_id': chosen_repo,
        'revision': chosen_revision,
        'filename': WEIGHT_FILENAME,
        'token': token,
        'auth_used': bool(token),
        'source': f'huggingface://{chosen_repo}/{WEIGHT_FILENAME}@{chosen_revision or "default"}',
    }


def parse_hf_error(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {'error_message': raw.strip()}
    return payload if isinstance(payload, dict) else {}


def classify_hf_failure(error_class: str | None, status_code: int | None) -> str:
    if error_class in {'GatedRepoError'} or (error_class == 'HfHubHTTPError' and status_code in {401, 403}):
        return 'auth'
    if error_class in {'RepositoryNotFoundError', 'RevisionNotFoundError', 'RemoteEntryNotFoundError', 'EntryNotFoundError'} or (error_class == 'HfHubHTTPError' and status_code == 404):
        return 'not-found'
    if error_class in {'LocalEntryNotFoundError', 'ConnectionError', 'Timeout', 'ProxyError'}:
        return 'network'
    return 'other'


def download_hf_weight(venv_dir: Path, target_path: Path, hf_source: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    script = [
        'import json, sys',
        'from pathlib import Path',
        'from shutil import copy2',
        'from huggingface_hub import hf_hub_download',
        f'repo_id = {hf_source["repo_id"]!r}',
        f'filename = {hf_source["filename"]!r}',
        f'revision = {hf_source["revision"]!r}',
        f'token = {hf_source["token"]!r}',
        f'target = Path({str(target_path)!r})',
        f'cache_dir = Path({str(cache_dir)!r})',
        'try:',
        '    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token, local_dir=cache_dir, local_dir_use_symlinks=False))',
        '    target.parent.mkdir(parents=True, exist_ok=True)',
        '    if downloaded.resolve() != target.resolve():',
        '        copy2(downloaded, target)',
        '    print(json.dumps({"ok": True, "downloaded": str(downloaded)}))',
        'except Exception as error:',
        '    print(json.dumps({"ok": False, "error_class": error.__class__.__name__, "error_message": str(error), "status_code": getattr(error, "status_code", None)}))',
        '    sys.exit(1)',
    ]
    result = subprocess.run(
        [str(venv_python(venv_dir)), '-c', '\n'.join(script)],
        capture_output=True,
        text=True,
        env=child_env(),
        check=False,
    )
    if result.returncode == 0:
        return {'ok': target_path.is_file()}
    payload = parse_hf_error(result.stdout or result.stderr)
    error_class = payload.get('error_class') if isinstance(payload.get('error_class'), str) else 'CalledProcessError'
    status_code = payload.get('status_code') if isinstance(payload.get('status_code'), int) else None
    error_message = payload.get('error_message') if isinstance(payload.get('error_message'), str) else (result.stderr.strip() or result.stdout.strip())
    return {
        'ok': False,
        'error_class': error_class,
        'status_code': status_code,
        'error_message': error_message,
        'failure_classification': classify_hf_failure(error_class, status_code),
    }


def acquire_required_weight(ext_dir: Path, payload: dict[str, Any], venv_dir: Path) -> dict[str, Any]:
    target_path = ext_dir / CHECKPOINT_RELATIVE
    attempted = [build_weight_source_descriptor('ext-dir', target_path)]

    def result_template(acquired: bool, resolved: dict[str, Any] | None) -> dict[str, Any]:
        return {
            'acquired': acquired,
            'target_path': str(target_path),
            'attempted_source_kinds': [item['kind'] for item in attempted],
            'attempted_sources': [item['source'] for item in attempted],
            'resolved_source_kind': resolved.get('kind') if resolved else None,
            'resolved_source': resolved.get('source') if resolved else None,
            'weight_source_repo_id': resolved.get('repo_id') if resolved else None,
            'weight_source_filename': resolved.get('filename', WEIGHT_FILENAME) if resolved else WEIGHT_FILENAME,
            'weight_source_revision': resolved.get('revision') if resolved else None,
            'weight_source_auth_used': resolved.get('auth_used') if resolved else None,
            'failure_classification': None,
            'error_class': None,
            'error_message': None,
        }

    if target_path.is_file():
        return result_template(True, attempted[0])

    local_candidates: list[dict[str, Any]] = []
    required_weight_path = payload.get('required_weight_path')
    if isinstance(required_weight_path, str) and required_weight_path.strip():
        local_candidates.append(build_weight_source_descriptor('required_weight_path', Path(required_weight_path.strip())))
    env_source = os.environ.get('ULTRASHAPE_WEIGHT_SOURCE_PATH')
    if env_source:
        local_candidates.append(build_weight_source_descriptor('env-local', Path(env_source)))
    local_candidates.append(build_weight_source_descriptor('repo-local', repo_root() / CHECKPOINT_RELATIVE))

    for candidate in local_candidates:
        attempted.append(candidate)
        if copy_required_weight(Path(candidate['source']), target_path):
            return result_template(True, candidate)

    hf_source = resolve_hf_source(payload)
    attempted.append(build_weight_source_descriptor(hf_source['kind'], hf_source['source']))
    attempted[-1].update({
        'repo_id': hf_source['repo_id'],
        'filename': hf_source['filename'],
        'revision': hf_source['revision'],
        'auth_used': hf_source['auth_used'],
    })
    hf_result = download_hf_weight(venv_dir, target_path, hf_source, ext_dir / '.hf-cache')
    if hf_result.get('ok'):
        return result_template(True, attempted[-1])
    failed = result_template(False, attempted[-1])
    failed['failure_classification'] = hf_result.get('failure_classification')
    failed['error_class'] = hf_result.get('error_class')
    failed['error_message'] = hf_result.get('error_message')
    return failed


def build_summary_and_readiness(
    *,
    ext_dir: Path,
    args: argparse.Namespace,
    host_facts: dict[str, Any],
    venv_dir: Path,
    config_ready: bool,
    vendor_ready: bool,
    pip_bootstrap: dict[str, Any],
    dependency_install: dict[str, Any],
    native_install: dict[str, Any],
    weight_result: dict[str, Any],
    runtime_import_smoke: dict[str, Any],
    checkpoint_smoke: dict[str, Any],
    required_import_failures: list[str],
    conditional_import_failures: list[str],
    degradable_import_failures: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = ext_dir / CHECKPOINT_RELATIVE
    missing_required: list[str] = []
    if not config_ready:
        missing_required.append(f'config:{CONFIG_RELATIVE.as_posix()}')
    if not vendor_ready:
        missing_required.append(f'vendor:{VENDOR_RELATIVE.as_posix()}')
    missing_required.extend(f'import:{name}' for name in required_import_failures)
    if not weight_result.get('acquired'):
        missing_required.append(f'weight:{CHECKPOINT_RELATIVE.as_posix()}')
    if not runtime_import_smoke.get('ok'):
        missing_required.append(f'runtime-import:{runtime_import_smoke.get("target", "ultrashape_runtime.local_runner")}')
    missing_required.extend(
        item
        for item in checkpoint_smoke.get('missing_required', [])
        if isinstance(item, str) and item not in missing_required
    )

    cubvh_state = native_install.get('cubvh') if isinstance(native_install.get('cubvh'), dict) else {}
    if cubvh_state.get('status') == 'blocked':
        missing_required.append('native-stage:cubvh')

    missing_conditional = [f'import:{name}' for name in conditional_import_failures]
    missing_degradable = [f'import:{name}' for name in degradable_import_failures]
    missing_optional = [*missing_conditional, *missing_degradable]

    required_imports_ok = (
        config_ready
        and vendor_ready
        and bool(runtime_import_smoke.get('ok'))
        and not required_import_failures
        and cubvh_state.get('status') != 'blocked'
    )
    weights_ready = checkpoint_path.is_file() and bool(weight_result.get('acquired')) and bool(checkpoint_smoke.get('ok'))
    runtime_closure_ready = config_ready and vendor_ready and bool(runtime_import_smoke.get('ok'))
    runtime_ready = required_imports_ok and weights_ready and runtime_closure_ready
    runtime_modes = build_runtime_modes(portable_available=required_imports_ok and runtime_closure_ready)

    status = 'blocked'
    if runtime_ready and missing_optional:
        status = 'degraded'
    elif runtime_ready:
        status = 'ready'

    readiness = {
        'backend': 'local',
        'checkpoint': str(checkpoint_path),
        'config_path': str(ext_dir / CONFIG_RELATIVE),
        'config_ready': config_ready,
        'cuda_version': args.cuda_version,
        'dependency_install': dependency_install,
        'ext_dir': str(ext_dir),
        'gpu_sm': args.gpu_sm,
        'host_facts': host_facts,
        'install_ready': not missing_required,
        'install_success': not missing_required,
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'missing_conditional': missing_conditional,
        'missing_degradable': missing_degradable,
        'native_install': native_install,
        'pip_bootstrap': pip_bootstrap,
        'python_exe': args.python_exe,
        'required_checkpoint_subtrees': checkpoint_smoke.get('required_subtrees', ['vae', 'dit', 'conditioner']),
        'required_imports_ok': required_imports_ok,
        'runtime_closure_import': runtime_import_smoke,
        'runtime_ready': runtime_ready,
        'runtime_closure_ready': runtime_closure_ready,
        'runtime_closure_reason': (
            'The vendored dual-mode runtime seam is staged and importable for local refinement.'
            if runtime_closure_ready
            else runtime_import_smoke.get('failure_message')
        ),
        'runtime_modes': runtime_modes,
        'checkpoint_smoke': checkpoint_smoke,
        'status': status,
        'vendor_path': str(ext_dir / VENDOR_RELATIVE),
        'vendor_ready': vendor_ready,
        'venv_dir': str(venv_dir),
        'weights_ready': weights_ready,
    }
    readiness.update(build_weight_diagnostics(weight_result))

    summary = dict(readiness)
    summary['pip_bootstrap'] = pip_bootstrap
    summary['dependency_install'] = dependency_install
    summary['native_install'] = native_install
    return readiness, summary


def build_runtime_modes(*, portable_available: bool) -> dict[str, Any]:
    selection = 'portable-only' if portable_available else 'blocked'
    return {
        'selection': selection,
        'requested': 'auto',
        'active': 'portable' if portable_available else None,
        'real': {
            'available': False,
            'adapter': REAL_MODE_ADAPTER,
            'reason': REAL_MODE_REASON,
            'blockers': ['adapter:authoritative-upstream-module-graph'],
        },
        'portable': {
            'available': portable_available,
            'authoritative': False,
            'reason': PORTABLE_MODE_REASON if portable_available else 'Portable fallback is blocked because required runtime prerequisites are missing.',
        },
    }


def build_weight_diagnostics(weight_result: dict[str, Any]) -> dict[str, Any]:
    return {
        'attempted_weight_source_kinds': weight_result.get('attempted_source_kinds', []),
        'attempted_weight_sources': weight_result.get('attempted_sources', []),
        'resolved_weight_source_kind': weight_result.get('resolved_source_kind'),
        'resolved_weight_source': weight_result.get('resolved_source'),
        'weight_source_repo_id': weight_result.get('weight_source_repo_id'),
        'weight_source_filename': weight_result.get('weight_source_filename', WEIGHT_FILENAME),
        'weight_source_revision': weight_result.get('weight_source_revision'),
        'weight_source_auth_used': weight_result.get('weight_source_auth_used'),
        'weight_source_failure_classification': weight_result.get('failure_classification'),
        'weight_source_error_class': weight_result.get('error_class'),
        'weight_source_error_message': weight_result.get('error_message'),
    }


def skipped_stage(name: str, reason: str) -> dict[str, Any]:
    return {
        'mode': 'skipped',
        'stage': name,
        'commands': [],
        'installed_modules': [],
        'reason': reason,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8')


def main() -> int:
    args = parse_args()
    ext_dir = Path(args.ext_dir).resolve()
    ext_dir.mkdir(parents=True, exist_ok=True)
    host_facts = detect_host_facts()

    config_ready, vendor_ready = stage_runtime_assets(ext_dir)
    venv_dir = ensure_venv(ext_dir)
    runtime_import_smoke = run_runtime_import_smoke(venv_dir, ext_dir)

    cubvh_prerequisites = detect_cubvh_prerequisites(host_facts)
    if not cubvh_prerequisites['ok']:
        blocked_reason = 'Skipped because cubvh prerequisites are already blocked.'
        pip_bootstrap = skipped_stage('pip_bootstrap', blocked_reason)
        dependency_install = skipped_stage('dependency_install', blocked_reason)
        native_install = {
            'cubvh': {
                'attempted': False,
                'required': True,
                'status': 'blocked',
                'source': CUBVH_SOURCE,
                'pinned_ref': CUBVH_PINNED_REF,
                'detected_prerequisites': cubvh_prerequisites['detected_prerequisites'],
                'missing_prerequisites': cubvh_prerequisites['missing_prerequisites'],
                'failure_message': cubvh_prerequisites['failure_message'],
            },
            'flash_attn': {
                'attempted': False,
                'required': False,
                'degradable': True,
                'status': 'pending',
                'commands': [],
            },
        }
        readiness, summary = build_summary_and_readiness(
            ext_dir=ext_dir,
            args=args,
            host_facts=host_facts,
            venv_dir=venv_dir,
            config_ready=config_ready,
            vendor_ready=vendor_ready,
            pip_bootstrap=pip_bootstrap,
            dependency_install=dependency_install,
            native_install=native_install,
            weight_result={
                'acquired': False,
                'attempted_source_kinds': [],
                'attempted_sources': [],
                'weight_source_filename': WEIGHT_FILENAME,
            },
            runtime_import_smoke=runtime_import_smoke,
            checkpoint_smoke=run_checkpoint_smoke(venv_dir, ext_dir, ext_dir / CHECKPOINT_RELATIVE),
            required_import_failures=['cubvh'],
            conditional_import_failures=[],
            degradable_import_failures=[],
        )
        write_json(ext_dir / READINESS_FILE, readiness)
        write_json(ext_dir / SUMMARY_FILE, summary)
        json.dump(summary, sys.stdout)
        sys.stdout.write('\n')
        return 1

    pip_bootstrap = bootstrap_packaging_tools(venv_dir)
    dependency_install = install_core_dependencies(venv_dir)
    cubvh_stage, cubvh_missing = install_cubvh_stage(venv_dir, ext_dir)
    flash_attn_stage, flash_attn_missing = install_flash_attn_stage(venv_dir, ext_dir, host_facts)
    native_install = {
        'cubvh': cubvh_stage,
        'flash_attn': flash_attn_stage,
    }

    weight_result = acquire_required_weight(ext_dir, args.payload, venv_dir)
    checkpoint_smoke = run_checkpoint_smoke(venv_dir, ext_dir, ext_dir / CHECKPOINT_RELATIVE)
    required_import_failures = run_import_smoke(venv_dir, ext_dir, REQUIRED_IMPORTS)
    conditional_import_failures = run_import_smoke(venv_dir, ext_dir, CONDITIONAL_IMPORTS)
    degradable_import_failures = sorted(set(run_import_smoke(venv_dir, ext_dir, DEGRADABLE_IMPORTS) + flash_attn_missing))

    if cubvh_missing and 'cubvh' not in required_import_failures:
        required_import_failures.append('cubvh')

    readiness, summary = build_summary_and_readiness(
        ext_dir=ext_dir,
        args=args,
        host_facts=host_facts,
        venv_dir=venv_dir,
        config_ready=config_ready,
        vendor_ready=vendor_ready,
        pip_bootstrap=pip_bootstrap,
        dependency_install=dependency_install,
        native_install=native_install,
        weight_result=weight_result,
        runtime_import_smoke=runtime_import_smoke,
        checkpoint_smoke=checkpoint_smoke,
        required_import_failures=sorted(set(required_import_failures)),
        conditional_import_failures=conditional_import_failures,
        degradable_import_failures=degradable_import_failures,
    )

    write_json(ext_dir / READINESS_FILE, readiness)
    write_json(ext_dir / SUMMARY_FILE, summary)
    json.dump(summary, sys.stdout)
    sys.stdout.write('\n')
    return 0 if readiness['install_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
