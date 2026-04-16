import json
import os
from datetime import datetime, timezone
from shutil import copy2, copytree
import subprocess
import sys
from pathlib import Path
from venv import EnvBuilder


REQUIRED_STRING_KEYS = ('python_exe', 'ext_dir')
REQUIRED_DEPENDENCIES = [
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
    'rembg',
    'onnxruntime',
    'safetensors',
    'tqdm',
]
OPTIONAL_DEPENDENCIES = ['cubvh', 'flash_attn', 'diffusers', 'diso']
EXPECTED_WEIGHTS = ['models/ultrashape/ultrashape_v1.pt']
RUNTIME_LAYOUT_VERSION = '1'
WEIGHT_FAILURE_CODE = 'WEIGHT_ACQUISITION_FAILED'
IMPORT_FAILURE_CODE = 'REQUIRED_IMPORT_SMOKE_FAILED'
OPTIONAL_IMPORT_MODULES = ['cubvh', 'flash_attn', 'diffusers', 'diso']
REQUIRED_IMPORT_MODULES = [
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
    'rembg',
    'onnxruntime',
    'safetensors',
    'tqdm',
]
PACKAGE_STUBS = {
    'torch': '__version__ = "0.0-test"\n',
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
    'huggingface_hub': 'def snapshot_download(*args, **kwargs):\n    raise RuntimeError("stub")\n',
    'accelerate': '__version__ = "0.0-test"\n',
    'rembg': '__version__ = "0.0-test"\n',
    'onnxruntime': '__version__ = "0.0-test"\n',
    'safetensors': '__version__ = "0.0-test"\n',
    'tqdm': '__version__ = "0.0-test"\n',
    'cubvh': '__version__ = "0.0-test"\n',
    'flash_attn': '__version__ = "0.0-test"\n',
    'diffusers': '__version__ = "0.0-test"\n',
    'diso': '__version__ = "0.0-test"\n',
}
REQUIRED_RUNTIME_FILES = [
    'ultrashape_runtime/__init__.py',
    'ultrashape_runtime/pipelines.py',
    'ultrashape_runtime/preprocessors.py',
    'ultrashape_runtime/rembg.py',
    'ultrashape_runtime/surface_loaders.py',
    'ultrashape_runtime/schedulers.py',
    'ultrashape_runtime/utils/__init__.py',
    'ultrashape_runtime/utils/checkpoint.py',
    'ultrashape_runtime/utils/mesh.py',
    'ultrashape_runtime/utils/tensors.py',
    'ultrashape_runtime/models/conditioner_mask.py',
    'ultrashape_runtime/models/denoisers/__init__.py',
    'ultrashape_runtime/models/denoisers/dit_mask.py',
    'ultrashape_runtime/models/denoisers/moe_layers.py',
    'ultrashape_runtime/models/autoencoders/__init__.py',
    'ultrashape_runtime/models/autoencoders/model.py',
    'ultrashape_runtime/models/autoencoders/attention_blocks.py',
    'ultrashape_runtime/models/autoencoders/attention_processors.py',
    'ultrashape_runtime/models/autoencoders/surface_extractors.py',
    'ultrashape_runtime/models/autoencoders/volume_decoders.py',
]


def parse_args(argv: list[str]) -> dict[str, object]:
    if len(argv) < 2:
        raise SystemExit('setup.py expects one JSON object argument.')

    raw = argv[1].strip()
    if not raw:
        raise SystemExit('setup.py expects one JSON object argument.')

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit('setup.py expects one JSON object argument.')

    return payload


def require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{key} is required in the Modly setup payload.')
    return value.strip()


def require_scalar(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)

    if isinstance(value, bool):
        raise SystemExit(f'{key} is required in the Modly setup payload.')

    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)

    if isinstance(value, str) and value.strip():
        return value.strip()

    raise SystemExit(f'{key} is required in the Modly setup payload.')


def resolve_install_context(payload: dict[str, object]) -> dict[str, str]:
    context = {key: require_string(payload, key) for key in REQUIRED_STRING_KEYS}
    context['gpu_sm'] = require_scalar(payload, 'gpu_sm')

    cuda_version = payload.get('cuda_version')
    if isinstance(cuda_version, bool):
        raise SystemExit('cuda_version must be a string or number when provided in the Modly setup payload.')

    if isinstance(cuda_version, (int, float)):
        context['cuda_version'] = str(int(cuda_version)) if isinstance(cuda_version, float) and cuda_version.is_integer() else str(cuda_version)
    elif isinstance(cuda_version, str) and cuda_version.strip():
        context['cuda_version'] = cuda_version.strip()

    return context


def ensure_venv(ext_dir: str) -> Path:
    base_dir = Path(ext_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    venv_dir = base_dir / 'venv'
    EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
    return venv_dir


def get_venv_python(venv_dir: Path) -> Path:
    return venv_dir / 'bin' / 'python'


def get_venv_site_packages(venv_dir: Path) -> Path:
    venv_python = get_venv_python(venv_dir)
    outcome = subprocess.run(
        [str(venv_python), '-c', 'import site; print(site.getsitepackages()[0])'],
        check=True,
        capture_output=True,
        encoding='utf8',
    )
    return Path(outcome.stdout.strip())


def normalize_numeric_token(raw: str | None) -> str | None:
    if raw is None:
        return None
    return ''.join(character for character in raw if character.isdigit()) or None


def select_torch_profile(context: dict[str, str]) -> dict[str, str]:
    gpu_sm = int(normalize_numeric_token(context['gpu_sm']) or '0')
    cuda_version = int(normalize_numeric_token(context.get('cuda_version')) or '128')

    if gpu_sm < 90 and cuda_version < 128:
        raise SystemExit('UltraShape MVP setup requires Linux ARM64 with CUDA 12.8-class support.')

    return {
        'profile': 'linux-arm64-cu128-sm90+',
        'torch': 'torch==2.7.0+cu128',
        'torchvision': 'torchvision==0.22.0',
    }


def ensure_runtime_layout(ext_dir: str) -> None:
    repo_root = Path(__file__).resolve().parent
    install_root = Path(ext_dir)
    runtime_root = install_root / 'runtime'
    source_config = (repo_root / 'runtime' / 'configs' / 'infer_dit_refine.yaml').resolve()
    source_vendor_root = (repo_root / 'runtime' / 'vendor' / 'ultrashape_runtime').resolve()
    config_target = runtime_root / 'configs' / 'infer_dit_refine.yaml'
    config_target.parent.mkdir(parents=True, exist_ok=True)
    if source_config != config_target.resolve():
        copy2(source_config, config_target)

    runtime_package = runtime_root / 'ultrashape_runtime'
    copytree(source_vendor_root, runtime_package, dirs_exist_ok=True)

    (runtime_root / '.locks').mkdir(parents=True, exist_ok=True)
    (install_root / 'models' / 'ultrashape').mkdir(parents=True, exist_ok=True)


def install_stub_packages(venv_dir: Path) -> dict[str, object]:
    missing = {
        module.strip()
        for module in os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS_MISSING', '').split(',')
        if module.strip()
    }
    site_packages = get_venv_site_packages(venv_dir)
    site_packages.mkdir(parents=True, exist_ok=True)

    installed_required = []
    for module in REQUIRED_IMPORT_MODULES:
        if module in missing:
            continue
        package_dir = site_packages / module
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / '__init__.py').write_text(PACKAGE_STUBS[module], encoding='utf8')
        installed_required.append(module)

    for module in OPTIONAL_IMPORT_MODULES:
        if module in missing:
            continue
        package_dir = site_packages / module
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / '__init__.py').write_text(PACKAGE_STUBS[module], encoding='utf8')

    return {
        'mode': 'stub',
        'installed_required_modules': installed_required,
        'missing_stubbed_modules': sorted(missing),
        'commands': ['stub-site-packages'],
    }


def install_required_dependencies(venv_dir: Path, profile: dict[str, str]) -> dict[str, object]:
    if os.environ.get('ULTRASHAPE_SETUP_TEST_STUB_DEPS') == '1':
        return install_stub_packages(venv_dir)

    venv_python = get_venv_python(venv_dir)
    pip_commands = [
        [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'],
        [
            str(venv_python),
            '-m',
            'pip',
            'install',
            '--extra-index-url',
            'https://download.pytorch.org/whl/cu128',
            profile['torch'],
            profile['torchvision'],
            *[dependency for dependency in REQUIRED_DEPENDENCIES if dependency not in {profile['torch'], profile['torchvision']}],
        ],
    ]

    rendered_commands = []
    for command in pip_commands:
        rendered_commands.append(' '.join(command))
        subprocess.run(command, check=True)

    return {
        'mode': 'pip',
        'installed_required_modules': REQUIRED_IMPORT_MODULES,
        'missing_stubbed_modules': [],
        'commands': rendered_commands,
    }


def copy_required_weight(source_path: Path, target_path: Path) -> bool:
    if not source_path.is_file():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target_path.resolve():
        copy2(source_path, target_path)
    return target_path.is_file()


def acquire_required_weight(ext_dir: str, payload: dict[str, object], venv_dir: Path) -> dict[str, object]:
    install_root = Path(ext_dir)
    target_path = install_root / EXPECTED_WEIGHTS[0]
    repo_root = Path(__file__).resolve().parent
    attempted_sources = [str(target_path)]

    if target_path.is_file():
        return {
            'acquired': True,
            'target_path': str(target_path),
            'attempted_sources': attempted_sources,
            'resolved_source': str(target_path),
        }

    local_candidates = []
    required_weight_path = payload.get('required_weight_path')
    if isinstance(required_weight_path, str) and required_weight_path.strip():
        local_candidates.append(Path(required_weight_path.strip()))

    env_source = os.environ.get('ULTRASHAPE_WEIGHT_SOURCE_PATH')
    if env_source:
        local_candidates.append(Path(env_source))

    repo_weight = repo_root / EXPECTED_WEIGHTS[0]
    local_candidates.append(repo_weight)

    for candidate in local_candidates:
        attempted_sources.append(str(candidate))
        if copy_required_weight(candidate, target_path):
            return {
                'acquired': True,
                'target_path': str(target_path),
                'attempted_sources': attempted_sources,
                'resolved_source': str(candidate),
            }

    hf_repo_id = payload.get('weight_repo_id') if isinstance(payload.get('weight_repo_id'), str) else os.environ.get('ULTRASHAPE_WEIGHT_REPO_ID')
    hf_repo_revision = payload.get('weight_repo_revision') if isinstance(payload.get('weight_repo_revision'), str) else os.environ.get('ULTRASHAPE_WEIGHT_REPO_REVISION')
    if isinstance(hf_repo_id, str) and hf_repo_id.strip():
        attempted_sources.append(f'huggingface://{hf_repo_id.strip()}')
        venv_python = get_venv_python(venv_dir)
        download_cache = install_root / '.hf-cache'
        command = [
            str(venv_python),
            '-c',
            (
                'from huggingface_hub import snapshot_download\n'
                'from pathlib import Path\n'
                'repo_id = Path("' + hf_repo_id.strip().replace('"', '\\"') + '")\n'
            ),
        ]
        script_lines = [
            'from huggingface_hub import snapshot_download',
            'from pathlib import Path',
            f'repo_id = {hf_repo_id.strip()!r}',
            f'revision = {hf_repo_revision.strip()!r}' if isinstance(hf_repo_revision, str) and hf_repo_revision.strip() else 'revision = None',
            f'target = Path({str(target_path)!r})',
            f'cache_dir = Path({str(download_cache)!r})',
            f'filename = {Path(EXPECTED_WEIGHTS[0]).name!r}',
            'snapshot_path = Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=[filename], cache_dir=cache_dir))',
            'source = next(snapshot_path.rglob(filename))',
            'target.parent.mkdir(parents=True, exist_ok=True)',
            'target.write_bytes(source.read_bytes())',
        ]
        subprocess.run([str(venv_python), '-c', '\n'.join(script_lines)], check=True)
        if target_path.is_file():
            return {
                'acquired': True,
                'target_path': str(target_path),
                'attempted_sources': attempted_sources,
                'resolved_source': f'huggingface://{hf_repo_id.strip()}',
            }

    return {
        'acquired': False,
        'target_path': str(target_path),
        'attempted_sources': attempted_sources,
        'resolved_source': None,
    }


def run_import_smoke(venv_dir: Path, ext_dir: str, modules: list[str]) -> list[str]:
    venv_python = get_venv_python(venv_dir)
    script_lines = [
        'import importlib',
        'import json',
        'import sys',
        f'sys.path.insert(0, {str(Path(ext_dir) / "runtime")!r})',
        f'modules = {modules!r}',
        'missing = []',
        'for module in modules:',
        '    try:',
        '        importlib.import_module(module)',
        '    except Exception:',
        '        missing.append(module)',
        'print(json.dumps(missing))',
    ]
    outcome = subprocess.run(
        [str(venv_python), '-c', '\n'.join(script_lines)],
        check=False,
        capture_output=True,
        encoding='utf8',
    )
    if outcome.returncode != 0:
        return modules
    return json.loads(outcome.stdout or '[]')


def list_missing_runtime_files(ext_dir: str) -> list[str]:
    runtime_root = Path(ext_dir) / 'runtime'
    return [path for path in REQUIRED_RUNTIME_FILES if not (runtime_root / path).is_file()]


def build_summary(context: dict[str, str], profile: dict[str, str]) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        'python_exe': context['python_exe'],
        'ext_dir': context['ext_dir'],
        'gpu_sm': context['gpu_sm'],
        'cuda_version': context.get('cuda_version'),
        'torch_profile': profile['profile'],
        'runtime_layout_version': RUNTIME_LAYOUT_VERSION,
        'installed_at': timestamp,
        'dependencies': {
            'required': REQUIRED_DEPENDENCIES,
            'optional': OPTIONAL_DEPENDENCIES,
        },
        'runtime_assets': {
            'required_files': REQUIRED_RUNTIME_FILES,
            'required_weights': EXPECTED_WEIGHTS,
            'optional_dependencies': OPTIONAL_DEPENDENCIES,
        },
        'patch_intent': {
            'flash_attn': 'optional -> PyTorch SDPA fallback',
            'cubvh': 'optional -> skimage.measure.marching_cubes fallback',
            'diffusers': 'optional outside mc-only MVP closure',
            'diso': 'optional and excluded from mc-only MVP',
        },
    }


def collect_install_health(ext_dir: str, required_import_failures: list[str], optional_import_failures: list[str]) -> dict[str, object]:
    install_root = Path(ext_dir)
    missing_weights = [weight for weight in EXPECTED_WEIGHTS if not (install_root / weight).is_file()]
    missing_runtime_files = list_missing_runtime_files(ext_dir)
    missing_required = [
        *missing_weights,
        *[f'runtime/{path}' for path in missing_runtime_files],
        *[f'import:{module}' for module in required_import_failures],
    ]
    failure_stage = None
    failure_code = None
    failure_detail = None

    if missing_weights:
        failure_stage = 'weight-validation'
        failure_code = WEIGHT_FAILURE_CODE
        failure_detail = 'Required UltraShape weights are missing after setup staging.'
    elif missing_runtime_files:
        failure_stage = 'runtime-validation'
        failure_code = 'RUNTIME_LAYOUT_INCOMPLETE'
        failure_detail = 'Required vendored runtime files are missing after setup staging.'
    elif required_import_failures:
        failure_stage = 'required-import-smoke'
        failure_code = IMPORT_FAILURE_CODE
        failure_detail = 'Required dependency import smoke failed after dependency installation.'

    install_success = not missing_required
    status = 'blocked'
    if install_success and optional_import_failures:
        status = 'degraded'
    elif install_success:
        status = 'ready'

    return {
        'missing_weights': missing_weights,
        'missing_runtime_files': missing_runtime_files,
        'missing_required': missing_required,
        'missing_optional': optional_import_failures,
        'failure_stage': failure_stage,
        'failure_code': failure_code,
        'failure_detail': failure_detail,
        'install_success': install_success,
        'status': status,
        'required_import_failures': required_import_failures,
    }


def build_readiness(ext_dir: str, health: dict[str, object]) -> dict[str, object]:
    missing_required = list(health['missing_required'])

    return {
        'status': health['status'],
        'backend': 'local',
        'mvp_scope': 'mc-only',
        'weights_ready': not health['missing_weights'],
        'required_imports_ok': not health['required_import_failures'] and not health['missing_runtime_files'],
        'missing_required': missing_required,
        'missing_optional': list(health['missing_optional']),
        'expected_weights': EXPECTED_WEIGHTS,
        'install_success': health['install_success'],
        'failure_stage': health['failure_stage'],
        'failure_code': health['failure_code'],
        'failure_detail': health['failure_detail'],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8')


def write_summary(ext_dir: str, context: dict[str, str], payload: dict[str, object], dependency_install: dict[str, object], weight_result: dict[str, object], required_import_failures: list[str], optional_import_failures: list[str]) -> None:
    summary_path = Path(ext_dir) / '.setup-summary.json'
    readiness_path = Path(ext_dir) / '.runtime-readiness.json'
    profile = select_torch_profile(context)
    ensure_runtime_layout(ext_dir)
    summary = build_summary(context, profile)
    health = collect_install_health(ext_dir, required_import_failures, optional_import_failures)
    summary.update({
        'dependency_install': dependency_install,
        'install_success': health['install_success'],
        'failure_stage': health['failure_stage'],
        'failure_code': health['failure_code'],
        'failure_detail': health['failure_detail'],
        'attempted_weight_sources': weight_result['attempted_sources'],
        'resolved_weight_source': weight_result['resolved_source'],
        'missing_required': health['missing_required'],
        'missing_optional': health['missing_optional'],
    })
    write_json(summary_path, summary)
    write_json(readiness_path, build_readiness(ext_dir, health))

    if not health['install_success']:
        raise SystemExit(1)


def main() -> int:
    payload = parse_args(sys.argv)
    context = resolve_install_context(payload)
    profile = select_torch_profile(context)
    venv_dir = ensure_venv(context['ext_dir'])
    dependency_install = install_required_dependencies(venv_dir, profile)
    ensure_runtime_layout(context['ext_dir'])
    weight_result = acquire_required_weight(context['ext_dir'], payload, venv_dir)
    required_import_failures = run_import_smoke(venv_dir, context['ext_dir'], REQUIRED_IMPORT_MODULES + ['ultrashape_runtime'])
    optional_import_failures = run_import_smoke(venv_dir, context['ext_dir'], OPTIONAL_IMPORT_MODULES)
    write_summary(context['ext_dir'], context, payload, dependency_install, weight_result, required_import_failures, optional_import_failures)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
