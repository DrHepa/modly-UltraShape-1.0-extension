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
WEIGHT_FILENAME = 'ultrashape_v1.pt'
WEIGHT_RELATIVE_PATH = f'models/ultrashape/{WEIGHT_FILENAME}'
EXPECTED_WEIGHTS = [WEIGHT_RELATIVE_PATH]
DEFAULT_WEIGHT_REPO_ID = 'infinith/UltraShape'
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
        'def _write_trace(payload):\n'
        '    trace_path = os.environ.get("ULTRASHAPE_SETUP_TEST_HF_TRACE_PATH")\n'
        '    if not trace_path:\n'
        '        return\n'
        '    Path(trace_path).write_text(json.dumps(payload), encoding="utf8")\n\n'
        'def hf_hub_download(repo_id, filename, revision=None, token=None, local_dir=None, local_dir_use_symlinks=False, **kwargs):\n'
        '    _write_trace({"api": "hf_hub_download", "repo_id": repo_id, "filename": filename, "revision": revision, "token": token})\n'
        '    scenario = os.environ.get("ULTRASHAPE_SETUP_TEST_HF_SCENARIO")\n'
        '    if scenario == "auth":\n'
        '        raise GatedRepoError("401 Unauthorized from Hugging Face")\n'
        '    if scenario == "network":\n'
        '        raise LocalEntryNotFoundError("Connection error while downloading UltraShape weights")\n'
        '    if scenario == "not-found":\n'
        '        raise RemoteEntryNotFoundError("404 missing ultrashape_v1.pt")\n'
        '    if scenario == "other":\n'
        '        raise RuntimeError("Unexpected HF failure during UltraShape acquisition")\n'
        '    if "ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE" not in os.environ:\n'
        '        raise RuntimeError("hf_hub_download test seam not configured")\n'
        '    target_dir = Path(local_dir) if local_dir else Path.cwd()\n'
        '    target_dir.mkdir(parents=True, exist_ok=True)\n'
        '    target_path = target_dir / filename\n'
        '    target_path.write_text(os.environ.get("ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE", "stub-weight"), encoding="utf8")\n'
        '    return str(target_path)\n\n'
        'def snapshot_download(*args, **kwargs):\n'
        '    raise RuntimeError("snapshot_download should not be used for UltraShape weight acquisition")\n'
    ),
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
HF_AUTH_ERROR_CLASSES = {'GatedRepoError'}
HF_NOT_FOUND_ERROR_CLASSES = {
    'RepositoryNotFoundError',
    'RevisionNotFoundError',
    'RemoteEntryNotFoundError',
    'EntryNotFoundError',
}
HF_NETWORK_ERROR_CLASSES = {
    'LocalEntryNotFoundError',
    'ConnectionError',
    'ConnectTimeout',
    'ReadTimeout',
    'Timeout',
    'ProxyError',
    'ChunkedEncodingError',
}


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
    copy_test_scenario = os.environ.get('ULTRASHAPE_SETUP_TEST_COPY_SCENARIO')
    if copy_test_scenario == 'permission-error':
        raise PermissionError('copy blocked for test')
    if copy_test_scenario == 'missing-file':
        raise FileNotFoundError('copy source disappeared for test')
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target_path.resolve():
        copy2(source_path, target_path)
    return target_path.is_file()


def render_weight_source(kind: str, source: Path | str | None) -> str | None:
    if source is None:
        return None
    if kind in {'ext-dir', 'required_weight_path', 'env-local', 'repo-local'}:
        return str(Path(source).resolve())
    return str(source)


def build_weight_source_descriptor(kind: str, source: Path | str) -> dict[str, object]:
    return {
        'kind': kind,
        'source': render_weight_source(kind, source),
    }


def render_hf_weight_source(repo_id: str, revision: str | None, filename: str = WEIGHT_FILENAME) -> str:
    return f'huggingface://{repo_id}/{filename}@{revision or "default"}'


def resolve_hf_source(payload: dict[str, object]) -> dict[str, object] | None:
    payload_repo_id = payload.get('weight_repo_id') if isinstance(payload.get('weight_repo_id'), str) and payload.get('weight_repo_id').strip() else None
    payload_revision = payload.get('weight_repo_revision') if isinstance(payload.get('weight_repo_revision'), str) and payload.get('weight_repo_revision').strip() else None
    env_repo_id = os.environ.get('ULTRASHAPE_WEIGHT_REPO_ID') or None
    env_revision = os.environ.get('ULTRASHAPE_WEIGHT_REPO_REVISION') or None
    token = payload.get('weight_hf_token') if isinstance(payload.get('weight_hf_token'), str) and payload.get('weight_hf_token').strip() else os.environ.get('ULTRASHAPE_WEIGHT_HF_TOKEN')
    repo_id = payload_repo_id or env_repo_id or DEFAULT_WEIGHT_REPO_ID
    revision = payload_revision or env_revision

    if not repo_id:
        return None

    kind = 'hf-default'
    if payload_repo_id or payload_revision or env_repo_id or env_revision:
        kind = 'hf-override'

    return {
        'kind': kind,
        'repo_id': repo_id,
        'revision': revision,
        'filename': WEIGHT_FILENAME,
        'token': token,
        'auth_used': bool(token),
        'source': render_hf_weight_source(repo_id, revision),
    }


def classify_local_weight_failure(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return 'not-found'
    if isinstance(error, OSError):
        return 'other'
    return 'other'


def classify_hf_weight_failure(error_class: str | None, status_code: int | None) -> str:
    if error_class in HF_AUTH_ERROR_CLASSES:
        return 'auth'
    if error_class == 'HfHubHTTPError' and status_code in {401, 403}:
        return 'auth'
    if error_class in HF_NOT_FOUND_ERROR_CLASSES:
        return 'not-found'
    if error_class == 'HfHubHTTPError' and status_code == 404:
        return 'not-found'
    if error_class in HF_NETWORK_ERROR_CLASSES or error_class == 'CalledProcessError':
        return 'network'
    return 'other'


def parse_hf_error_payload(raw_output: str) -> dict[str, object]:
    if not raw_output.strip():
        return {}
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {'error_message': raw_output.strip()}
    return parsed if isinstance(parsed, dict) else {}


def download_hf_weight(venv_dir: Path, target_path: Path, hf_source: dict[str, object], cache_dir: Path) -> dict[str, object]:
    venv_python = get_venv_python(venv_dir)
    script_lines = [
        'import json',
        'import sys',
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
        '    else:',
        '        target.write_bytes(downloaded.read_bytes())',
        '    print(json.dumps({"ok": True, "downloaded": str(downloaded)}))',
        'except Exception as error:',
        '    print(json.dumps({',
        '        "ok": False,',
        '        "error_class": error.__class__.__name__,',
        '        "error_message": str(error),',
        '        "status_code": getattr(error, "status_code", None),',
        '    }))',
        '    sys.exit(1)',
    ]
    outcome = subprocess.run(
        [str(venv_python), '-c', '\n'.join(script_lines)],
        check=False,
        capture_output=True,
        encoding='utf8',
    )
    if outcome.returncode != 0:
        error_payload = parse_hf_error_payload(outcome.stdout)
        error_class = error_payload.get('error_class') if isinstance(error_payload.get('error_class'), str) else 'CalledProcessError'
        error_message = error_payload.get('error_message') if isinstance(error_payload.get('error_message'), str) else (outcome.stderr.strip() or outcome.stdout.strip() or 'HF child process failed')
        status_code = error_payload.get('status_code') if isinstance(error_payload.get('status_code'), int) else None
        return {
            'ok': False,
            'error_class': error_class,
            'error_message': error_message,
            'status_code': status_code,
            'failure_classification': classify_hf_weight_failure(error_class, status_code),
        }
    return {'ok': target_path.is_file()}


def acquire_required_weight(ext_dir: str, payload: dict[str, object], venv_dir: Path) -> dict[str, object]:
    install_root = Path(ext_dir)
    target_path = install_root / WEIGHT_RELATIVE_PATH
    repo_root = Path(__file__).resolve().parent
    attempted_descriptors = [build_weight_source_descriptor('ext-dir', target_path)]

    def build_result(acquired: bool, resolved_descriptor: dict[str, object] | None) -> dict[str, object]:
        attempted_sources = [descriptor['source'] for descriptor in attempted_descriptors]
        attempted_source_kinds = [descriptor['kind'] for descriptor in attempted_descriptors]
        resolved_source_kind = resolved_descriptor['kind'] if resolved_descriptor else None
        resolved_source = resolved_descriptor['source'] if resolved_descriptor else None
        return {
            'acquired': acquired,
            'target_path': str(target_path),
            'attempted_source_kinds': attempted_source_kinds,
            'attempted_sources': attempted_sources,
            'resolved_source_kind': resolved_source_kind,
            'resolved_source': resolved_source,
            'weight_source_repo_id': resolved_descriptor.get('repo_id') if resolved_descriptor else None,
            'weight_source_filename': resolved_descriptor.get('filename', WEIGHT_FILENAME) if resolved_descriptor else WEIGHT_FILENAME,
            'weight_source_revision': resolved_descriptor.get('revision') if resolved_descriptor else None,
            'weight_source_auth_used': resolved_descriptor.get('auth_used') if resolved_descriptor else None,
            'failure_classification': None,
            'error_class': None,
            'error_message': None,
        }

    def build_failure_result(
        descriptor: dict[str, object] | None = None,
        *,
        failure_classification: str,
        error_class: str,
        error_message: str,
    ) -> dict[str, object]:
        result = build_result(False, None)
        if descriptor is not None:
            result['weight_source_repo_id'] = descriptor.get('repo_id')
            result['weight_source_filename'] = descriptor.get('filename', WEIGHT_FILENAME)
            result['weight_source_revision'] = descriptor.get('revision')
            result['weight_source_auth_used'] = descriptor.get('auth_used')
        result['failure_classification'] = failure_classification
        result['error_class'] = error_class
        result['error_message'] = error_message
        return result

    if target_path.is_file():
        return build_result(True, attempted_descriptors[0])

    local_candidates = []
    required_weight_path = payload.get('required_weight_path')
    if isinstance(required_weight_path, str) and required_weight_path.strip():
        local_candidates.append(build_weight_source_descriptor('required_weight_path', Path(required_weight_path.strip())))

    env_source = os.environ.get('ULTRASHAPE_WEIGHT_SOURCE_PATH')
    if env_source:
        local_candidates.append(build_weight_source_descriptor('env-local', Path(env_source)))

    repo_weight = repo_root / WEIGHT_RELATIVE_PATH
    local_candidates.append(build_weight_source_descriptor('repo-local', repo_weight))

    for candidate in local_candidates:
        attempted_descriptors.append(candidate)
        try:
            if copy_required_weight(Path(candidate['source']), target_path):
                return build_result(True, candidate)
        except Exception as error:
            return build_failure_result(
                candidate,
                failure_classification=classify_local_weight_failure(error),
                error_class=error.__class__.__name__,
                error_message=str(error),
            )

    hf_source = resolve_hf_source(payload)
    if hf_source is not None:
        attempted_descriptors.append(build_weight_source_descriptor(hf_source['kind'], hf_source['source']))
        attempted_descriptors[-1].update({
            'repo_id': hf_source['repo_id'],
            'filename': hf_source['filename'],
            'revision': hf_source['revision'],
            'auth_used': hf_source['auth_used'],
        })
        download_cache = install_root / '.hf-cache'
        hf_result = download_hf_weight(venv_dir, target_path, hf_source, download_cache)
        if hf_result['ok']:
            return build_result(True, attempted_descriptors[-1])
        return build_failure_result(
            attempted_descriptors[-1],
            failure_classification=str(hf_result['failure_classification']),
            error_class=str(hf_result['error_class']),
            error_message=str(hf_result['error_message']),
        )

    return build_result(False, None)


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


def build_readiness(ext_dir: str, health: dict[str, object], weight_result: dict[str, object]) -> dict[str, object]:
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
    }


def build_weight_diagnostics(weight_result: dict[str, object]) -> dict[str, object]:
    return {
        'attempted_weight_source_kinds': weight_result['attempted_source_kinds'],
        'attempted_weight_sources': weight_result['attempted_sources'],
        'resolved_weight_source_kind': weight_result['resolved_source_kind'],
        'resolved_weight_source': weight_result['resolved_source'],
        'weight_source_repo_id': weight_result['weight_source_repo_id'],
        'weight_source_filename': weight_result['weight_source_filename'],
        'weight_source_revision': weight_result['weight_source_revision'],
        'weight_source_auth_used': weight_result['weight_source_auth_used'],
        'weight_source_failure_classification': weight_result['failure_classification'],
        'weight_source_error_class': weight_result['error_class'],
        'weight_source_error_message': weight_result['error_message'],
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
        'missing_required': health['missing_required'],
        'missing_optional': health['missing_optional'],
    })
    summary.update(build_weight_diagnostics(weight_result))
    write_json(summary_path, summary)
    readiness = build_readiness(ext_dir, health, weight_result)
    readiness.update(build_weight_diagnostics(weight_result))
    write_json(readiness_path, readiness)

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
