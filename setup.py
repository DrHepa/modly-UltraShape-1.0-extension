import json
import os
import sys
from pathlib import Path
from venv import EnvBuilder


REQUIRED_STRING_KEYS = ('python_exe', 'ext_dir')


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


def write_summary(ext_dir: str, context: dict[str, str]) -> None:
    summary_path = Path(ext_dir) / '.setup-summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'python_exe': context['python_exe'],
                'ext_dir': context['ext_dir'],
                'gpu_sm': context['gpu_sm'],
                'cuda_version': context.get('cuda_version'),
                'dependencies': [],
            },
            indent=2,
        )
        + '\n',
        encoding='utf8',
    )


def main() -> int:
    payload = parse_args(sys.argv)
    context = resolve_install_context(payload)
    ensure_venv(context['ext_dir'])
    write_summary(context['ext_dir'], context)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
