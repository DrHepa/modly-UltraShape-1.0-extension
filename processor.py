#!/usr/bin/env python3
"""Stable shell processor for the clean-room UltraShape rewrite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

CONTRACT = {
    "backend": "local",
    "kind": "process-refiner",
    "required_inputs": ["reference_image", "coarse_mesh"],
    "output_artifact": "output_dir/refined.glb",
    "forbidden_execution_modes": ["remote", "hybrid", "model-wrapper"],
}

ALLOWED_PARAMS = {
    "backend",
    "steps",
    "guidance_scale",
    "seed",
    "preserve_scale",
    "output_format",
    "coarse_mesh",
}

PUBLIC_RUNTIME_ERROR_CODES = {
    "DEPENDENCY_MISSING",
    "WEIGHTS_MISSING",
    "LOCAL_RUNTIME_UNAVAILABLE",
}
DEFAULT_OUTPUT_DIR = "output_dir"
DEFAULT_OUTPUT_FORMAT = "glb"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_PRESERVE_SCALE = True


def emit(payload: dict[str, Any], exit_code: int) -> int:
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return exit_code


def read_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Request payload must be a JSON object.")
    return data


def invalid_input(message: str) -> int:
    return emit(
        {
            "ok": False,
            "error": {
                "code": "INVALID_INPUT",
                "message": message,
            },
        },
        1,
    )


def runtime_error(code: str, message: str) -> int:
    normalized = code if code in PUBLIC_RUNTIME_ERROR_CODES else "LOCAL_RUNTIME_UNAVAILABLE"
    return emit(
        {
            "ok": False,
            "error": {
                "code": normalized,
                "message": message,
            },
        },
        1,
    )


def _read_preferred_inputs(payload: dict[str, Any]) -> tuple[object, object]:
    return payload.get("reference_image"), payload.get("coarse_mesh")


def _read_fallback_inputs(payload: dict[str, Any], params: dict[str, Any]) -> tuple[object, object]:
    reference_image = None
    fallback_input = payload.get("input")
    if isinstance(fallback_input, dict):
        reference_image = fallback_input.get("filePath")

    return reference_image, params.get("coarse_mesh")


def normalize_request(payload: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    params = payload.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, "params must be a JSON object when provided."

    invalid_keys = sorted(set(params.keys()) - ALLOWED_PARAMS)
    if invalid_keys:
        return None, f"Unsupported params for the stable shell: {', '.join(invalid_keys)}."

    backend = params.get("backend")
    if backend not in (None, "local"):
        return None, "Only local execution is allowed in this clean-room shell."

    preferred_reference_image, preferred_coarse_mesh = _read_preferred_inputs(payload)
    has_preferred_reference_image = "reference_image" in payload
    has_preferred_coarse_mesh = "coarse_mesh" in payload

    if has_preferred_reference_image and has_preferred_coarse_mesh:
        reference_image, coarse_mesh = preferred_reference_image, preferred_coarse_mesh
    elif not has_preferred_reference_image and not has_preferred_coarse_mesh:
        reference_image, coarse_mesh = _read_fallback_inputs(payload, params)
    else:
        reference_image, coarse_mesh = preferred_reference_image, preferred_coarse_mesh

    if not isinstance(reference_image, str) or not reference_image.strip() or not isinstance(coarse_mesh, str) or not coarse_mesh.strip():
        return None, "reference_image and coarse_mesh are both required for process-refiner requests."

    return {
        "reference_image": reference_image,
        "coarse_mesh": coarse_mesh,
    }, None


def command_describe_contract() -> int:
    return emit(CONTRACT, 0)


def command_validate_request() -> int:
    try:
        payload = read_payload()
    except (ValueError, json.JSONDecodeError) as exc:
        return invalid_input(str(exc))

    normalized, error = normalize_request(payload)
    if error is not None:
        return invalid_input(error)

    return emit({"ok": True, "normalized_request": normalized}, 0)


def command_process(output_dir: str | None) -> int:
    try:
        payload = read_payload()
    except (ValueError, json.JSONDecodeError) as exc:
        return invalid_input(str(exc))

    normalized_request, error = normalize_request(payload)
    if error is not None:
        return invalid_input(error)

    readiness, readiness_error = load_runtime_readiness()
    if readiness is None:
        return runtime_error("LOCAL_RUNTIME_UNAVAILABLE", readiness_error or "Runtime readiness is unavailable.")

    preflight_error = evaluate_runtime_readiness(readiness)
    if preflight_error is not None:
        return runtime_error(*preflight_error)

    runner_job = build_runner_job(
        readiness=readiness,
        request=normalized_request,
        params=payload.get("params") if isinstance(payload.get("params"), dict) else {},
        output_dir=output_dir,
    )
    success, result_or_error = run_local_runner(runner_job)
    if success:
        return emit({"ok": True, "result": result_or_error}, 0)
    return runtime_error(result_or_error["code"], result_or_error["message"])


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def runtime_vendor_path() -> Path:
    return repo_root() / "runtime" / "vendor"


def runtime_config_path() -> str:
    return str(repo_root() / "runtime" / "configs" / "infer_dit_refine.yaml")


def runtime_readiness_path() -> Path:
    return repo_root() / ".runtime-readiness.json"


def load_runtime_readiness() -> tuple[dict[str, Any] | None, str | None]:
    path = runtime_readiness_path()
    if not path.is_file():
        return None, "Runtime readiness file is missing; run setup.py to stage the local runtime first."

    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return None, "Runtime readiness file is unreadable; rerun setup.py before processing requests."

    if not isinstance(payload, dict):
        return None, "Runtime readiness file must contain a JSON object."
    return payload, None


def evaluate_runtime_readiness(readiness: dict[str, Any]) -> tuple[str, str] | None:
    backend = readiness.get("backend")
    if backend != "local":
        return "LOCAL_RUNTIME_UNAVAILABLE", "Runtime readiness must declare backend=local for this extension."

    missing_required = readiness.get("missing_required")
    missing_items = [item for item in missing_required if isinstance(item, str)] if isinstance(missing_required, list) else []

    if readiness.get("required_imports_ok") is False:
        message = (
            f"Required runtime imports are unavailable: {', '.join(missing_items)}."
            if missing_items
            else "Required runtime imports are unavailable for the local runner."
        )
        return "DEPENDENCY_MISSING", message

    if readiness.get("weights_ready") is False:
        message = (
            f"Required runtime weights are unavailable: {', '.join(missing_items)}."
            if missing_items
            else "Required runtime weights are unavailable for the local runner."
        )
        return "WEIGHTS_MISSING", message

    if readiness.get("status") != "ready":
        return "LOCAL_RUNTIME_UNAVAILABLE", "Runtime readiness does not report status=ready for the local runner."

    return None


def build_runner_job(
    *,
    readiness: dict[str, Any],
    request: dict[str, str],
    params: dict[str, Any],
    output_dir: str | None,
) -> dict[str, Any]:
    ext_dir = readiness.get("ext_dir")
    if not isinstance(ext_dir, str) or not ext_dir.strip():
        ext_dir = str(repo_root())

    config_path = readiness.get("config_path")
    if not isinstance(config_path, str) or not config_path.strip():
        config_path = runtime_config_path()

    checkpoint = readiness.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        checkpoint = None

    return {
        "reference_image": request["reference_image"],
        "coarse_mesh": request["coarse_mesh"],
        "output_dir": output_dir or DEFAULT_OUTPUT_DIR,
        "output_format": str(params.get("output_format") or DEFAULT_OUTPUT_FORMAT),
        "checkpoint": checkpoint,
        "config_path": config_path,
        "ext_dir": ext_dir,
        "backend": "local",
        "steps": int(params.get("steps", DEFAULT_STEPS)),
        "guidance_scale": float(params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
        "seed": params.get("seed") if isinstance(params.get("seed"), int) or params.get("seed") is None else params.get("seed"),
        "preserve_scale": bool(params.get("preserve_scale", DEFAULT_PRESERVE_SCALE)),
    }


def run_local_runner(job: dict[str, Any]) -> tuple[bool, dict[str, Any] | dict[str, str]]:
    python_path_entries = [str(runtime_vendor_path())]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "ultrashape_runtime.local_runner"],
            cwd=str(repo_root()),
            input=json.dumps(job),
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(python_path_entries),
            },
            check=False,
        )
    except OSError as exc:
        return False, {"code": "LOCAL_RUNTIME_UNAVAILABLE", "message": str(exc)}

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return False, {
            "code": "LOCAL_RUNTIME_UNAVAILABLE",
            "message": "Local runner returned unreadable output.",
        }

    if completed.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("result"), dict):
        return True, payload["result"]

    if isinstance(payload, dict):
        code = payload.get("error_code")
        message = payload.get("error_message")
        if isinstance(code, str) and isinstance(message, str):
            return False, {"code": code, "message": message}

    return False, {
        "code": "LOCAL_RUNTIME_UNAVAILABLE",
        "message": (completed.stderr or "Local runner failed without a structured public error.").strip(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stable UltraShape shell processor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--describe-contract", action="store_true")
    group.add_argument("--validate-request", action="store_true")
    group.add_argument("--process", action="store_true")
    parser.add_argument("--output-dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.describe_contract:
        return command_describe_contract()
    if args.validate_request:
        return command_validate_request()
    return command_process(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
