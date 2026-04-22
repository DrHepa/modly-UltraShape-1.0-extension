#!/usr/bin/env python3
"""Minimal UltraShape generator lifecycle shell."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Any

from services.generators.base import BaseGenerator

PUBLIC_RUNTIME_ERROR_CODES = {
    "DEPENDENCY_MISSING",
    "WEIGHTS_MISSING",
    "LOCAL_RUNTIME_UNAVAILABLE",
    "INVALID_INPUT",
}

PUBLIC_GENERATE_PARAM_DEFAULTS: dict[str, Any] = {
    "steps": 30,
    "guidance_scale": 7.5,
    "seed": None,
    "preserve_scale": True,
}

NON_OPERATIVE_MODLY_GENERATE_PARAM_KEYS = {"remesh", "enable_texture", "texture_resolution"}
PUBLIC_GENERATE_ALLOWED_PARAM_KEYS = {"mesh_path", *PUBLIC_GENERATE_PARAM_DEFAULTS.keys()}
LEGACY_GENERATE_ALIAS_KEYS = {"reference_image", "coarse_mesh", "input", "input.filePath"}


class PublicRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        normalized = code if code in PUBLIC_RUNTIME_ERROR_CODES else "LOCAL_RUNTIME_UNAVAILABLE"
        super().__init__(message)
        self.code = normalized


class UltraShapeGenerator(BaseGenerator):
    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(model_dir, outputs_dir)
        self._last_job: dict[str, Any] | None = None
        self._last_pythonpath: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._runtime_readiness: dict[str, Any] | None = None

    def is_downloaded(self) -> bool:
        readiness = self._load_runtime_readiness(allow_missing=True)
        checkpoint_ready = self._checkpoint_path().is_file()
        config_ready = self._config_path().is_file()
        vendor_ready = self._vendor_path().is_dir()

        if not (checkpoint_ready and config_ready and vendor_ready):
            return False
        if readiness is None:
            return False
        return bool(readiness.get("required_imports_ok")) and bool(readiness.get("weights_ready"))

    def load(self) -> bool:
        readiness = self._require_runtime_ready()
        self._runtime_readiness = readiness
        self._loaded = True
        return self._loaded

    def generate(
        self,
        image_bytes: bytes | None,
        params: dict[str, Any] | None = None,
        progress_cb: Any | None = None,
        cancel_event: Any | None = None,
    ) -> str:
        del progress_cb, cancel_event
        if not self._loaded:
            self.load()

        normalized_params = self._normalize_generate_params(params)
        mesh_path = normalized_params["mesh_path"]
        if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
            raise PublicRuntimeError("INVALID_INPUT", "image_bytes must be a non-empty bytes payload.")

        source_mesh = self._resolve_mesh_path(mesh_path)

        runtime_readiness = self._runtime_readiness or self._require_runtime_ready()
        output_dir = Path(tempfile.mkdtemp(prefix="ultrashape-generator-"))
        reference_dir = Path(tempfile.mkdtemp(prefix="ultrashape-reference-"))
        reference_path = reference_dir / "reference.png"
        reference_path.write_bytes(bytes(image_bytes))

        runner_job = self._build_runner_job(
            readiness=runtime_readiness,
            reference_image=reference_path,
            coarse_mesh=source_mesh,
            output_dir=output_dir,
            params=normalized_params,
        )
        runner_result = self._run_local_runner(runner_job)
        self._last_job = {
            "backend": runner_job["backend"],
            "output_format": runner_job["output_format"],
            "config_path": runner_job["config_path"],
            "checkpoint": runner_job["checkpoint"],
        }
        self._last_result = runner_result
        return str(runner_result["file_path"])

    def _normalize_generate_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            normalized_params: dict[str, Any] = {}
        elif isinstance(params, dict):
            normalized_params = dict(params)
        else:
            raise PublicRuntimeError("INVALID_INPUT", "params must be a JSON object when provided.")

        for ignored_key in NON_OPERATIVE_MODLY_GENERATE_PARAM_KEYS:
            normalized_params.pop(ignored_key, None)

        alias_fields = sorted(key for key in LEGACY_GENERATE_ALIAS_KEYS if key in normalized_params)
        if alias_fields:
            if any(key in normalized_params for key in PUBLIC_GENERATE_ALLOWED_PARAM_KEYS):
                raise PublicRuntimeError(
                    "INVALID_INPUT",
                    f"Mixed public contract fields with legacy alias fields are not supported: {', '.join(alias_fields)}.",
                )
            raise PublicRuntimeError(
                "INVALID_INPUT",
                f"params contains legacy alias fields that are not supported: {', '.join(alias_fields)}. Use image_bytes plus params.mesh_path and public params only.",
            )

        unexpected_fields = sorted(key for key in normalized_params if key not in PUBLIC_GENERATE_ALLOWED_PARAM_KEYS)
        if unexpected_fields:
            raise PublicRuntimeError(
                "INVALID_INPUT",
                f"Unsupported params fields: {', '.join(unexpected_fields)}.",
            )

        mesh_path = normalized_params.get("mesh_path")
        if not isinstance(mesh_path, str) or not mesh_path.strip():
            raise PublicRuntimeError("INVALID_INPUT", "params.mesh_path is required for mesh refinement.")

        steps = normalized_params.get("steps", PUBLIC_GENERATE_PARAM_DEFAULTS["steps"])
        if not isinstance(steps, int) or steps <= 0:
            raise PublicRuntimeError("INVALID_INPUT", "params.steps must be a positive integer.")

        guidance_scale = normalized_params.get("guidance_scale", PUBLIC_GENERATE_PARAM_DEFAULTS["guidance_scale"])
        if not isinstance(guidance_scale, (int, float)) or float(guidance_scale) <= 0:
            raise PublicRuntimeError("INVALID_INPUT", "params.guidance_scale must be a positive number.")

        seed = normalized_params.get("seed", PUBLIC_GENERATE_PARAM_DEFAULTS["seed"])
        if seed is not None and not isinstance(seed, int):
            raise PublicRuntimeError("INVALID_INPUT", "params.seed must be an integer or null.")

        preserve_scale = normalized_params.get("preserve_scale", PUBLIC_GENERATE_PARAM_DEFAULTS["preserve_scale"])
        if not isinstance(preserve_scale, bool):
            raise PublicRuntimeError("INVALID_INPUT", "params.preserve_scale must be a boolean.")

        return {
            "mesh_path": mesh_path.strip(),
            "steps": steps,
            "guidance_scale": float(guidance_scale),
            "seed": seed,
            "preserve_scale": preserve_scale,
        }

    def _resolve_mesh_path(self, mesh_path: str) -> Path:
        candidate = Path(mesh_path)
        candidates: list[Path] = []

        def append_candidate(path: Path) -> None:
            if path not in candidates:
                candidates.append(path)

        if candidate.is_absolute():
            append_candidate(candidate)
        else:
            outputs_dir = Path(self.outputs_dir)
            append_candidate(outputs_dir / candidate)
            append_candidate(outputs_dir.parent / candidate)

            workspace_dir = os.environ.get("WORKSPACE_DIR")
            if isinstance(workspace_dir, str) and workspace_dir.strip():
                append_candidate(Path(workspace_dir.strip()) / candidate)

        for resolved_candidate in candidates:
            if resolved_candidate.exists():
                return resolved_candidate

        raise PublicRuntimeError("INVALID_INPUT", self._format_missing_mesh_diagnostics(mesh_path, candidates))

    def _format_missing_mesh_diagnostics(self, mesh_path: str, candidates: list[Path]) -> str:
        workspace_dir = os.environ.get("WORKSPACE_DIR")
        diagnostics = [
            "Mesh input could not be resolved.",
            f"original mesh_path: {mesh_path}",
            f"self.outputs_dir: {Path(self.outputs_dir)}",
            f"WORKSPACE_DIR: {workspace_dir if isinstance(workspace_dir, str) and workspace_dir.strip() else '<unset>'}",
            "candidates:",
        ]

        for index, candidate in enumerate(candidates, start=1):
            parent = candidate.parent
            diagnostics.append(
                f"  {index}. path={candidate} exists={candidate.exists()} parent={parent} parent_exists={parent.exists()}"
            )

        return "\n".join(diagnostics)

    def unload(self) -> bool:
        self._last_job = None
        self._last_pythonpath = None
        self._last_result = None
        self._runtime_readiness = None
        self._loaded = False
        return self._loaded

    def _require_runtime_ready(self) -> dict[str, Any]:
        readiness = self._load_runtime_readiness(allow_missing=False)

        missing_required = readiness.get("missing_required")
        missing_items = [item for item in missing_required if isinstance(item, str)] if isinstance(missing_required, list) else []

        if readiness.get("required_imports_ok") is False:
            message = (
                f"Required runtime imports are unavailable: {', '.join(missing_items)}."
                if missing_items
                else "Required runtime imports are unavailable for the local generator."
            )
            raise PublicRuntimeError("DEPENDENCY_MISSING", message)

        if readiness.get("weights_ready") is False:
            message = (
                f"Required runtime weights are unavailable: {', '.join(missing_items)}."
                if missing_items
                else "Required runtime weights are unavailable for the local generator."
            )
            raise PublicRuntimeError("WEIGHTS_MISSING", message)

        if readiness.get("status") not in {"ready", "degraded"}:
            raise PublicRuntimeError(
                "LOCAL_RUNTIME_UNAVAILABLE",
                "Runtime readiness does not report a usable local generator state.",
            )

        return readiness

    def _load_runtime_readiness(self, *, allow_missing: bool) -> dict[str, Any] | None:
        path = self._readiness_path()
        if not path.is_file():
            if allow_missing:
                return None
            raise PublicRuntimeError(
                "LOCAL_RUNTIME_UNAVAILABLE",
                "Runtime readiness file is missing; run setup.py before using the generator.",
            )

        try:
            payload = json.loads(path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicRuntimeError(
                "LOCAL_RUNTIME_UNAVAILABLE",
                f"Runtime readiness file is unreadable: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise PublicRuntimeError(
                "LOCAL_RUNTIME_UNAVAILABLE",
                "Runtime readiness file must contain a JSON object.",
            )
        return payload

    def _build_runner_job(
        self,
        *,
        readiness: dict[str, Any],
        reference_image: Path,
        coarse_mesh: Path,
        output_dir: Path,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        checkpoint = readiness.get("checkpoint")
        config_path = readiness.get("config_path")
        ext_dir = readiness.get("ext_dir")

        return {
            "reference_image": str(reference_image),
            "coarse_mesh": str(coarse_mesh),
            "output_dir": str(output_dir),
            "output_format": "glb",
            "checkpoint": checkpoint if isinstance(checkpoint, str) else None,
            "config_path": str(config_path) if isinstance(config_path, str) and config_path.strip() else str(self._config_path()),
            "ext_dir": str(ext_dir) if isinstance(ext_dir, str) and ext_dir.strip() else str(self._repo_root()),
            "backend": "local",
            "steps": params["steps"],
            "guidance_scale": params["guidance_scale"],
            "seed": params["seed"],
            "preserve_scale": params["preserve_scale"],
        }

    def _run_local_runner(self, job: dict[str, Any]) -> dict[str, Any]:
        python_path_entries = [self._runtime_vendor_parent()]
        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            python_path_entries.append(existing_pythonpath)
        self._last_pythonpath = os.pathsep.join(python_path_entries)

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "ultrashape_runtime.local_runner"],
                cwd=str(self._repo_root()),
                input=json.dumps(job),
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PYTHONPATH": self._last_pythonpath,
                },
                check=False,
            )
        except OSError as exc:
            raise PublicRuntimeError("LOCAL_RUNTIME_UNAVAILABLE", str(exc)) from exc

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise PublicRuntimeError("LOCAL_RUNTIME_UNAVAILABLE", "Local runner returned unreadable output.") from exc

        if completed.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("result"), dict):
            return payload["result"]

        if isinstance(payload, dict):
            error_code = payload.get("error_code")
            error_message = payload.get("error_message")
            if isinstance(error_code, str) and isinstance(error_message, str):
                raise PublicRuntimeError(error_code, error_message)

        message = (completed.stderr or "Local runner failed without a structured public error.").strip()
        raise PublicRuntimeError("LOCAL_RUNTIME_UNAVAILABLE", message)

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent

    def _readiness_path(self) -> Path:
        return self._repo_root() / ".runtime-readiness.json"

    def _checkpoint_path(self) -> Path:
        return self._repo_root() / "models" / "ultrashape" / "ultrashape_v1.pt"

    def _config_path(self) -> Path:
        return self._repo_root() / "runtime" / "configs" / "infer_dit_refine.yaml"

    def _vendor_path(self) -> Path:
        return self._repo_root() / "runtime" / "vendor" / "ultrashape_runtime"

    def _runtime_vendor_parent(self) -> str:
        if self._runtime_readiness is not None:
            vendor_path = self._runtime_readiness.get("vendor_path")
            if isinstance(vendor_path, str) and vendor_path.strip():
                candidate = Path(vendor_path)
                return str(candidate.parent if candidate.name == 'ultrashape_runtime' else candidate)
        return str(self._vendor_path().parent)
