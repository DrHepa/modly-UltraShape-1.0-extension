"""Patched DiT mask seam with optional flash-attn fallback."""

try:
    import flash_attn  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on ARM64 MVP installs
    flash_attn = None


def flash_attn_available() -> bool:
    return flash_attn is not None
