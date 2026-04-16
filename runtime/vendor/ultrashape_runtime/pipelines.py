"""Minimal pipeline contracts for the vendored UltraShape runtime."""


def build_refine_pipeline() -> dict[str, str]:
    return {'name': 'ultrashape-refine', 'scope': 'mc-only'}
