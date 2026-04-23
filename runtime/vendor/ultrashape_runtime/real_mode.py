"""Authoritative real-mode seam for the vendored UltraShape runtime."""

from __future__ import annotations

REAL_MODE_ADAPTER = 'ultrashape_runtime.real_mode.run_real_refine_pipeline'
REAL_MODE_BLOCKER = 'authoritative upstream torch module graph adapter is not vendored yet.'


class RealModeUnavailableError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def describe_real_mode() -> dict[str, object]:
    return {
        'available': False,
        'adapter': REAL_MODE_ADAPTER,
        'reason': (
            'Authoritative real mode is optional and remains unavailable until the exact upstream '
            'torch module graph adapter is vendored and the required runtime dependencies are present.'
        ),
        'blockers': ['adapter:authoritative-upstream-module-graph'],
        'entrypoint': 'scripts.infer_dit_refine.run_inference',
    }


def run_real_refine_pipeline(**_: object) -> dict[str, object]:
    raise RealModeUnavailableError(f'Real runtime mode requested but unavailable: {REAL_MODE_BLOCKER}')
