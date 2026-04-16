"""Curated UltraShape runtime subset for the local mc-only MVP."""

from .local_runner import run_refine_job

RUNTIME_SCOPE = 'mc-only'
RUNTIME_LAYOUT = 'vendored-minimal'

__all__ = ['RUNTIME_SCOPE', 'RUNTIME_LAYOUT', 'run_refine_job']
