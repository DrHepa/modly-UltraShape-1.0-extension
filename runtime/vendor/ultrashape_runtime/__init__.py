"""Curated UltraShape runtime subset for the local mc-only MVP."""

from .local_runner import run_refine_job
from .utils.checkpoint import expected_checkpoint_subtrees

RUNTIME_SCOPE = 'mc-only'
RUNTIME_LAYOUT = 'vendored-minimal'
CHECKPOINT_REQUIRED_SUBTREES = expected_checkpoint_subtrees()

__all__ = ['CHECKPOINT_REQUIRED_SUBTREES', 'RUNTIME_SCOPE', 'RUNTIME_LAYOUT', 'run_refine_job']
