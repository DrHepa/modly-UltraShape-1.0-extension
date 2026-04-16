"""Utility exports for the vendored UltraShape runtime subset."""

from .checkpoint import expected_checkpoint_name
from .mesh import default_surface_algorithm
from .tensors import prefer_sdpa_attention

__all__ = [
    'default_surface_algorithm',
    'expected_checkpoint_name',
    'prefer_sdpa_attention',
]
