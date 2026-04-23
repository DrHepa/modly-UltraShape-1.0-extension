"""UltraShape runtime package markers for the explicit dual-mode closure."""

RUNTIME_SCOPE = 'mc-only'
RUNTIME_LAYOUT = 'vendored-dual-mode-closure'
RUNTIME_MODE_STRATEGY = 'explicit-dual-mode'
SUPPORTED_RUNTIME_MODES = ('auto', 'real', 'portable')
DEFAULT_RUNTIME_MODE = 'auto'
CHECKPOINT_REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')
UPSTREAM_CLOSURE_READY = False
UPSTREAM_CLOSURE_REASON = (
    'Authoritative real mode is optional and remains unavailable until the exact upstream '
    'torch module graph adapter is vendored and the required runtime dependencies are present.'
)

__all__ = [
    'CHECKPOINT_REQUIRED_SUBTREES',
    'DEFAULT_RUNTIME_MODE',
    'RUNTIME_LAYOUT',
    'RUNTIME_MODE_STRATEGY',
    'RUNTIME_SCOPE',
    'SUPPORTED_RUNTIME_MODES',
    'UPSTREAM_CLOSURE_READY',
    'UPSTREAM_CLOSURE_REASON',
]
