"""UltraShape runtime package markers for the clean-room upstream closure."""

RUNTIME_SCOPE = 'mc-only'
RUNTIME_LAYOUT = 'vendored-upstream-closure'
CHECKPOINT_REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')
UPSTREAM_CLOSURE_READY = True
UPSTREAM_CLOSURE_REASON = 'The vendored UltraShape payload ships the clean-room upstream closure for local mc-only refinement.'

__all__ = [
    'CHECKPOINT_REQUIRED_SUBTREES',
    'RUNTIME_LAYOUT',
    'RUNTIME_SCOPE',
    'UPSTREAM_CLOSURE_READY',
    'UPSTREAM_CLOSURE_REASON',
]
