"""UltraShape runtime package markers for the local-only rewrite."""

RUNTIME_SCOPE = 'mc-only'
RUNTIME_LAYOUT = 'vendored-minimal'
CHECKPOINT_REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')
UPSTREAM_CLOSURE_READY = True
UPSTREAM_CLOSURE_REASON = 'The vendored local-only package now ships the ported upstream MVP closure for the supported mc-only -> refined.glb flow.'

__all__ = [
    'CHECKPOINT_REQUIRED_SUBTREES',
    'RUNTIME_LAYOUT',
    'RUNTIME_SCOPE',
    'UPSTREAM_CLOSURE_READY',
    'UPSTREAM_CLOSURE_REASON',
]
