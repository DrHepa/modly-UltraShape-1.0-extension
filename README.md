# UltraShape clean-room local-only rewrite

This repository is in clean-room, local-only rewrite mode.

The ONLY public shell authority in this rewrite is:
- `manifest.json`
- `generator.py`
- `setup.py`

`manifest.json` now declares a truthful local model extension for `image + mesh -> mesh` refinement. The shell does not publish fallback aliases, mixed partial payloads, or process-entry authority.

Private runtime implementation details remain staged behind the shell. They support the local model lifecycle, but they are not the public contract.

`setup.py` stages the vendored runtime closure into the install root and writes truthful readiness artifacts for the generator lifecycle. `ready` means the staged config, vendored runtime, required imports, and required checkpoint are all actually present. The shell does NOT claim synthetic success.

Legacy process-shell authority has been removed. No public artifact re-authorizes any process-era entrypoint or fallback alias.

## Batch 1 non-goals

- Do not recreate `src/`.
- Do not restore fallback fixture bundles.
- Do not restore patch-authority directories.
