# UltraShape Hunyuan-aligned shell

This repository exposes a Hunyuan-aligned public shell over the existing UltraShape model extension.

The model shell is the sole public authority: `manifest.json`, `setup.py`, and `generator.py`.

`manifest.json` publishes the shell contract, `setup.py` is the install/bootstrap authority, and `generator.py` is the only public lifecycle authority.

`runtime/**` and `models/ultrashape/**` remain private implementation details.

`setup.py` stages the vendored runtime closure into the install root and writes truthful readiness artifacts for the generator lifecycle. `ready` means the staged config, vendored runtime, required imports, and required checkpoint are all actually present. The shell does NOT claim synthetic success.

No legacy process entrypoint or fallback alias remains public.

## Batch 1 non-goals

- Do not recreate `src/`.
- Do not restore fallback fixture bundles.
- Do not restore patch-authority directories.
