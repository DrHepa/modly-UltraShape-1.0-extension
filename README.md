# UltraShape clean-room local-only rewrite

This repository is in clean-room, local-only rewrite mode.

Shell authority in this rewrite is limited to:
- `manifest.json`
- `setup.py`
- `processor.py`
- `.runtime-readiness.json`
- `.setup-summary.json`
- public stdout events
- `output_dir/refined.glb`

`setup.py` stages the vendored runtime closure into the install root and writes truthful readiness artifacts. `ready` means the staged config, vendored runtime, required imports, and required checkpoint are all actually present. The shell does NOT claim synthetic success.

## Temporary processor seam

`processor.py` still carries the only temporary fallback seam for Modly compatibility:
- preferred truth: `reference_image` + `coarse_mesh`
- temporary fallback: `input.filePath` + `params.coarse_mesh`

Fallback names are a temporary compatibility seam inside `processor.py` and are not shell authority anywhere else.

That seam stays confined to `processor.py` until Modly ships native two-input routing.

## Batch 1 non-goals

- Do not recreate `src/`.
- Do not restore fallback fixture bundles.
- Do not restore patch-authority directories.
