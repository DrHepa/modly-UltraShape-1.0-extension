# UltraShape runtime patch staging

This directory documents the install-time patch intent for the local-only UltraShape MVP.

- `runtime/vendor/ultrashape_runtime/models/denoisers/dit_mask.py` keeps `flash_attn` optional and degrades to the PyTorch SDPA path.
- `runtime/vendor/ultrashape_runtime/models/autoencoders/surface_extractors.py` prefers `cubvh` when available and otherwise declares the `skimage.measure.marching_cubes` fallback.
- `diffusers` and `diso` remain explicitly optional in readiness metadata and are NOT required for the `mc`-only MVP.
- The vendored subset is intentionally the minimum inference closure needed for Batch 2; local execution wiring still belongs to the later `processor.py` batch.
