# UltraShape Hunyuan-aligned shell

This repository exposes a Hunyuan-aligned public shell over the existing UltraShape model extension.

The model shell is the sole public authority: `manifest.json`, `setup.py`, and `generator.py`.

`manifest.json` publishes the shell contract, `setup.py` is the install/bootstrap authority, and `generator.py` is the only public lifecycle authority.

`runtime/**` and `models/ultrashape/**` remain private implementation details.

The private runtime is an explicit dual-mode UltraShape seam: real mode targets the closest achievable upstream closure when the exact environment is available, and portable mode is the reduced-environment fallback.

`setup.py` stages that private runtime into the install root and writes truthful readiness artifacts for the generator lifecycle, including whether real mode is unavailable, portable fallback is active, or the runtime is blocked. `ready` means the staged config, vendored runtime, required imports, and required checkpoint are all actually present.

The public shell exposes only the supported local refinement contract and keeps legacy processor-era paths out of authority.
