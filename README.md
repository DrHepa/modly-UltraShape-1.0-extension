# modly-UltraShape-1.0-extension

Pure process extension for UltraShape mesh refinement.

## Installable Modly contract

The active install surface is the repo-root Python boundary only:

- `manifest.json`
- `setup.py`
- `processor.py`
- `README.md`
- `runtime/configs/infer_dit_refine.yaml`
- `runtime/vendor/ultrashape_runtime/**`
- `runtime/patches/README.md`

Do NOT curate a separate CommonJS payload. GitHub install validation must target the extracted repo root exactly as Modly receives it.

This repo stays on the root `manifest.json` + `setup.py` + `processor.py` contract.

## Linux ARM64 local-first runtime

This MVP is LOCAL-FIRST and LOCAL-ONLY for execution.

- Host target: Linux ARM64
- GPU target: CUDA 12.8-class / SM 90+
- Torch profile: `torch==2.7.0+cu128` + `torchvision==0.22.0`
- Runtime scope: `mc-only`
- Active runtime backend: `local`

`setup.py` parses Modly JSON args (`python_exe`, `ext_dir`, `gpu_sm`, optional `cuda_version`), creates `ext_dir/venv`, installs the real-refinement MVP dependency surface into that venv, copies the vendored runtime into `ext_dir/runtime/ultrashape_runtime/**`, stages `ext_dir/runtime/configs/infer_dit_refine.yaml`, acquires `ext_dir/models/ultrashape/ultrashape_v1.pt` from a provided local source or configured Hugging Face source, runs import smoke across required, conditional, and degradable tiers, and writes:

- `ext_dir/.setup-summary.json`
- `ext_dir/.runtime-readiness.json`

`processor.py` must trust `.runtime-readiness.json` as the authoritative install/runtime signal. It does NOT reintroduce remote-first fallback behavior.

## Readiness states

`.runtime-readiness.json` is the source of truth for smoke verification and runtime behavior. It documents whether the installed runtime can execute the checkpoint-backed real-refinement closure truthfully.

- `ready` — required weights and the supported real-refinement dependency tier passed import smoke; the staged contract is eligible for the local-only / mc-only / glb-only path
- `degraded` — install succeeded, but only CONDITIONAL or DEGRADABLE gaps remain; readiness must record those gaps explicitly instead of pretending the runtime closure is fully available
- `blocked` — runtime cannot operate locally; processor must emit `LOCAL_RUNTIME_UNAVAILABLE`

Required readiness fields:

- `status`
- `backend`
- `mvp_scope`
- `weights_ready`
- `required_imports_ok`
- `missing_required[]`
- `missing_optional[]`
- `missing_conditional[]`
- `missing_degradable[]`
- `expected_weights[]`

## Public runtime outcomes

Allowed stdout events:

- `{"type":"progress","percent":number,"label":string}`
- `{"type":"log","message":string}`
- `{"type":"done","result":{"filePath":"/abs/path/refined.glb"}}`
- `{"type":"error","message":string,"code":"..."}`

Public runtime error codes are limited to:

- `INVALID_PARAMS`
- `MISSING_INPUT`
- `UNREADABLE_ASSET`
- `UNSUPPORTED_ASSET_TYPE`
- `DEPENDENCY_MISSING`
- `WEIGHTS_MISSING`
- `LOCAL_RUNTIME_UNAVAILABLE`

Smoke validation should accept ONLY:

1. `done` when readiness is truly `ready`, or
2. an explicit readiness-driven public error (`WEIGHTS_MISSING`, `DEPENDENCY_MISSING`, or `LOCAL_RUNTIME_UNAVAILABLE`).

`BACKEND_UNAVAILABLE` is obsolete for this MVP and must not be treated as the normal install/runtime outcome.

## Dependency policy

Required dependency tier for the supported real-refinement path:

- `torch==2.7.0+cu128`
- `torchvision==0.22.0`
- `numpy`
- `trimesh`
- `Pillow`
- `opencv-python-headless`
- `scikit-image`
- `PyYAML`
- `omegaconf`
- `einops`
- `transformers`
- `huggingface_hub`
- `accelerate`
- `cubvh`
- `safetensors`
- `tqdm`

Conditional dependencies (only skippable when the reference image is already cut out / alpha-ready):

- `rembg`
- `onnxruntime`

Degradable dependencies:

- `flash_attn`

Missing conditional or degradable dependencies must be recorded explicitly in readiness; they do not justify remote fallback, and missing `cubvh` is NOT degradable for the supported path.

## Weights policy

Required MVP weights are expected at:

- `ext_dir/models/ultrashape/ultrashape_v1.pt`

Missing required weights are a FATAL install failure. `setup.py` must copy or download `ultrashape_v1.pt` during install, write truthful failure metadata, and exit non-zero instead of reporting a successful `degraded` state.

## Input contract

`processor.py` reads exactly one JSON object line from stdin.

Fallback fixture bundle fields remain documented as: `reference_image`, `coarse_mesh`, `output_dir`, `checkpoint`, and `params`.

Input resolution order:

1. Preferred named inputs:
   - `payload.input.inputs.reference_image.filePath`
   - `payload.input.inputs.coarse_mesh.filePath`
2. Temporary compatibility seam:
    - `payload.input.filePath` becomes `reference_image`
    - `payload.params.coarse_mesh` provides `coarse_mesh`

The temporary `params.coarse_mesh` seam exists only until Modly exposes native multi-input routing for both semantic ids.

## Extension identity

- Repository/project: `modly-UltraShape-1.0-extension`
- Extension id: `modly.ultrashape-refiner-process`
- Node id: `ultrashape-refiner`
- Semantic contract: `reference_image + coarse_mesh -> refined_mesh`

This extension is a refiner, NOT a coarse-mesh generator or single-image generation wrapper.

## Coarse-mesh policy

Hunyuan is the recommended and currently validated upstream source, but it is NOT required. Any coarse mesh that satisfies the accepted mesh-format validation may be used.

## Fixture layout

`fixtures/requests/refiner-bundle/`

- `request.json` — compatibility-seam documentation fixture for the local-only smoke path
- `assets/reference-image.png` — install-smoke reference image fixture
- `assets/coarse-mesh.glb` — binary GLB coarse-mesh fixture used by install smoke
- `expected/output/refined-mesh.glb` — packaged binary GLB baseline kept in the copied payload for smoke comparisons

## Future Modly seam

The following items belong to Modly core, not this extension repo:

- native named multi-input routing for `reference_image` and `coarse_mesh`
- workflow plumbing that supplies both inputs without the temporary seam
- Generate panel `Type mismatch` follow-up

When Modly supports the native seam, this extension should drop the temporary compatibility seam without renaming the semantic contract.

## Smoke-test expectation

The faithful smoke from this repo is:

1. validate the extracted repo-root install surface,
2. run `setup.py` with Modly-style JSON args,
3. read `.runtime-readiness.json` as authoritative evidence,
4. invoke `processor.py` with valid fixture assets,
5. accept either `done` for local-ready installs or the explicit readiness-driven public error.
