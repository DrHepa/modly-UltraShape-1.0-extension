# modly-UltraShape-1.0-extension

Pure process extension for UltraShape mesh refinement.

## Installable Modly contract

The active install surface is the repo root Python boundary only:

- root `manifest.json` + `setup.py` + `processor.py`
- root `README.md`

Do NOT curate a separate CommonJS payload. GitHub install validation should target the extracted repo root exactly as Modly receives it.

### GitHub install expectation

1. Modly extracts the repository root into its extensions directory.
2. Modly discovers `manifest.json` and sees `entry: "processor.py"`.
3. Modly invokes `python setup.py '{...}'` to create `ext_dir/venv`.
4. Modly runs `processor.py`, sending one JSON object line on stdin.

Discovery success MUST NOT depend on any legacy CommonJS mirror or curated copy.

### Processor protocol

`processor.py` reads exactly one JSON object line from stdin and writes JSON lines to stdout.

Allowed stdout events:

- `{"type":"progress","percent":number,"label":string}`
- `{"type":"log","message":string}`
- `{"type":"done","result":{"filePath":"/abs/path/refined.glb"}}`
- `{"type":"error","message":string,"code":"..."}`

Input resolution order:

1. Prefer named inputs:
   - `payload.input.inputs.reference_image.filePath`
   - `payload.input.inputs.coarse_mesh.filePath`
2. Temporary fallback seam:
   - `payload.input.filePath` becomes `reference_image`
   - `payload.params.coarse_mesh` provides the second input

The temporary `params.coarse_mesh` seam exists ONLY until Modly exposes native multi-input process routing for both semantic ids.

### Setup responsibilities

`setup.py` is intentionally minimal:

- parse Modly JSON args (`python_exe`, `ext_dir`, `gpu_sm`, optional `cuda_version`)
- create `ext_dir/venv`
- remain idempotent on re-install
- avoid heavyweight model/bootstrap side effects

Because the Python boundary currently uses only stdlib facilities, the minimal dependency set is empty.

## Extension identity

- Repository/project: `modly-UltraShape-1.0-extension`
- Extension id: `modly.ultrashape-refiner-process`
- Node id: `ultrashape-refiner`
- Semantic contract: `reference_image + coarse_mesh -> refined_mesh`

This extension is a refiner, NOT a coarse-mesh generator or single-image generation wrapper.

## Runtime stance

- Remote/hybrid-first execution remains the stance.
- Linux ARM64 is treated as a strong reason to prefer remote or hybrid execution.
- Missing or unsupported backend configuration must surface explicit `BACKEND_UNAVAILABLE`.
- Discovery + `BACKEND_UNAVAILABLE` is an acceptable smoke outcome when external runtime services are unavailable.

## Coarse-mesh policy

Hunyuan is the recommended and currently validated upstream source, but it is NOT required. Any coarse mesh that satisfies the accepted mesh-format validation may be used.

## Temporary fallback usage

For repo-local validation, `fixtures/requests/refiner-bundle/request.json` keeps the semantic contract visible while Modly still lacks native multi-input routing.

This fixture documents the temporary `params.coarse_mesh` seam. It is not the install surface.

Fallback bundle fields:

- `reference_image`
- `coarse_mesh`
- `output_dir`
- `checkpoint`
- `params`

Example:

```json
{
  "reference_image": "./assets/reference-image.png",
  "coarse_mesh": "./assets/coarse-mesh.glb",
  "output_dir": "./expected/output",
  "checkpoint": null,
  "params": {
    "backend": "remote",
    "steps": 30,
    "guidance_scale": 5.5,
    "seed": null,
    "preserve_scale": true,
    "output_format": "glb"
  }
}
```

## Fixture layout

`fixtures/requests/refiner-bundle/`

- `request.json` — fallback documentation fixture
- `assets/reference-image.png` — placeholder reference image asset
- `assets/coarse-mesh.glb` — placeholder coarse mesh asset
- `expected/output/refined-mesh.glb` — placeholder refined mesh output layout

## Future Modly seam

The following follow-ups belong to Modly core, not this extension repo:

- native named multi-input routing for `reference_image` and `coarse_mesh`
- workflow plumbing that supplies both inputs without the temporary seam
- Generate panel `Type mismatch` follow-up

When Modly supports the native seam, this extension should drop the temporary fallback without renaming the semantic contract.

## Smoke-test expectation

The best faithful smoke from this repo is:

1. validate the extracted repo-root install surface,
2. run `setup.py` with Modly-style JSON args,
3. invoke `processor.py` with valid fixture assets,
4. accept either `done` or explicit `BACKEND_UNAVAILABLE` after successful discovery.
