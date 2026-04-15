# modly-UltraShape-1.0-extension

Pure process extension for UltraShape mesh refinement.

## Installable Modly contract

This repo now ships the installable process-extension surface expected by current Modly discovery/runtime rules:

- root `manifest.json`
- root `processor.js`
- committed CommonJS runtime mirror under `runtime/modly/**`

The installed payload is a CURATED COPY. Do NOT copy the repo root `package.json` into the installed extension folder, because the repo is ESM-only for development while Modly must load `processor.js` through CommonJS `require()`.

### Copy-install steps

Copy only these artifacts into `/home/drhepa/Documentos/Modly/extensions/modly.ultrashape-refiner-process`:

- `manifest.json`
- `processor.js`
- `README.md`
- `runtime/modly/**`

Expected discovery signal: Modly manifest scanning should discover process node `modly.ultrashape-refiner-process/ultrashape-refiner` from `manifest.nodes[]` with both legacy `input/output` and named `inputs[]/outputs[]` present.

### Processor contract

`processor.js` exports `module.exports = async function(input, params, context)` and expects named Modly inputs:

- `input.inputs.reference_image.filePath`
- `input.inputs.coarse_mesh.filePath`

The processor maps those inputs into the UltraShape runtime request, resolves `output_dir` from `context.workspaceDir ?? context.tempDir`, merges params against manifest defaults, and returns:

```json
{
  "filePath": "/absolute/path/to/refined.glb"
}
```

## Extension identity

- Repository/project: `modly-UltraShape-1.0-extension`
- Extension id: `modly.ultrashape-refiner-process`
- Node id: `ultrashape-refiner`
- Semantic contract: `reference_image + coarse_mesh -> refined_mesh`

This extension is a refiner, NOT a coarse-mesh generator or single-image generation wrapper.

## Runtime stance

- Remote/hybrid-first execution is the current stance.
- Linux ARM64 is treated as a strong reason to prefer remote or hybrid execution.
- Local execution remains a deferred/optional boundary until there is a supported story for it.
- If no eligible backend is configured, the accepted execution failure is an explicit `BACKEND_UNAVAILABLE` error. Discovery/install can still be considered valid in that case.

## Coarse-mesh policy

The coarse mesh is origin-agnostic. Hunyuan is the recommended and currently validated upstream source, but it is NOT required. Any coarse mesh that satisfies the accepted mesh-format validation may be used.

## Temporary fallback usage

Before Modly core supports native multi-input process ports, this extension can be exercised through the temporary testing-only bundle at `fixtures/requests/refiner-bundle/request.json`.

This fallback is NON-NATIVE and TEMPORARY. It exists only to preserve the real semantic contract during testing without inventing a fake single-input workflow contract.

The installable path is the PRIMARY contract now. The fallback bundle remains useful for repo-local tests and fixture documentation only.

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

The fallback keeps naming parity with the final semantic contract: `reference_image + coarse_mesh -> refined_mesh`.

## Fixture layout

`fixtures/requests/refiner-bundle/`

- `request.json` — temporary testing bundle
- `assets/reference-image.png` — placeholder reference image asset
- `assets/coarse-mesh.glb` — placeholder coarse mesh asset
- `expected/output/refined-mesh.glb` — placeholder refined mesh output layout

## Future Modly seam

The following work belongs to Modly core and is intentionally deferred from this extension batch:

- Native named multi-input process ports for `reference_image` and `coarse_mesh`
- Port-aware routing and validation across workflows
- UI handles that let operators wire both inputs explicitly
- Workflow execution plumbing that runs this process natively without the fallback bundle

When that core seam exists, this extension should plug into those native capabilities without renaming the semantic contract.

## Smoke-test expectation

The best install smoke test from this repo boundary is:

1. copy the curated payload into Modly's `extensions` directory,
2. verify manifest discovery of `modly.ultrashape-refiner-process/ultrashape-refiner`,
3. invoke `processor.js` through the current Modly JS process runner.

If the backend/runtime integration is unavailable outside this repo, the expected fallback outcome is a clear `BACKEND_UNAVAILABLE` execution error rather than a discovery failure.
