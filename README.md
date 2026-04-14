# modly-UltraShape-1.0-extension

Pure process extension for UltraShape mesh refinement.

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

## Coarse-mesh policy

The coarse mesh is origin-agnostic. Hunyuan is the recommended and currently validated upstream source, but it is NOT required. Any coarse mesh that satisfies the accepted mesh-format validation may be used.

## Temporary fallback usage

Before Modly core supports native multi-input process ports, this extension can be exercised through the temporary testing-only bundle at `fixtures/requests/refiner-bundle/request.json`.

This fallback is NON-NATIVE and TEMPORARY. It exists only to preserve the real semantic contract during testing without inventing a fake single-input workflow contract.

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
