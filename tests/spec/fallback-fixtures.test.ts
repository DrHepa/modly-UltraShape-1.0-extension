import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const manifest = JSON.parse(readFileSync(resolve(repoRoot, 'manifest.json'), 'utf8')) as {
  testing_fallback_only: {
    semantic_contract: string;
    bundle_schema: Record<string, string>;
  };
};
const fixtureRoot = resolve(repoRoot, 'fixtures/requests/refiner-bundle');
const fixtureRequest = JSON.parse(readFileSync(resolve(fixtureRoot, 'request.json'), 'utf8')) as Record<
  string,
  unknown
>;
const readme = readFileSync(resolve(repoRoot, 'README.md'), 'utf8');

describe('UltraShape fallback fixtures and docs', () => {
  it('keeps fixture request field names aligned with the manifest fallback schema', () => {
    expect(Object.keys(fixtureRequest).sort()).toEqual(
      Object.keys(manifest.testing_fallback_only.bundle_schema).sort(),
    );
  });

  it('ships placeholder assets and expected output layout for the fallback bundle', () => {
    expect(existsSync(resolve(fixtureRoot, 'assets/reference-image.png'))).toBe(true);
    expect(existsSync(resolve(fixtureRoot, 'assets/coarse-mesh.glb'))).toBe(true);
    expect(existsSync(resolve(fixtureRoot, 'expected/output/refined-mesh.glb'))).toBe(true);
  });

  it('documents the semantic contract, temporary fallback, and deferred future seam', () => {
    expect(manifest.testing_fallback_only.semantic_contract).toBe('reference_image + coarse_mesh -> refined.glb');
    expect(readme).toContain('local-only');
    expect(readme).toContain('glb-only');
    expect(readme).toContain('temporary `params.coarse_mesh` seam');
    expect(readme).toContain('root `manifest.json` + `setup.py` + `processor.py`');
    expect(readme).toContain('Hunyuan is the recommended and currently validated upstream source, but it is NOT required');
    expect(readme).toContain('## Future Modly seam');
    expect(readme).not.toContain('processor.js');
    expect(readme).not.toContain('runtime/modly/**');

    for (const fieldName of Object.keys(manifest.testing_fallback_only.bundle_schema)) {
      expect(readme).toContain(`\`${fieldName}\``);
    }
  });
});
