import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('stable shell manifest', () => {
  it('declares the honest local-only process-refiner shell', () => {
    const manifestPath = repoPath('manifest.json');

    expect(existsSync(manifestPath)).toBe(true);

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

    expect(manifest.shell).toEqual({
      local_only: true,
      processor: 'processor.py',
      setup: 'setup.py',
    });
    expect(manifest.contract).toEqual({
      kind: 'process-refiner',
      required_inputs: ['reference_image', 'coarse_mesh'],
      output_artifact: 'output_dir/refined.glb',
    });
    expect(manifest.forbidden_execution_modes).toEqual([
      'remote',
      'hybrid',
      'model-wrapper',
    ]);
    expect(JSON.stringify(manifest)).not.toContain('filePath');
    expect(JSON.stringify(manifest)).not.toContain('params.coarse_mesh');
    expect(JSON.stringify(manifest)).not.toContain('input.filePath');
    expect(JSON.stringify(manifest)).not.toContain('fallback');
  });
});
