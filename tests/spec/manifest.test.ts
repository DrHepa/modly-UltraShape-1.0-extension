import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('stable shell manifest', () => {
  it('declares the honest local-only model shell', () => {
    const manifestPath = repoPath('manifest.json');

    expect(existsSync(manifestPath)).toBe(true);
    expect(existsSync(repoPath('processor.py'))).toBe(false);

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

    expect(manifest.type).toBe('model');
    expect(manifest.local_only).toBe(true);
    expect(manifest.generator_class).toBe('UltraShapeGenerator');
    expect(manifest.nodes).toHaveLength(1);
    expect(manifest.nodes[0]).toMatchObject({
      id: 'refine',
      name: 'Refine Mesh',
      input: 'image',
      inputs: ['image', 'mesh'],
      output: 'mesh',
    });

    expect(manifest).not.toHaveProperty('shell');
    expect(manifest).not.toHaveProperty('contract');
    expect(manifest).not.toHaveProperty('forbidden_execution_modes');
    expect(JSON.stringify(manifest)).not.toContain('processor.py');
    expect(JSON.stringify(manifest)).not.toContain('setup.py');
    expect(JSON.stringify(manifest)).not.toContain('process-refiner');
    expect(JSON.stringify(manifest)).not.toContain('reference_image');
    expect(JSON.stringify(manifest)).not.toContain('coarse_mesh');
  });
});
