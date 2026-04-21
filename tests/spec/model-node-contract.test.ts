import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('model node contract', () => {
  it('declares an honest image + mesh -> mesh node for the model shell', () => {
    const manifest = JSON.parse(readFileSync(repoPath('manifest.json'), 'utf8'));
    const [node] = manifest.nodes;

    expect(node).toMatchObject({
      id: 'refine',
      input: 'image',
      inputs: ['image', 'mesh'],
      output: 'mesh',
    });
  });

  it('keeps mesh routing as node contract truth instead of legacy aliases', () => {
    const manifest = JSON.parse(readFileSync(repoPath('manifest.json'), 'utf8'));
    const [node] = manifest.nodes;
    const paramIds = Array.isArray(node.params_schema)
      ? node.params_schema.map((param: { id?: unknown }) => param.id)
      : [];

    expect(paramIds).not.toContain('reference_image');
    expect(paramIds).not.toContain('coarse_mesh');
    expect(paramIds).not.toContain('mesh_path');
    expect(JSON.stringify(manifest)).not.toContain('input.filePath');
    expect(JSON.stringify(manifest)).not.toContain('params.coarse_mesh');
    expect(JSON.stringify(manifest)).not.toContain('reference_image');
    expect(JSON.stringify(manifest)).not.toContain('coarse_mesh');
  });
});
