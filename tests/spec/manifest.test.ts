import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('stable shell manifest', () => {
  it('declares the exact Hunyuan-aligned top-level shell contract', () => {
    const manifestPath = repoPath('manifest.json');

    expect(existsSync(manifestPath)).toBe(true);
    expect(existsSync(repoPath('processor.py'))).toBe(false);

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

    expect(Object.keys(manifest)).toEqual([
      'id',
      'name',
      'source',
      'description',
      'version',
      'author',
      'generator_class',
      'tags',
      'nodes',
    ]);
    expect(manifest.id).toBe('modly.ultrashape-refiner-model');
    expect(manifest.name).toBe('UltraShape Refiner');
    expect(manifest.source).toBe('https://github.com/DrHepa/modly-UltraShape-1.0-extension');
    expect(manifest.description).not.toContain('runtime');
    expect(manifest.description).toContain('generator');
    expect(manifest.author).toBe('DrHepa');
    expect(manifest.generator_class).toBe('UltraShapeGenerator');
    expect(manifest.nodes).toHaveLength(1);
    expect(manifest).not.toHaveProperty('type');
    expect(manifest).not.toHaveProperty('local_only');
    expect(manifest).not.toHaveProperty('shell');
    expect(manifest).not.toHaveProperty('contract');
    expect(manifest).not.toHaveProperty('forbidden_execution_modes');
    expect(JSON.stringify(manifest)).not.toContain('processor.py');
    expect(JSON.stringify(manifest)).not.toContain('processor.js');
    expect(JSON.stringify(manifest)).not.toContain('process-refiner');
  });

  it('declares the exact Hunyuan-aligned node shell keys without process leakage', () => {
    const manifest = JSON.parse(readFileSync(repoPath('manifest.json'), 'utf8'));
    const [node] = manifest.nodes;

    expect(Object.keys(node)).toEqual([
      'id',
      'name',
      'input',
      'inputs',
      'output',
      'hf_repo',
      'download_check',
      'hf_skip_prefixes',
      'description',
      'params_schema',
    ]);
    expect(node.id).toBe('refine');
    expect(node.name).toBe('Refine Mesh');
    expect(node.input).toBe('image');
    expect(node.inputs).toEqual(['image', 'mesh']);
    expect(node.output).toBe('mesh');
    expect(node.description).toContain('mesh');
    expect(JSON.stringify(node)).not.toContain('processor');
    expect(JSON.stringify(node)).not.toContain('process');
  });
});
