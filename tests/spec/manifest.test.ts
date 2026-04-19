import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const manifestPath = resolve(process.cwd(), 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
  id: string;
  entry: string;
  nodes: Array<{
    id: string;
    name: string;
    type: string;
    input: string;
    output: string;
    inputs?: Array<Record<string, unknown>>;
    params_schema: Array<Record<string, unknown>>;
  }>;
  type: string;
  testing_fallback_only?: unknown;
  fallback_input_contract?: unknown;
  draft_manifest_seam?: unknown;
};

describe('UltraShape refiner manifest', () => {
  it('matches the Python installable Modly process contract without CommonJS-only node fields', () => {
    expect(manifest.id).toBe('modly.ultrashape-refiner-process');
    expect(manifest.type).toBe('process');
    expect(manifest.entry).toBe('processor.py');
    expect(manifest.nodes).toHaveLength(1);

    const [node] = manifest.nodes;

    expect(node).toMatchObject({
      id: 'ultrashape-refiner',
      name: 'UltraShape Refiner Process',
      type: 'process',
      input: 'image',
      output: 'mesh',
    });
    expect(node).not.toHaveProperty('outputs');

    expect(node.inputs).toEqual([
      {
        name: 'reference_image',
        label: 'Reference Image',
        type: 'image',
        required: true,
      },
      {
        name: 'coarse_mesh',
        label: 'Coarse Mesh',
        type: 'mesh',
        required: true,
      },
    ]);
  });

  it('uses canonical visible params_schema descriptors for the UI contract', () => {
    const descriptors = manifest.nodes[0].params_schema;

    expect(Array.isArray(descriptors)).toBe(true);
    expect(descriptors).toEqual([
      {
        id: 'backend',
        label: 'Backend',
        type: 'select',
        default: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'local', label: 'local' },
        ],
      },
      {
        id: 'steps',
        label: 'Steps',
        type: 'int',
        default: 30,
      },
      {
        id: 'guidance_scale',
        label: 'Guidance Scale',
        type: 'float',
        default: 5.5,
      },
      {
        id: 'seed',
        label: 'Seed',
        type: 'int',
        default: -1,
      },
      {
        id: 'output_format',
        label: 'Output Format',
        type: 'select',
        default: 'glb',
        options: [{ value: 'glb', label: 'glb' }],
      },
    ]);

    for (const hiddenParamId of ['checkpoint', 'coarse_mesh', 'preserve_scale']) {
      expect(descriptors).not.toContainEqual(expect.objectContaining({ id: hiddenParamId }));
      expect(descriptors).not.toContainEqual(expect.objectContaining({ name: hiddenParamId }));
    }
  });

  it('does not publish fallback authority outside processor.py', () => {
    expect(manifest).not.toHaveProperty('testing_fallback_only');
    expect(manifest).not.toHaveProperty('fallback_input_contract');
    expect(manifest).not.toHaveProperty('draft_manifest_seam');
  });
});
