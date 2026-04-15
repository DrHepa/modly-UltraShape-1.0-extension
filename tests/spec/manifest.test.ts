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
    input: string;
    output: string;
    params_schema: Array<Record<string, unknown>>;
  }>;
  type: string;
  testing_fallback_only: {
    status: string;
    temporary: boolean;
    non_native: boolean;
    semantic_contract: string;
    bundle_schema: Record<string, string>;
    draft_manifest_seam: {
      reason: string;
      forward_compatible_intent: string;
    };
  };
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
      input: 'image',
      output: 'mesh',
    });
    expect(node).not.toHaveProperty('type');
    expect(node).not.toHaveProperty('inputs');
    expect(node).not.toHaveProperty('outputs');
  });

  it('uses the proven descriptor-array params_schema with current defaults', () => {
    const descriptors = manifest.nodes[0].params_schema;

    expect(Array.isArray(descriptors)).toBe(true);
    expect(descriptors).toEqual([
      expect.objectContaining({
        name: 'checkpoint',
        type: 'file',
        required: false,
      }),
      expect.objectContaining({
        name: 'coarse_mesh',
        type: 'file',
        required: false,
      }),
      expect.objectContaining({
        name: 'backend',
        type: 'select',
        default: 'auto',
        options: ['auto', 'local', 'remote', 'hybrid'],
      }),
      expect.objectContaining({
        name: 'steps',
        type: 'number',
        default: 30,
      }),
      expect.objectContaining({
        name: 'guidance_scale',
        type: 'number',
        default: 5.5,
      }),
      expect.objectContaining({
        name: 'seed',
        type: 'number',
        required: false,
      }),
      expect.objectContaining({
        name: 'preserve_scale',
        type: 'boolean',
        default: true,
      }),
      expect.objectContaining({
        name: 'output_format',
        type: 'select',
        default: 'glb',
        options: ['glb', 'obj', 'fbx', 'ply'],
      }),
    ]);
    expect(descriptors).not.toContainEqual(
      expect.objectContaining({
        name: 'checkpoint',
        default: null,
      }),
    );
  });

  it('labels the testing fallback as temporary and non-native', () => {
    expect(manifest.testing_fallback_only.status).toBe('temporary_non_native');
    expect(manifest.testing_fallback_only.temporary).toBe(true);
    expect(manifest.testing_fallback_only.non_native).toBe(true);
    expect(manifest.testing_fallback_only.semantic_contract).toBe(
      'reference_image + coarse_mesh -> refined_mesh',
    );
    expect(manifest.testing_fallback_only.bundle_schema).toEqual({
      reference_image: 'path',
      coarse_mesh: 'path',
      output_dir: 'path',
      checkpoint: 'path|null',
      params: 'object',
    });
    expect(manifest.testing_fallback_only.draft_manifest_seam.reason).toContain(
      'may not yet express the ideal multi-input process contract',
    );
  });
});
