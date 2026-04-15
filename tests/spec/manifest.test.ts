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
    inputs: Array<{ id: string; name: string; type: string; required: boolean }>;
    outputs: Array<{ id: string; name: string; type: string; required: boolean }>;
    params_schema: {
      type: string;
      properties: Record<string, Record<string, unknown>>;
    };
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
  it('matches the installable Modly process contract and semantic ports', () => {
    expect(manifest.id).toBe('modly.ultrashape-refiner-process');
    expect(manifest.type).toBe('process');
    expect(manifest.entry).toBe('processor.js');
    expect(manifest.nodes).toHaveLength(1);

    const [node] = manifest.nodes;

    expect(node).toMatchObject({
      id: 'ultrashape-refiner',
      name: 'UltraShape Refiner Process',
      type: 'process',
      input: 'image',
      output: 'mesh',
    });
    expect(node.inputs).toEqual([
      { id: 'reference_image', name: 'Reference Image', type: 'image', required: true },
      { id: 'coarse_mesh', name: 'Coarse Mesh', type: 'mesh', required: true },
    ]);
    expect(node.outputs).toEqual([
      { id: 'refined_mesh', name: 'Refined Mesh', type: 'mesh', required: true },
    ]);
  });

  it('preserves params_schema defaults and install-time validation constraints from the spec', () => {
    const properties = manifest.nodes[0].params_schema.properties;

    expect(manifest.nodes[0].params_schema.type).toBe('object');
    expect(properties.checkpoint).toMatchObject({
      type: ['string', 'null'],
      default: null,
    });
    expect(properties.backend).toMatchObject({
      type: 'string',
      enum: ['auto', 'local', 'remote', 'hybrid'],
      default: 'auto',
    });
    expect(properties.steps).toMatchObject({
      type: 'integer',
      minimum: 1,
      default: 30,
    });
    expect(properties.guidance_scale).toMatchObject({
      type: 'number',
      exclusiveMinimum: 0,
      default: 5.5,
    });
    expect(properties.seed).toMatchObject({
      type: ['integer', 'null'],
      default: null,
    });
    expect(properties.preserve_scale).toMatchObject({
      type: 'boolean',
      default: true,
    });
    expect(properties.output_format).toMatchObject({
      type: 'string',
      enum: ['glb', 'obj', 'fbx', 'ply'],
      default: 'glb',
    });
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
