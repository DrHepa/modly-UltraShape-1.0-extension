import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const manifestPath = resolve(process.cwd(), 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
  id: string;
  type: string;
  node: {
    id: string;
    inputs: Array<{ id: string; kind: string; required: boolean }>;
    outputs: Array<{ id: string; kind: string; required: boolean }>;
  };
  params: Record<string, { default: unknown }>;
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
  it('matches extension identity and semantic ports', () => {
    expect(manifest.id).toBe('modly.ultrashape-refiner-process');
    expect(manifest.type).toBe('process');
    expect(manifest.node.id).toBe('ultrashape-refiner');
    expect(manifest.node.inputs).toEqual([
      { id: 'reference_image', kind: 'image', required: true },
      { id: 'coarse_mesh', kind: 'mesh', required: true },
    ]);
    expect(manifest.node.outputs).toEqual([
      { id: 'refined_mesh', kind: 'mesh', required: true },
    ]);
  });

  it('preserves the required param defaults from the spec', () => {
    expect(manifest.params.checkpoint.default).toBeNull();
    expect(manifest.params.backend.default).toBe('auto');
    expect(manifest.params.steps.default).toBe(30);
    expect(manifest.params.guidance_scale.default).toBe(5.5);
    expect(manifest.params.seed.default).toBeNull();
    expect(manifest.params.preserve_scale.default).toBe(true);
    expect(manifest.params.output_format.default).toBe('glb');
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
