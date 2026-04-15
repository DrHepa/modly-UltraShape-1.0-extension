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
  testing_fallback_only: {
    status: string;
    temporary: boolean;
    non_native: boolean;
    semantic_contract: string;
    bundle_schema: Record<string, string>;
    hidden_params: {
      checkpoint: {
        location: string;
        default: null;
        reason: string;
      };
      preserve_scale: {
        location: string;
        default: boolean;
        reason: string;
      };
    };
    fallback_input_contract: {
      primary: {
        reference_image: string;
        coarse_mesh: string;
      };
      secondary: {
        reference_image: string;
        coarse_mesh: string;
      };
      rationale: string;
    };
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
          { value: 'remote', label: 'remote' },
          { value: 'hybrid', label: 'hybrid' },
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
        options: [
          { value: 'glb', label: 'glb' },
          { value: 'obj', label: 'obj' },
          { value: 'fbx', label: 'fbx' },
          { value: 'ply', label: 'ply' },
        ],
      },
    ]);

    for (const hiddenParamId of ['checkpoint', 'coarse_mesh', 'preserve_scale']) {
      expect(descriptors).not.toContainEqual(expect.objectContaining({ id: hiddenParamId }));
      expect(descriptors).not.toContainEqual(expect.objectContaining({ name: hiddenParamId }));
    }
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
    expect(manifest.testing_fallback_only.hidden_params).toEqual({
      checkpoint: {
        location: 'runtime/default metadata',
        default: null,
        reason: 'Deferred because checkpoint is an asset dependency better represented outside the current panel controls.',
      },
      preserve_scale: {
        location: 'runtime-only default',
        default: true,
        reason: 'Deferred because the current UI does not faithfully support the required boolean control.',
      },
    });
    expect(manifest.testing_fallback_only.fallback_input_contract).toEqual({
      primary: {
        reference_image: 'input.inputs.reference_image.filePath',
        coarse_mesh: 'input.inputs.coarse_mesh.filePath',
      },
      secondary: {
        reference_image: 'input.filePath',
        coarse_mesh: 'params.coarse_mesh',
      },
      rationale:
        'Temporary fallback seam only until Modly supplies native multi-input process routing for both semantic ids.',
    });
    expect(manifest.testing_fallback_only.draft_manifest_seam.reason).toContain(
      'may not yet express the ideal multi-input process contract',
    );
  });
});
