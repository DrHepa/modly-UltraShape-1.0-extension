import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import * as adapterBoundary from '../../src/adapters/ultrashape/client.js';
import { UltraShapeLocalAdapter } from '../../src/adapters/ultrashape/local.js';

const repoRoot = process.cwd();

const runtimePaths = [
  'src/processes/ultrashape-refiner/index.ts',
  'src/processes/ultrashape-refiner/validate.ts',
  'src/processes/ultrashape-refiner/normalize.ts',
  'src/processes/ultrashape-refiner/runtime.ts',
  'src/processes/ultrashape-refiner/progress.ts',
  'src/adapters/ultrashape/client.ts',
  'src/adapters/ultrashape/local.ts',
];

const configAndDocsPaths = ['manifest.json', 'processor.py', 'setup.py', 'README.md'];

const vendoredRuntimePaths = [
  'runtime/configs/infer_dit_refine.yaml',
  'runtime/patches/README.md',
  'runtime/vendor/ultrashape_runtime/__init__.py',
  'runtime/vendor/ultrashape_runtime/pipelines.py',
  'runtime/vendor/ultrashape_runtime/preprocessors.py',
  'runtime/vendor/ultrashape_runtime/rembg.py',
  'runtime/vendor/ultrashape_runtime/surface_loaders.py',
  'runtime/vendor/ultrashape_runtime/schedulers.py',
  'runtime/vendor/ultrashape_runtime/utils/__init__.py',
  'runtime/vendor/ultrashape_runtime/utils/checkpoint.py',
  'runtime/vendor/ultrashape_runtime/utils/mesh.py',
  'runtime/vendor/ultrashape_runtime/utils/tensors.py',
  'runtime/vendor/ultrashape_runtime/models/conditioner_mask.py',
  'runtime/vendor/ultrashape_runtime/models/denoisers/__init__.py',
  'runtime/vendor/ultrashape_runtime/models/denoisers/dit_mask.py',
  'runtime/vendor/ultrashape_runtime/models/denoisers/moe_layers.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/__init__.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/model.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/attention_blocks.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/attention_processors.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/surface_extractors.py',
  'runtime/vendor/ultrashape_runtime/models/autoencoders/volume_decoders.py',
];

const fixturePaths = [
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
];

const testPaths = [
  'tests/spec/manifest.test.ts',
  'tests/spec/request-contract.test.ts',
  'tests/spec/runtime-flow.test.ts',
  'tests/spec/fallback-fixtures.test.ts',
  'tests/spec/repository-structure.test.ts',
];

describe('UltraShape repository structure contract', () => {
  it('keeps runtime, config/docs, fixtures, and tests separated by path and role', () => {
    for (const filePath of [
      ...runtimePaths,
      ...configAndDocsPaths,
      ...vendoredRuntimePaths,
      ...fixturePaths,
      ...testPaths,
    ]) {
      expect(existsSync(resolve(repoRoot, filePath)), `${filePath} should exist`).toBe(true);
    }

    for (const filePath of runtimePaths) {
      expect(
        filePath.startsWith('src/') ||
          filePath.startsWith('runtime/modly/'),
      ).toBe(true);
    }

    for (const filePath of configAndDocsPaths) {
      expect(
        filePath.startsWith('src/') || filePath.startsWith('adapters/') || filePath.startsWith('fixtures/') || filePath.startsWith('tests/'),
      ).toBe(false);
    }

    for (const filePath of fixturePaths) {
      expect(filePath.startsWith('fixtures/')).toBe(true);
    }

    for (const filePath of testPaths) {
      expect(filePath.startsWith('tests/spec/')).toBe(true);
    }
  });

  it('keeps the local-only TypeScript compatibility adapter inside src/', () => {
    expect(typeof adapterBoundary).toBe('object');
    expect(UltraShapeLocalAdapter).toBeTypeOf('function');
    expect(existsSync(resolve(repoRoot, 'src/adapters/ultrashape/remote.ts'))).toBe(false);
    expect(existsSync(resolve(repoRoot, 'adapters/ultrashape/client.ts'))).toBe(false);
  });

  it('keeps the repo-root Python boundary as the only active install surface', () => {
    expect(existsSync(resolve(repoRoot, 'package.json'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'processor.py'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'setup.py'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'runtime/vendor/ultrashape_runtime'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'processor.js'))).toBe(false);
    expect(existsSync(resolve(repoRoot, 'runtime/modly'))).toBe(false);
  });
});
