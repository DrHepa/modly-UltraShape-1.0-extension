import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import * as adapterBoundary from '../../adapters/ultrashape/client.js';
import { UltraShapeLocalAdapter } from '../../adapters/ultrashape/local.js';
import { UltraShapeRemoteAdapter } from '../../adapters/ultrashape/remote.js';

const repoRoot = process.cwd();

const runtimePaths = [
  'src/processes/ultrashape-refiner/index.ts',
  'src/processes/ultrashape-refiner/validate.ts',
  'src/processes/ultrashape-refiner/normalize.ts',
  'src/processes/ultrashape-refiner/runtime.ts',
  'src/processes/ultrashape-refiner/progress.ts',
  'src/adapters/ultrashape/client.ts',
  'src/adapters/ultrashape/remote.ts',
  'src/adapters/ultrashape/local.ts',
  'adapters/ultrashape/client.ts',
  'adapters/ultrashape/remote.ts',
  'adapters/ultrashape/local.ts',
];

const configAndDocsPaths = ['manifest.json', 'processor.py', 'setup.py', 'README.md'];

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
      ...fixturePaths,
      ...testPaths,
    ]) {
      expect(existsSync(resolve(repoRoot, filePath)), `${filePath} should exist`).toBe(true);
    }

    for (const filePath of runtimePaths) {
      expect(
        filePath.startsWith('src/') ||
          filePath.startsWith('adapters/') ||
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

  it('exposes the adapter boundary through the spec-required top-level path without changing runtime exports', () => {
    expect(typeof adapterBoundary).toBe('object');
    expect(UltraShapeLocalAdapter).toBeTypeOf('function');
    expect(UltraShapeRemoteAdapter).toBeTypeOf('function');
  });

  it('keeps the repo-root Python boundary as the only active install surface', () => {
    expect(existsSync(resolve(repoRoot, 'package.json'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'processor.py'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'setup.py'))).toBe(true);
    expect(existsSync(resolve(repoRoot, 'processor.js'))).toBe(false);
    expect(existsSync(resolve(repoRoot, 'runtime/modly'))).toBe(false);
  });
});
