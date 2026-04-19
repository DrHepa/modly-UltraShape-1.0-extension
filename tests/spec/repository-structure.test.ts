import { existsSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();

function listFiles(root: string): string[] {
  if (!existsSync(root)) {
    return [];
  }

  const pending = [root];
  const files: string[] = [];

  while (pending.length > 0) {
    const current = pending.pop();
    if (!current) {
      continue;
    }

    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const nextPath = resolve(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(nextPath);
        continue;
      }

      files.push(nextPath);
    }
  }

  return files;
}

const shellPaths = ['manifest.json', 'processor.py', 'setup.py', 'README.md'];

const vendoredRuntimePaths = [
  'runtime/configs/infer_dit_refine.yaml',
  'runtime/vendor/ultrashape_runtime/__init__.py',
  'runtime/vendor/ultrashape_runtime/pipelines.py',
  'runtime/vendor/ultrashape_runtime/preprocessors.py',
  'runtime/vendor/ultrashape_runtime/rembg.py',
  'runtime/vendor/ultrashape_runtime/surface_loaders.py',
  'runtime/vendor/ultrashape_runtime/schedulers.py',
  'runtime/vendor/ultrashape_runtime/utils/__init__.py',
  'runtime/vendor/ultrashape_runtime/utils/checkpoint.py',
  'runtime/vendor/ultrashape_runtime/utils/voxelize.py',
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

const testPaths = [
  'tests/spec/manifest.test.ts',
  'tests/spec/request-contract.test.ts',
  'tests/spec/runtime-flow.test.ts',
  'tests/spec/repository-structure.test.ts',
];

const removedAuthorityPaths = [
  'runtime/patches/README.md',
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
  'tests/spec/fallback-fixtures.test.ts',
];

describe('UltraShape repository structure contract', () => {
  it('keeps shell, vendored runtime, and active specs separated by path and role', () => {
    for (const filePath of [
      ...shellPaths,
      ...vendoredRuntimePaths,
      ...testPaths,
    ]) {
      expect(existsSync(resolve(repoRoot, filePath)), `${filePath} should exist`).toBe(true);
    }

    for (const filePath of shellPaths) {
      expect(
        filePath.startsWith('adapters/') || filePath.startsWith('fixtures/') || filePath.startsWith('tests/'),
      ).toBe(false);
    }

    for (const filePath of testPaths) {
      expect(filePath.startsWith('tests/spec/')).toBe(true);
    }

    for (const filePath of removedAuthorityPaths) {
      expect(existsSync(resolve(repoRoot, filePath)), `${filePath} should be removed`).toBe(false);
    }
  });

  it('keeps only the repo-root shell and the upstream runtime closure on disk', () => {
    expect(listFiles(resolve(repoRoot, 'src'))).toEqual([]);
    expect(existsSync(resolve(repoRoot, 'tests/spec/ts-runtime-boundary.test.ts'))).toBe(false);

    expect(
      readdirSync(resolve(repoRoot, 'runtime/vendor/ultrashape_runtime/utils'))
        .filter((entry) => entry !== '__pycache__')
        .sort(),
    ).toEqual(['__init__.py', 'checkpoint.py', 'voxelize.py']);
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
