import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const curatedPayloadPaths = [
  'manifest.json',
  'processor.js',
  'README.md',
  'runtime/modly/package.json',
  'runtime/modly/processes/ultrashape-refiner/index.js',
  'runtime/modly/processes/ultrashape-refiner/validate.js',
  'runtime/modly/processes/ultrashape-refiner/normalize.js',
  'runtime/modly/processes/ultrashape-refiner/preflight.js',
  'runtime/modly/processes/ultrashape-refiner/runtime.js',
  'runtime/modly/processes/ultrashape-refiner/progress.js',
  'runtime/modly/processes/ultrashape-refiner/types.js',
  'runtime/modly/adapters/ultrashape/client.js',
  'runtime/modly/adapters/ultrashape/remote.js',
  'runtime/modly/adapters/ultrashape/local.js',
];

function copyCuratedPayload() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-install-sim-'));
  const installDir = join(root, 'extensions', 'modly.ultrashape-refiner-process');

  for (const relativePath of curatedPayloadPaths) {
    cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
  }

  return {
    installDir,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

describe('UltraShape curated install payload', () => {
  it('matches the expected copied payload layout for the Modly extensions directory', () => {
    const simulation = copyCuratedPayload();

    try {
      for (const relativePath of curatedPayloadPaths) {
        expect(existsSync(resolve(simulation.installDir, relativePath)), `${relativePath} should be copied`).toBe(true);
      }

      expect(existsSync(resolve(simulation.installDir, 'package.json'))).toBe(false);

      const runtimePackage = JSON.parse(
        readFileSync(resolve(simulation.installDir, 'runtime/modly/package.json'), 'utf8'),
      ) as { type?: string };
      expect(runtimePackage.type).toBe('commonjs');
    } finally {
      simulation.cleanup();
    }
  });

  it('loads processor.js from the copied payload through require()', () => {
    const simulation = copyCuratedPayload();

    try {
      const requireFromInstall = createRequire(resolve(simulation.installDir, 'index.js'));
      const processor = requireFromInstall('./processor.js');

      expect(typeof processor).toBe('function');
    } finally {
      simulation.cleanup();
    }
  });
});
