import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const extractedRootPaths = [
  'manifest.json',
  'setup.py',
  'processor.py',
  'README.md',
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
];

function copyExtractedRoot() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-github-install-'));
  const installDir = join(root, 'modly-UltraShape-1.0-extension');

  for (const relativePath of extractedRootPaths) {
    cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
  }

  return {
    installDir,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

describe('UltraShape GitHub install smoke', () => {
  it('validates extracted-root discovery, setup.py install, and acceptable processor smoke outcome', () => {
    const simulation = copyExtractedRoot();

    try {
      expect(existsSync(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/request.json'))).toBe(true);

      const manifest = JSON.parse(readFileSync(resolve(simulation.installDir, 'manifest.json'), 'utf8')) as {
        id: string;
        entry: string;
      };
      expect(manifest.id).toBe('modly.ultrashape-refiner-process');
      expect(manifest.entry).toBe('processor.py');

      const setup = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: 90,
        cuda_version: 128,
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
      });
      expect(setup.status).toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);

      const smokePayload = {
        input: {
          filePath: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/reference-image.png'),
        },
        params: {
          coarse_mesh: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb'),
        },
        workspaceDir: resolve(simulation.installDir, 'smoke-output'),
      };

      const outcome = spawnSync('python3', ['processor.py'], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        input: `${JSON.stringify(smokePayload)}\n`,
      });

      const events = outcome.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(outcome.status).toBe(0);
      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('BACKEND_UNAVAILABLE'),
        code: 'BACKEND_UNAVAILABLE',
      });
    } finally {
      simulation.cleanup();
    }
  });
});
