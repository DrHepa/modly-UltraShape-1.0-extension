import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const installSurfacePaths = [
  'manifest.json',
  'processor.py',
  'setup.py',
  'README.md',
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
];

function copyInstallSurface() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-install-sim-'));
  const installDir = join(root, 'extensions', 'modly.ultrashape-refiner-process');

  for (const relativePath of installSurfacePaths) {
    cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
  }

  return {
    installDir,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

describe('UltraShape Python install surface', () => {
  it('matches the expected GitHub-extracted payload layout for the Modly extensions directory', () => {
    const simulation = copyInstallSurface();

    try {
      for (const relativePath of installSurfacePaths) {
        expect(existsSync(resolve(simulation.installDir, relativePath)), `${relativePath} should be copied`).toBe(true);
      }

      expect(existsSync(resolve(simulation.installDir, 'package.json'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'processor.js'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'runtime/modly'))).toBe(false);
    } finally {
      simulation.cleanup();
    }
  });

  it('keeps manifest entry, setup contract, and processor smoke aligned inside the copied payload', () => {
    const simulation = copyInstallSurface();

    try {
      const manifest = JSON.parse(readFileSync(resolve(simulation.installDir, 'manifest.json'), 'utf8')) as {
        entry: string;
      };
      expect(manifest.entry).toBe('processor.py');

      const outcome = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: '90',
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
      });

      expect(outcome.status).toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);

      const smoke = spawnSync('python3', ['processor.py'], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            filePath: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/reference-image.png'),
          },
          params: {
            coarse_mesh: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb'),
          },
          workspaceDir: resolve(simulation.installDir, 'smoke-output'),
        })}\n`,
      });

      const events = smoke.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(smoke.status).toBe(0);
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
