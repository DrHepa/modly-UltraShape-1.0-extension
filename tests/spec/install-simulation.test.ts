import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  copyInstallSurface,
  createRuntimeInputs,
  stageCheckpoint,
  writeRuntimeStubModules,
} from './install-test-helpers.js';

function runSetup(cwd: string, extDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['-S', 'setup.py', '--ext-dir', extDir], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

function runProcessor(cwd: string, payload: unknown, outputDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['processor.py', '--process', '--output-dir', outputDir], {
    cwd,
    encoding: 'utf8',
    input: JSON.stringify(payload),
    env: {
      ...process.env,
      ...env,
    },
  });
}

describe('install simulation', () => {
  it('lets processor.py use setup-produced readiness for the real local runner path', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-sim-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const outputDir = path.join(sandbox, 'output');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    const inputs = createRuntimeInputs(sandbox);

    try {
      const setup = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setup.status).toBe(0);

      const result = runProcessor(
        checkout,
        {
          reference_image: inputs.referenceImage,
          coarse_mesh: inputs.coarseMesh,
        },
        outputDir,
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          backend: 'local',
          format: 'glb',
          file_path: path.join(outputDir, 'refined.glb'),
        },
      });
      expect(existsSync(path.join(outputDir, 'refined.glb'))).toBe(true);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps processor.py honest when setup reported blocked readiness', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-blocked-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const outputDir = path.join(sandbox, 'output');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    const inputs = createRuntimeInputs(sandbox);

    try {
      const setup = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setup.status).toBe(1);

      const result = runProcessor(
        checkout,
        {
          reference_image: inputs.referenceImage,
          coarse_mesh: inputs.coarseMesh,
        },
        outputDir,
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(1);
      expect(JSON.parse(result.stdout)).toEqual({
        ok: false,
        error: {
          code: 'WEIGHTS_MISSING',
          message: 'Required runtime weights are unavailable: weight:models/ultrashape/ultrashape_v1.pt.',
        },
      });
      expect(existsSync(path.join(outputDir, 'refined.glb'))).toBe(false);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
