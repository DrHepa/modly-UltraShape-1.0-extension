import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  copyInstallSurface,
  createRuntimeInputs,
  runGeneratorProbe,
  stageCheckpoint,
  writeRuntimeStubModules,
} from './install-test-helpers.js';

const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==';

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

describe('install simulation', () => {
  it('lets generator.py consume setup-produced vendor path and real runtime inputs from a copied checkout', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-generator-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    const inputs = createRuntimeInputs(sandbox);

    try {
      const setup = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setup.status).toBe(0);

      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: inputs.coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      const payload = JSON.parse(result.stdout);
      expect(payload).toEqual([
        {
          method: 'generate',
          ok: true,
          result: expect.stringMatching(/\.glb$/),
          loaded: true,
          debug: expect.any(Object),
        },
      ]);
      expect(payload[0].result).toContain('ultrashape-generator-');
      expect(payload[0].debug.last_result).toMatchObject({
        backend: 'local',
        subtrees_loaded: ['vae', 'dit', 'conditioner'],
      });
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('coarse_mesh');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('reference_image');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('treats setup-produced readiness as the install authority for generator load and generate', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-readiness-authority-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    const inputs = createRuntimeInputs(sandbox);

    try {
      const setup = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setup.status).toBe(0);

      writeFileSync(
        path.join(checkout, '.runtime-readiness.json'),
        JSON.stringify(
          {
            backend: 'local',
            checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
            config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
            ext_dir: checkout,
            install_ready: false,
            install_success: false,
            missing_required: ['weight:models/ultrashape/ultrashape_v1.pt'],
            required_imports_ok: true,
            runtime_ready: false,
            status: 'blocked',
            vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
            weights_ready: false,
          },
          null,
          2,
        ),
      );

      const result = runGeneratorProbe(
        checkout,
        [
          { method: 'is_downloaded' },
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: inputs.coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: false, loaded: false },
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'WEIGHTS_MISSING',
            message: 'Required runtime weights are unavailable: weight:models/ultrashape/ultrashape_v1.pt.',
          },
          loaded: false,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps copied checkouts limited to the model shell authority', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-sim-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      expect(existsSync(path.join(checkout, 'generator.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'setup.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'processor.py'))).toBe(false);
      expect(readFileSync(path.join(checkout, 'README.md'), 'utf8')).not.toContain('processor.py');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps generator.py honest when setup reported blocked readiness', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-install-blocked-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    const inputs = createRuntimeInputs(sandbox);

    try {
      const setup = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setup.status).toBe(1);

      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: inputs.coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'WEIGHTS_MISSING',
            message: 'Required runtime weights are unavailable: weight:models/ultrashape/ultrashape_v1.pt.',
          },
          loaded: false,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
