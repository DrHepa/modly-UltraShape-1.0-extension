import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { copyInstallSurface, stageCheckpoint, writeRuntimeStubModules } from './install-test-helpers.js';

type Summary = {
  cuda_version?: string | null;
  config_ready: boolean;
  gpu_sm?: string | null;
  install_ready: boolean;
  install_success: boolean;
  missing_required: string[];
  runtime_ready: boolean;
  runtime_closure_ready: boolean;
  status: string;
  vendor_path: string;
};

function runSetup(cwd: string, extDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync(
    'python3',
    [
      '-S',
      'setup.py',
      '--ext-dir',
      extDir,
      '--python-exe',
      '/opt/modly/python/bin/python3',
      JSON.stringify({ gpu_sm: '89', cuda_version: '12.4' }),
    ],
    {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  },
  );
}

describe('GitHub install smoke', () => {
  it('stages readiness artifacts from a copied clean-room checkout', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-github-install-'));
    const checkout = path.join(sandbox, 'modly-UltraShape-1.0-extension');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);

    try {
      const result = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });

      expect(result.status).toBe(0);
      expect(existsSync(path.join(checkout, '.setup-summary.json'))).toBe(true);
      expect(existsSync(path.join(checkout, '.runtime-readiness.json'))).toBe(true);
      expect(existsSync(path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'processor.py'))).toBe(false);
      expect(existsSync(path.join(checkout, 'processor.js'))).toBe(false);
      expect(existsSync(path.join(checkout, 'src'))).toBe(false);
      expect(existsSync(path.join(checkout, 'runtime', 'modly'))).toBe(false);

      const summary = JSON.parse(readFileSync(path.join(checkout, '.setup-summary.json'), 'utf8')) as Summary;
      expect(summary).toMatchObject({
        gpu_sm: '89',
        cuda_version: '12.4',
        config_ready: true,
        runtime_ready: true,
        install_success: true,
        install_ready: true,
        runtime_closure_ready: true,
        status: 'ready',
        missing_required: [],
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
      });
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('wheel');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps copied checkout installs blocked when the required checkpoint is absent', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-github-blocked-'));
    const checkout = path.join(sandbox, 'modly-UltraShape-1.0-extension');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);

    try {
      const result = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });

      expect(result.status).toBe(1);

      const summary = JSON.parse(readFileSync(path.join(checkout, '.setup-summary.json'), 'utf8')) as Summary;
      expect(summary).toMatchObject({
        gpu_sm: '89',
        cuda_version: '12.4',
        config_ready: true,
        runtime_ready: false,
        install_success: false,
        install_ready: false,
        runtime_closure_ready: true,
        status: 'blocked',
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
      });
      expect(summary.missing_required).toContain('weight:models/ultrashape/ultrashape_v1.pt');
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('wheel');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
