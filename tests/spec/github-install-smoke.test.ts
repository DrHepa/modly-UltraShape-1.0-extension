import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { copyInstallSurface } from './install-test-helpers.js';

type Summary = {
  cuda_version?: string | null;
  config_ready: boolean;
  dependency_install?: Record<string, unknown>;
  gpu_sm?: string | null;
  host_facts?: Record<string, unknown>;
  install_ready: boolean;
  install_success: boolean;
  missing_optional?: string[];
  missing_required: string[];
  native_install?: Record<string, unknown>;
  runtime_ready: boolean;
  runtime_closure_ready: boolean;
  runtime_modes?: Record<string, unknown>;
  status: string;
  venv_dir?: string;
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
  it('creates a real install footprint from a copied repository checkout', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-github-install-'));
    const checkout = path.join(sandbox, 'modly-UltraShape-1.0-extension');
    copyInstallSurface(checkout);

    try {
      const result = runSetup(checkout, checkout, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
        ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
        ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: 'stub-weight',
      });

      expect(result.status).toBe(0);
      expect(existsSync(path.join(checkout, '.setup-summary.json'))).toBe(true);
      expect(existsSync(path.join(checkout, '.runtime-readiness.json'))).toBe(true);
      expect(existsSync(path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'venv', 'bin', 'python'))).toBe(true);
      expect(existsSync(path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'))).toBe(true);
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
        runtime_modes: {
          selection: 'portable-only',
          requested: 'auto',
          active: 'portable',
          real: {
            available: false,
            adapter: 'ultrashape_runtime.real_mode.run_real_refine_pipeline',
          },
          portable: {
            available: true,
          },
        },
        status: 'degraded',
        missing_required: [],
        venv_dir: path.join(checkout, 'venv'),
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
      });
      expect(summary.missing_optional).toEqual(['import:flash_attn']);
      expect(summary.host_facts).toMatchObject({ platform: 'linux', machine: 'aarch64' });
      expect(summary.dependency_install).toBeTruthy();
      expect(summary.native_install).toMatchObject({ cubvh: expect.objectContaining({ status: 'ready' }) });
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps copied checkout installs blocked when the required native cubvh prerequisites are absent', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-github-blocked-'));
    const checkout = path.join(sandbox, 'modly-UltraShape-1.0-extension');
    copyInstallSurface(checkout);

    try {
      const result = runSetup(checkout, checkout, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
        ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
        ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING: 'compiler',
      });

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
        runtime_modes: {
          selection: 'blocked',
          active: null,
          portable: {
            available: false,
          },
        },
        status: 'blocked',
        venv_dir: path.join(checkout, 'venv'),
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
      });
      expect(summary.missing_required).toContain('native-stage:cubvh');
      expect(summary.native_install).toMatchObject({
        cubvh: expect.objectContaining({ status: 'blocked' }),
      });
      expect(JSON.stringify(summary).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
