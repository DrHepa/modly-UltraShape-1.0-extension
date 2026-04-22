import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { copyInstallSurface } from './install-test-helpers.js';

type Readiness = {
  backend: string;
  checkpoint: string;
  config_path: string;
  cuda_version?: string | null;
  ext_dir: string;
  gpu_sm?: string | null;
  install_ready: boolean;
  install_success: boolean;
  missing_required: string[];
  missing_optional: string[];
  native_install?: Record<string, unknown>;
  required_imports_ok: boolean;
  runtime_closure_ready: boolean;
  status: string;
  venv_dir?: string;
  vendor_path: string;
  weights_ready: boolean;
};

function runSetup(
  cwd: string,
  options: {
    extDir: string;
    pythonExe: string;
    payload?: Record<string, unknown>;
    env?: NodeJS.ProcessEnv;
  },
) {
  const args = ['-S', 'setup.py', '--ext-dir', options.extDir, '--python-exe', options.pythonExe];
  if (options.payload) {
    args.push(JSON.stringify(options.payload));
  }

  return spawnSync('python3', args, {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...options.env,
    },
  });
}

function readReadiness(extDir: string) {
  return JSON.parse(readFileSync(path.join(extDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
}

function readSetupSummary(extDir: string) {
  return readFileSync(path.join(extDir, '.setup-summary.json'), 'utf8');
}

describe('setup.py install truth', () => {
  it('creates a venv, installs dependencies, acquires weights, and stays honest when payload host facts are absent', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-contract-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);
    const pythonExe = '/opt/modly/python/bin/python3';

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe,
        payload: {},
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: 'stub-weight',
          CUDA_HOME: '/usr/local/cuda-13.0',
          CUDA_PATH: '/usr/local/cuda-13.0',
          PATH: `/usr/local/cuda-13.0/bin:${process.env.PATH ?? ''}`,
          LD_LIBRARY_PATH: `/usr/local/cuda-13.0/lib64:${process.env.LD_LIBRARY_PATH ?? ''}`,
          LIBRARY_PATH: `/usr/local/cuda-13.0/lib64:${process.env.LIBRARY_PATH ?? ''}`,
        },
      });

      expect(result.status).toBe(0);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        backend: 'local',
        python_exe: pythonExe,
        ext_dir: checkout,
        gpu_sm: null,
        cuda_version: null,
        config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
        required_imports_ok: true,
        weights_ready: true,
        install_success: true,
        install_ready: true,
        runtime_closure_ready: true,
        status: 'ready',
        venv_dir: path.join(checkout, 'venv'),
      });
      expect(readiness.missing_required).toEqual([]);
      expect(existsSync(path.join(checkout, 'venv', 'bin', 'python'))).toBe(true);
      expect(existsSync(path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'))).toBe(true);
      expect(readiness.native_install).toMatchObject({
        cubvh: expect.objectContaining({
          status: 'ready',
          torch_cuda_profile: 'cu128',
          expected_cuda_home: '/usr/local/cuda-12.8',
          selected_cuda_home: '/usr/local/cuda-12.8',
          toolkit_pinned: true,
          env_overrides: expect.objectContaining({
            CUDA_HOME: '/usr/local/cuda-12.8',
            CUDA_PATH: '/usr/local/cuda-12.8',
            PATH: expect.stringMatching(/^\/usr\/local\/cuda-12\.8\/bin:/),
            LD_LIBRARY_PATH: expect.stringMatching(/^\/usr\/local\/cuda-12\.8\/lib64:/),
            LIBRARY_PATH: expect.stringMatching(/^\/usr\/local\/cuda-12\.8\/lib64:/),
          }),
        }),
      });
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).not.toContain('fallback');
      expect(result.stdout.toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).not.toContain('fallback');
      expect(JSON.stringify(readiness).toLowerCase()).not.toContain('hunyuan');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).not.toContain('fallback');
      expect(readSetupSummary(checkout).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('keeps readiness blocked and public metadata honest when required native cubvh prerequisites are missing', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-ready-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);
    const pythonExe = '/opt/modly/python/bin/python3';

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe,
        payload: {},
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING: 'compiler',
        },
      });

      expect(result.status).toBe(1);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        backend: 'local',
        python_exe: pythonExe,
        ext_dir: checkout,
        gpu_sm: null,
        cuda_version: null,
        config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        vendor_path: path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'),
        required_imports_ok: false,
        weights_ready: false,
        install_success: false,
        install_ready: false,
        runtime_closure_ready: true,
        status: 'blocked',
        venv_dir: path.join(checkout, 'venv'),
      });
      expect(readiness.missing_required).toContain('native-stage:cubvh');
      expect(existsSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'venv', 'bin', 'python'))).toBe(true);
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).not.toContain('fallback');
      expect(result.stdout.toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).not.toContain('fallback');
      expect(JSON.stringify(readiness).toLowerCase()).not.toContain('hunyuan');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).not.toContain('fallback');
      expect(readSetupSummary(checkout).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
