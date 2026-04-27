import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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
  runtime_modes?: Record<string, unknown>;
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

const FLASH_ATTN_SKIP_MESSAGE =
  'flash_attn stage skipped on Linux ARM64 host; continuing with degraded PyTorch SDPA fallback.';

function createFakeUpstreamCheckout(root: string) {
  const checkout = path.join(root, 'fake-ultrashape-checkout');
  mkdirSync(path.join(checkout, 'scripts'), { recursive: true });
  mkdirSync(path.join(checkout, 'configs'), { recursive: true });
  mkdirSync(path.join(checkout, 'ultrashape'), { recursive: true });
  writeFileSync(path.join(checkout, 'LICENSE'), 'Fake UltraShape checkout for setup readiness contract tests.\n', 'utf8');
  writeFileSync(path.join(checkout, 'scripts', '__init__.py'), '', 'utf8');
  writeFileSync(path.join(checkout, 'scripts', 'infer_dit_refine.py'), 'def run_inference(args):\n    return None\n', 'utf8');
  writeFileSync(path.join(checkout, 'configs', 'infer_dit_refine.yaml'), 'runtime:\n  fake: true\n', 'utf8');
  writeFileSync(path.join(checkout, 'ultrashape', '__init__.py'), '', 'utf8');
  return checkout;
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
        status: 'degraded',
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
        venv_dir: path.join(checkout, 'venv'),
      });
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual(['import:flash_attn']);
      expect(readiness.runtime_modes?.selection).toBe('portable-only');
      expect(readiness.runtime_modes?.real).toMatchObject({
        available: false,
        blockers: expect.arrayContaining(['dependency:flash_attn']),
      });
      expect(JSON.stringify(readiness.runtime_modes ?? {})).toContain('Portable fallback');
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
          diagnostics: expect.objectContaining({
            cubvh_callables: {
              sparse_marching_cubes: true,
              sparse_marching_cubes_cpu: true,
            },
          }),
          self_tests: {
            cuda: expect.objectContaining({
              available: true,
              executable: true,
              skipped: false,
              launch_blocking: '1',
              input: expect.objectContaining({ cell_count: 1 }),
            }),
            cpu: expect.objectContaining({
              available: true,
              executable: true,
              skipped: false,
              input: expect.objectContaining({ cell_count: 1 }),
            }),
          },
          rebuild_guidance: expect.objectContaining({
            recommended: false,
          }),
        }),
        flash_attn: {
          attempted: false,
          required: false,
          degradable: true,
          status: 'degraded',
          commands: [],
          import_smoke_missing: ['flash_attn'],
          failure_message:
            FLASH_ATTN_SKIP_MESSAGE,
          skip_reason:
            FLASH_ATTN_SKIP_MESSAGE,
        },
      });
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).toContain(FLASH_ATTN_SKIP_MESSAGE);
      expect(result.stdout.toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).toContain(FLASH_ATTN_SKIP_MESSAGE);
      expect(JSON.stringify(readiness).toLowerCase()).not.toContain('hunyuan');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).toContain(FLASH_ATTN_SKIP_MESSAGE);
      expect(readSetupSummary(checkout).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('does not mark cubvh executable readiness from import-only success', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-cubvh-import-only-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe: '/opt/modly/python/bin/python3',
        payload: {},
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_CUBVH_STUB_SCENARIO: 'import-only',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: 'stub-weight',
        },
      });

      expect(result.status).toBe(1);
      const readiness = readReadiness(checkout);
      const summary = JSON.parse(readSetupSummary(checkout)) as Readiness;
      expect(readiness).toMatchObject({
        install_success: false,
        install_ready: false,
        required_imports_ok: false,
        status: 'blocked',
        native_install: {
          cubvh: expect.objectContaining({
            status: 'blocked',
            import_smoke_missing: [],
            diagnostics: expect.objectContaining({
              cubvh_callables: {
                sparse_marching_cubes: false,
                sparse_marching_cubes_cpu: false,
              },
            }),
            self_tests: {
              cuda: expect.objectContaining({
                available: false,
                executable: false,
                skipped: false,
                error_class: 'missing_callable',
                input: expect.objectContaining({ cell_count: 1 }),
              }),
              cpu: expect.objectContaining({
                available: false,
                executable: false,
                skipped: true,
                error_class: 'cpu_fallback_unavailable',
              }),
            },
            rebuild_guidance: expect.objectContaining({
              recommended: true,
              reason: 'force_source_compile_or_match_torch_cuda_arch',
            }),
          }),
        },
      });
      expect(readiness.missing_required).toContain('native-stage:cubvh');
      expect(summary.native_install).toMatchObject(readiness.native_install ?? {});
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('reports real-available when explicit checkout, flash_attn, config, checkpoint, and smoke imports are ready', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-real-available-'));
    const checkout = path.join(sandbox, 'repo');
    const upstreamCheckout = createFakeUpstreamCheckout(sandbox);
    copyInstallSurface(checkout);

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe: '/opt/modly/python/bin/python3',
        payload: { ultrashape_checkout_path: upstreamCheckout },
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: 'stub-weight',
          ULTRASHAPE_SETUP_TEST_FORCE_FLASH_ATTN_READY: '1',
        },
      });

      expect(result.status).toBe(0);

      const readiness = readReadiness(checkout);
      const summary = JSON.parse(readSetupSummary(checkout)) as Readiness;
      expect(readiness).toMatchObject({
        install_success: true,
        install_ready: true,
        status: 'ready',
        runtime_modes: {
          selection: 'real-available',
          requested: 'auto',
          active: 'real',
          real: {
            available: true,
            source: 'checkout',
            checkout_path: upstreamCheckout,
            entrypoint: 'scripts.infer_dit_refine.run_inference',
            blockers: [],
            config: { available: true },
            checkpoint: { available: true },
            dependencies: {
              flash_attn: { available: true, required: true },
            },
          },
          portable: {
            available: true,
            authoritative: false,
          },
        },
      });
      expect(readiness.missing_optional).toEqual([]);
      expect(summary.runtime_modes).toMatchObject(readiness.runtime_modes ?? {});
      expect(JSON.stringify(summary.runtime_modes ?? {})).toContain('torch_cuda_profile');
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
        runtime_modes: {
          selection: 'blocked',
          requested: 'auto',
          active: null,
          real: {
            available: false,
          },
          portable: {
            available: false,
          },
        },
        venv_dir: path.join(checkout, 'venv'),
      });
      expect(readiness.missing_required).toContain('native-stage:cubvh');
      expect(readiness.native_install).toMatchObject({
        cubvh: expect.objectContaining({
          diagnostics: expect.objectContaining({
            cuda_available: false,
            cuda_unavailable_reason: 'cubvh prerequisites blocked before CUDA metadata probing',
            cubvh_callables: {
              sparse_marching_cubes: false,
              sparse_marching_cubes_cpu: false,
            },
          }),
          self_tests: {
            cuda: expect.objectContaining({
              available: false,
              executable: false,
              skipped: true,
              error_class: 'cubvh_prerequisites_blocked',
              input: expect.objectContaining({ cell_count: 1 }),
            }),
            cpu: expect.objectContaining({
              available: false,
              executable: false,
              skipped: true,
              error_class: 'cpu_fallback_unavailable',
              input: expect.objectContaining({ cell_count: 1 }),
            }),
          },
          rebuild_guidance: expect.objectContaining({
            recommended: true,
            reason: 'install_cubvh_prerequisites_before_self_test',
          }),
        }),
      });
      expect(existsSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'venv', 'bin', 'python'))).toBe(true);
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).toContain('Portable fallback is blocked because required runtime prerequisites are missing.');
      expect(result.stdout.toLowerCase()).not.toContain('hunyuan');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).toContain('Portable fallback is blocked because required runtime prerequisites are missing.');
      expect(JSON.stringify(readiness).toLowerCase()).not.toContain('hunyuan');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).toContain('Portable fallback is blocked because required runtime prerequisites are missing.');
      expect(readSetupSummary(checkout).toLowerCase()).not.toContain('hunyuan');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('short-circuits dependency bootstrap when cubvh prerequisites are already blocked', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-short-circuit-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const result = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import setup',
            '',
            'def fail(name):',
            '    def _raise(*args, **kwargs):',
            '        del args, kwargs',
            '        raise AssertionError(f"{name} should not run when cubvh prerequisites are already blocked")',
            '    return _raise',
            '',
            'setup.bootstrap_packaging_tools = fail("bootstrap_packaging_tools")',
            'setup.install_core_dependencies = fail("install_core_dependencies")',
            'raise SystemExit(setup.main())',
          ].join('\n'),
          '--ext-dir',
          checkout,
          '--python-exe',
          '/opt/modly/python/bin/python3',
        ],
        {
          cwd: checkout,
          encoding: 'utf8',
          env: {
            ...process.env,
            ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
            ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
            ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
            ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING: 'compiler',
          },
        },
      );

      expect(result.status).toBe(1);
      expect(result.stderr).not.toContain('should not run when cubvh prerequisites are already blocked');
      const readiness = readReadiness(checkout);
      expect(readiness.native_install).toMatchObject({
        cubvh: expect.objectContaining({
          diagnostics: expect.objectContaining({
            cuda_unavailable_reason: 'cubvh prerequisites blocked before CUDA metadata probing',
          }),
          self_tests: {
            cuda: expect.objectContaining({ skipped: true, error_class: 'cubvh_prerequisites_blocked' }),
            cpu: expect.objectContaining({ skipped: true, error_class: 'cpu_fallback_unavailable' }),
          },
          rebuild_guidance: expect.objectContaining({ reason: 'install_cubvh_prerequisites_before_self_test' }),
        }),
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('blocks readiness when the staged vendored closure cannot import', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-runtime-import-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    writeFileSync(
      path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'),
      'raise ModuleNotFoundError("broken vendored runtime closure")\n',
      'utf8',
    );

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe: '/opt/modly/python/bin/python3',
        payload: {},
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: 'stub-weight',
        },
      });

      expect(result.status).toBe(1);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        required_imports_ok: false,
        weights_ready: true,
        install_success: false,
        install_ready: false,
        runtime_closure_ready: false,
        status: 'blocked',
        runtime_modes: {
          selection: 'blocked',
          active: null,
          portable: {
            available: false,
          },
        },
      });
      expect(readiness.missing_required).toContain('runtime-import:ultrashape_runtime.local_runner');
      expect(JSON.stringify(readiness)).toContain('broken vendored runtime closure');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('blocks readiness when the staged checkpoint is missing required subtrees', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-checkpoint-smoke-'));
    const checkout = path.join(sandbox, 'repo');
    const weightSourceRoot = path.join(sandbox, 'weight-source');
    copyInstallSurface(checkout);

    const checkpointPath = path.join(weightSourceRoot, 'broken-ultrashape.pt');
    mkdirSync(weightSourceRoot, { recursive: true });
    writeFileSync(
      checkpointPath,
      JSON.stringify({
        vae: { tensors: { weights: [0.1, 0.2, 0.3, 0.4] } },
        dit: { tensors: { weights: [0.5, 0.6, 0.7, 0.8] } },
      }),
      'utf8',
    );

    try {
      const result = runSetup(checkout, {
        extDir: checkout,
        pythonExe: '/opt/modly/python/bin/python3',
        payload: {},
        env: {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
          ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
          ULTRASHAPE_WEIGHT_SOURCE_PATH: checkpointPath,
        },
      });

      expect(result.status).toBe(1);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        required_imports_ok: true,
        weights_ready: false,
        install_success: false,
        install_ready: false,
        runtime_closure_ready: true,
        status: 'blocked',
        runtime_modes: {
          selection: 'portable-only',
          active: 'portable',
        },
      });
      expect(readiness.missing_required).toContain('checkpoint-subtree:conditioner');
      expect(JSON.stringify(readiness)).toContain('Required checkpoint subtrees are missing: conditioner.');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
