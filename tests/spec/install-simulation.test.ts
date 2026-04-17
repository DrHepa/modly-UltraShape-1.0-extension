import { cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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
  'runtime/configs/infer_dit_refine.yaml',
  'runtime/patches/README.md',
  'runtime/vendor/ultrashape_runtime/__init__.py',
  'runtime/vendor/ultrashape_runtime/local_runner.py',
  'runtime/vendor/ultrashape_runtime/pipelines.py',
  'runtime/vendor/ultrashape_runtime/preprocessors.py',
  'runtime/vendor/ultrashape_runtime/rembg.py',
  'runtime/vendor/ultrashape_runtime/surface_loaders.py',
  'runtime/vendor/ultrashape_runtime/schedulers.py',
  'runtime/vendor/ultrashape_runtime/utils/__init__.py',
  'runtime/vendor/ultrashape_runtime/utils/checkpoint.py',
  'runtime/vendor/ultrashape_runtime/utils/mesh.py',
  'runtime/vendor/ultrashape_runtime/utils/tensors.py',
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
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
];

type Readiness = {
  status: 'ready' | 'degraded' | 'blocked';
  backend: string;
  weights_ready: boolean;
  required_imports_ok: boolean;
  missing_required: string[];
  missing_optional: string[];
  missing_conditional: string[];
  missing_degradable: string[];
  expected_weights: string[];
};

type HfTrace = {
  api: string;
  repo_id: string;
  filename: string;
  revision: string | null;
  token: string | null;
};

type NativeStageSummary = {
  attempted: boolean;
  status: 'ready' | 'degraded' | 'blocked' | 'pending' | 'skipped';
  required?: boolean;
  degradable?: boolean;
  source?: string;
  pinned_ref?: string;
  build_isolation?: boolean;
  commands?: string[];
  import_smoke_missing?: string[];
  failure_message?: string | null;
};

type SetupSummary = {
  torch_profile: string;
  runtime_layout_version: string;
  install_success: boolean;
  attempted_weight_source_kinds: string[];
  resolved_weight_source_kind: string;
  weight_source_repo_id: string;
  weight_source_filename: string;
  native_install_contract: {
    order: string[];
    cubvh_required: boolean;
    flash_attn_optional: boolean;
  };
  native_install: {
    core: NativeStageSummary & { installed_modules?: string[] };
    cubvh: NativeStageSummary;
    flash_attn: NativeStageSummary;
  };
};

function expectBinaryGlb(path: string) {
  const payload = readFileSync(path);
  expect(payload.subarray(0, 4).toString('ascii')).toBe('glTF');
  expect(payload.length).toBeGreaterThanOrEqual(24);
}

function checkpointBundleText() {
  return JSON.stringify({
    format: 'ultrashape-checkpoint-bundle/v1',
    vae: { tensors: { latent_basis: [0.11, 0.33, 0.55, 0.77] } },
    dit: { tensors: { attention_bias: [0.21, 0.41, 0.61, 0.81] } },
    conditioner: { tensors: { mask_bias: [0.14, 0.24, 0.64, 0.74] } },
  });
}

function expectedProcessorOutcome(readiness: Readiness): 'done' | 'WEIGHTS_MISSING' | 'DEPENDENCY_MISSING' | 'LOCAL_RUNTIME_UNAVAILABLE' {
  if (!readiness.weights_ready) {
    return 'WEIGHTS_MISSING';
  }

  if (!readiness.required_imports_ok) {
    return 'DEPENDENCY_MISSING';
  }

  if (readiness.status === 'blocked' || readiness.backend !== 'local') {
    return 'LOCAL_RUNTIME_UNAVAILABLE';
  }

  return 'done';
}

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

function stageRequiredWeight(installDir: string) {
  const weightPath = resolve(installDir, 'models/ultrashape/ultrashape_v1.pt');
  mkdirSync(resolve(installDir, 'models/ultrashape'), { recursive: true });
  writeFileSync(weightPath, checkpointBundleText());
  return weightPath;
}

function buildSetupEnv(extra: NodeJS.ProcessEnv = {}) {
  return {
    ...process.env,
    ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
    ...extra,
  };
}

describe('UltraShape Python install surface', () => {
  it('matches the expected GitHub-extracted payload layout for the Modly extensions directory', () => {
    const simulation = copyInstallSurface();

    try {
      for (const relativePath of installSurfacePaths) {
        expect(existsSync(resolve(simulation.installDir, relativePath)), `${relativePath} should be copied`).toBe(true);
      }

      expectBinaryGlb(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb'));
      expectBinaryGlb(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb'));

      const fixtureRequest = JSON.parse(readFileSync(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/request.json'), 'utf8')) as {
        params: {
          backend: string;
          output_format: string;
        };
      };
      expect(fixtureRequest.params.backend).toBe('local');
      expect(fixtureRequest.params.output_format).toBe('glb');

      expect(existsSync(resolve(simulation.installDir, 'package.json'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'processor.js'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'runtime/modly'))).toBe(false);
    } finally {
      simulation.cleanup();
    }
  });

  it('uses the default HF source for extracted-root setup when local sources are absent', () => {
    const simulation = copyInstallSurface();
    const hfTracePath = resolve(simulation.installDir, '.hf-download-trace.json');

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
        env: buildSetupEnv({
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: checkpointBundleText(),
          ULTRASHAPE_SETUP_TEST_HF_TRACE_PATH: hfTracePath,
        }),
      });

      expect(outcome.status).toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, '.setup-summary.json'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, '.runtime-readiness.json'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/configs/infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/__init__.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/local_runner.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/denoisers/dit_mask.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/autoencoders/surface_extractors.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/.locks'))).toBe(true);

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as SetupSummary;
      expect(summary.torch_profile).toBe('linux-arm64-cu128-sm90+');
      expect(summary.runtime_layout_version).toBe('1');
      expect(summary.install_success).toBe(true);
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir', 'repo-local', 'hf-default']);
      expect(summary.resolved_weight_source_kind).toBe('hf-default');
      expect(summary.weight_source_repo_id).toBe('infinith/UltraShape');
      expect(summary.weight_source_filename).toBe('ultrashape_v1.pt');
      expect(summary.native_install_contract).toEqual({
        order: ['core', 'cubvh', 'flash_attn'],
        cubvh_required: true,
        flash_attn_optional: true,
      });
      expect(summary.native_install).toMatchObject({
        core: {
          attempted: true,
          status: 'ready',
        },
        cubvh: {
          attempted: true,
          required: true,
          status: 'ready',
          source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
          pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
          build_isolation: false,
          import_smoke_missing: [],
          failure_message: null,
        },
        flash_attn: {
          attempted: true,
          required: false,
          degradable: true,
          status: 'ready',
          build_isolation: false,
          import_smoke_missing: [],
          failure_message: null,
        },
      });
      expect(summary.native_install.cubvh.commands).toEqual([
        expect.stringContaining('--no-build-isolation'),
      ]);
      expect(summary.native_install.flash_attn.commands).toEqual([
        expect.stringContaining('--no-build-isolation flash-attn'),
      ]);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness & {
        install_success: boolean;
        failure_code: string | null;
        attempted_weight_source_kinds: string[];
        resolved_weight_source_kind: string;
      };
      expect(readiness.install_success).toBe(true);
      expect(readiness.failure_code).toBe(null);
      expect(readiness.status).toBe('ready');
      expect(readiness.backend).toBe('local');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.attempted_weight_source_kinds).toEqual(['ext-dir', 'repo-local', 'hf-default']);
      expect(readiness.resolved_weight_source_kind).toBe('hf-default');
      expect(readFileSync(resolve(simulation.installDir, 'models/ultrashape/ultrashape_v1.pt'), 'utf8')).toBe(checkpointBundleText());

      const hfTrace = JSON.parse(readFileSync(hfTracePath, 'utf8')) as HfTrace;
      expect(hfTrace).toEqual({
        api: 'hf_hub_download',
        repo_id: 'infinith/UltraShape',
        filename: 'ultrashape_v1.pt',
        revision: null,
        token: null,
      });
    } finally {
      simulation.cleanup();
    }
  });

  it('prefers payload HF overrides over env HF defaults and keeps the download seam single-file', () => {
    const simulation = copyInstallSurface();
    const hfTracePath = resolve(simulation.installDir, '.hf-download-trace.json');

    try {
      const outcome = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: '90',
        weight_repo_id: 'payload/UltraShape',
        weight_repo_revision: 'payload-main',
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        env: buildSetupEnv({
          ULTRASHAPE_WEIGHT_REPO_ID: 'env/UltraShape',
          ULTRASHAPE_WEIGHT_REPO_REVISION: 'env-main',
          ULTRASHAPE_WEIGHT_HF_TOKEN: 'token-from-env',
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: checkpointBundleText(),
          ULTRASHAPE_SETUP_TEST_HF_TRACE_PATH: hfTracePath,
        }),
      });

      expect(outcome.status).toBe(0);
      expect(readFileSync(resolve(simulation.installDir, 'models/ultrashape/ultrashape_v1.pt'), 'utf8')).toBe(checkpointBundleText());

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as {
        attempted_weight_source_kinds: string[];
        resolved_weight_source_kind: string;
        weight_source_repo_id: string;
        weight_source_filename: string;
        weight_source_revision: string;
        weight_source_auth_used: boolean;
      };
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir', 'repo-local', 'hf-override']);
      expect(summary.resolved_weight_source_kind).toBe('hf-override');
      expect(summary.weight_source_repo_id).toBe('payload/UltraShape');
      expect(summary.weight_source_filename).toBe('ultrashape_v1.pt');
      expect(summary.weight_source_revision).toBe('payload-main');
      expect(summary.weight_source_auth_used).toBe(true);

      const hfTrace = JSON.parse(readFileSync(hfTracePath, 'utf8')) as HfTrace;
      expect(hfTrace).toEqual({
        api: 'hf_hub_download',
        repo_id: 'payload/UltraShape',
        filename: 'ultrashape_v1.pt',
        revision: 'payload-main',
        token: 'token-from-env',
      });
    } finally {
      simulation.cleanup();
    }
  });

  it('keeps manifest entry, setup contract, and processor smoke aligned inside the copied payload when the required weight is staged', () => {
    const simulation = copyInstallSurface();

    try {
      const manifest = JSON.parse(readFileSync(resolve(simulation.installDir, 'manifest.json'), 'utf8')) as {
        entry: string;
      };
      expect(manifest.entry).toBe('processor.py');
      const sourceWeight = stageRequiredWeight(resolve(simulation.installDir, '..', 'weight-cache'));

      const outcome = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: '90',
        required_weight_path: sourceWeight,
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        env: buildSetupEnv(),
      });

      expect(outcome.status).toBe(0);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect(readiness.status).toBe('ready');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);

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

      const expectedOutcome = expectedProcessorOutcome(readiness);
      if (expectedOutcome === 'done') {
        const smokeOutputPath = resolve(simulation.installDir, 'smoke-output/refined.glb');
        expect(events.at(-1)).toEqual({
          type: 'done',
          result: {
            filePath: smokeOutputPath,
          },
        });
        expect(existsSync(smokeOutputPath)).toBe(true);
        expectBinaryGlb(smokeOutputPath);
        expect(readFileSync(smokeOutputPath)).not.toEqual(
          readFileSync(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb')),
        );
      } else {
        expect(events.at(-1)).toEqual({
          type: 'error',
          message: expect.stringContaining(expectedOutcome),
          code: expectedOutcome,
        });
      }
    } finally {
      simulation.cleanup();
    }
  });

  it('keeps copied-payload setup successful but degraded when only flash_attn is absent', () => {
    const simulation = copyInstallSurface();
    const sourceWeight = stageRequiredWeight(resolve(simulation.installDir, '..', 'weight-cache'));

    try {
      const outcome = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: '90',
        required_weight_path: sourceWeight,
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        env: buildSetupEnv({
          ULTRASHAPE_SETUP_TEST_FLASH_ATTN_STAGE_FAIL: 'import',
        }),
      });

      expect(outcome.status).toBe(0);
      expect(`${outcome.stderr}${outcome.stdout}`).toContain(
        'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
      );

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as SetupSummary;
      expect(summary.install_success).toBe(true);
      expect(summary.native_install_contract).toEqual({
        order: ['core', 'cubvh', 'flash_attn'],
        cubvh_required: true,
        flash_attn_optional: true,
      });
      expect(summary.native_install).toMatchObject({
        core: {
          attempted: true,
          status: 'ready',
        },
        cubvh: {
          attempted: true,
          required: true,
          status: 'ready',
          source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
          pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
          build_isolation: false,
          import_smoke_missing: [],
          failure_message: null,
        },
        flash_attn: {
          attempted: true,
          required: false,
          degradable: true,
          status: 'degraded',
          build_isolation: false,
          import_smoke_missing: ['flash_attn'],
          failure_message: 'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
        },
      });

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness & {
        install_success: boolean;
        failure_code: string | null;
      };
      expect(readiness.install_success).toBe(true);
      expect(readiness.failure_code).toBe(null);
      expect(readiness.status).toBe('degraded');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual(['flash_attn']);
      expect(readiness.missing_optional).toEqual(['flash_attn']);
    } finally {
      simulation.cleanup();
    }
  });
});
