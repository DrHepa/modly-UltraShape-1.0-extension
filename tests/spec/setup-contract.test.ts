import { cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const setupPath = resolve(repoRoot, 'setup.py');

type NativeInstallContract = {
  order: string[];
  cubvh_required: boolean;
  flash_attn_optional: boolean;
};

type NativeInstallStage = {
  attempted: boolean;
  status: string;
  required?: boolean;
  degradable?: boolean;
  source?: string;
  pinned_ref?: string;
  build_isolation?: boolean;
  commands?: string[];
  cuda_toolkit?: {
    torch_requirement?: string;
    cuda_tag?: string | null;
    expected_version?: string | null;
    candidate_roots?: string[];
    root?: string | null;
    matched_profile?: boolean;
    environment_overrides?: Record<string, string>;
  };
  missing_prerequisites?: string[];
  detected_prerequisites?: Record<string, unknown>;
  import_smoke_missing?: string[];
  failure_message?: string | null;
};

type NativeInstallSummary = {
  core: NativeInstallStage;
  cubvh: NativeInstallStage;
  flash_attn: NativeInstallStage;
};

type WeightSourceArtifact = {
  attempted_weight_source_kinds?: string[];
  attempted_weight_sources?: string[];
  resolved_weight_source_kind?: string | null;
  resolved_weight_source?: string | null;
  weight_source_repo_id?: string | null;
  weight_source_filename?: string | null;
  weight_source_revision?: string | null;
  weight_source_auth_used?: boolean | null;
  weight_source_failure_classification?: string | null;
  weight_source_error_class?: string | null;
  weight_source_error_message?: string | null;
};

type SetupStatusArtifact = {
  install_success?: boolean;
  failure_stage?: string | null;
  failure_code?: string | null;
};

type DependencySummary = {
  required: string[];
  conditional: string[];
  degradable: string[];
};

type InstallSurfaceSummary = {
  layout: string;
  entry: string;
  backend_modes: string[];
  resolved_backend: string;
  output_formats: string[];
  remote_hybrid_supported: boolean;
};

type SetupSummaryWithNative = SetupStatusArtifact & WeightSourceArtifact & {
  native_install_contract: NativeInstallContract;
  native_install: NativeInstallSummary;
  install_surface?: InstallSurfaceSummary;
  dependencies?: DependencySummary;
  missing_required?: string[];
};

type SetupReadiness = SetupStatusArtifact & WeightSourceArtifact & {
  status: string;
  required_imports_ok?: boolean;
  missing_required?: string[];
  missing_degradable?: string[];
  missing_optional?: string[];
};

const extractedPayloadPaths = [
  'manifest.json',
  'setup.py',
  'processor.py',
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
] as const;

function runSetup(argument: Record<string, unknown>) {
  return spawnSync('python3', [setupPath, JSON.stringify(argument)], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
    },
  });
}

function copyFileOrDirectory(relativePath: string, installDir: string) {
  cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
}

function runSetupWithEnv(argument: Record<string, unknown>, env: NodeJS.ProcessEnv) {
  return spawnSync('python3', [setupPath, JSON.stringify(argument)], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

function readSetupArtifacts<
  TSummary extends SetupSummaryWithNative = SetupSummaryWithNative,
  TReadiness extends SetupReadiness = SetupReadiness,
>(installDir: string) {
  const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as TSummary;
  const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as TReadiness;
  return { summary, readiness };
}

describe('UltraShape setup.py contract', () => {
  it('fails fast with explicit blocked cubvh metadata when Linux ARM64 prerequisites are missing', () => {
    const scenarios = [
      {
        missing: 'git',
        expectedMissing: ['git'],
        expectedDetected: {
          host: 'linux-arm64',
          git: false,
        },
      },
      {
        missing: 'compiler',
        expectedMissing: ['compiler-toolchain'],
        expectedDetected: {
          host: 'linux-arm64',
          git: true,
          compiler: null,
        },
      },
      {
        missing: 'cuda',
        expectedMissing: ['cuda-build-tooling'],
        expectedDetected: {
          host: 'linux-arm64',
          git: true,
          compiler: 'g++',
          cuda: null,
        },
      },
      {
        missing: 'eigen',
        expectedMissing: ['eigen-headers'],
        expectedDetected: {
          host: 'linux-arm64',
          git: true,
          compiler: 'g++',
          cuda: '/usr/local/cuda-12.8',
          eigen: null,
        },
      },
    ] as const;

    for (const scenario of scenarios) {
      const root = mkdtempSync(join(tmpdir(), `ultrashape-setup-cubvh-prereq-${scenario.missing}-`));
      const installDir = join(root, 'extension-root');
      const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
      mkdirSync(join(root, 'download-cache'), { recursive: true });
      writeFileSync(sourceWeight, 'copied-weight');

      try {
        const outcome = runSetupWithEnv({
          python_exe: 'python3',
          ext_dir: installDir,
          gpu_sm: 90,
          cuda_version: 12.8,
          required_weight_path: sourceWeight,
        }, {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING: scenario.missing,
        });

        expect(outcome.status, `${scenario.missing} should block setup`).not.toBe(0);
        expect(`${outcome.stderr}${outcome.stdout}`).toContain(
          'cubvh source build requires Linux ARM64 with git, compiler toolchain, CUDA build tooling, and Eigen headers available.',
        );

        const { summary, readiness } = readSetupArtifacts(installDir);
        expect(summary.native_install_contract).toEqual({
          order: ['core', 'cubvh', 'flash_attn'],
          cubvh_required: true,
          flash_attn_optional: true,
        });
        expect(summary.install_surface).toEqual({
          layout: 'repo-root-python-only',
          entry: 'processor.py',
          backend_modes: ['auto', 'local'],
          resolved_backend: 'local',
          output_formats: ['glb'],
          remote_hybrid_supported: false,
        });
        expect(summary.native_install).toMatchObject({
          core: {
            attempted: true,
            status: 'ready',
          },
          cubvh: {
            attempted: false,
            required: true,
            status: 'blocked',
            source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
            pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
            build_isolation: false,
            missing_prerequisites: scenario.expectedMissing,
            detected_prerequisites: scenario.expectedDetected,
            failure_message: 'cubvh source build requires Linux ARM64 with git, compiler toolchain, CUDA build tooling, and Eigen headers available.',
          },
          flash_attn: {
            attempted: false,
            status: 'pending',
          },
        });
        expect(readiness.status).toBe('blocked');
        expect(readiness.install_success).toBe(false);
        expect(readiness.required_imports_ok).toBe(true);
        expect(readiness.failure_stage).toBe('cubvh-prerequisites');
        expect(readiness.failure_code).toBe('CUBVH_PREREQUISITES_MISSING');
        expect(readiness.missing_required).toEqual(['native-stage:cubvh']);
        expect(readiness.missing_degradable).toEqual([]);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    }
  });

  it('fails install when the required weight is missing but still writes truthful failure metadata', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-'));
    const installDir = join(root, 'extension-root');
    const firstRun = runSetupWithEnv({
      python_exe: 'python3',
      ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
    }, {
      ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
    });

    try {
      expect(firstRun.status).not.toBe(0);
      expect(existsSync(join(installDir, 'venv'))).toBe(true);
      expect(readFileSync(join(installDir, 'venv', 'pyvenv.cfg'), 'utf8')).toContain('home');
      expect(existsSync(join(installDir, '.setup-summary.json'))).toBe(true);
      expect(existsSync(join(installDir, '.runtime-readiness.json'))).toBe(true);
      expect(existsSync(join(installDir, 'runtime', 'configs', 'infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(join(installDir, 'runtime', 'ultrashape_runtime', '__init__.py'))).toBe(true);
      expect(existsSync(join(installDir, 'runtime', '.locks'))).toBe(true);

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        torch_profile: string;
        runtime_layout_version: string;
        install_success: boolean;
        failure_stage: string;
        failure_code: string;
        dependencies: {
          required: string[];
          conditional: string[];
          degradable: string[];
        };
      } & SetupSummaryWithNative;
      expect(summary.torch_profile).toBe('linux-arm64-cu128-sm90+');
      expect(summary.runtime_layout_version).toBe('1');
      expect(summary.install_success).toBe(false);
      expect(summary.failure_stage).toBe('weight-validation');
      expect(summary.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(summary.dependencies.required).toContain('diffusers');
      expect(summary.dependencies.required).toContain('cubvh');
      expect(summary.dependencies.conditional).toEqual(['rembg', 'onnxruntime']);
      expect(summary.dependencies.degradable).toEqual(['flash_attn']);
      expect(summary.native_install_contract).toEqual({
        order: ['core', 'cubvh', 'flash_attn'],
        cubvh_required: true,
        flash_attn_optional: true,
      });
      expect(summary.native_install).toMatchObject({
        core: {
          attempted: true,
        },
        cubvh: {
          attempted: true,
          required: true,
          pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
          source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
          build_isolation: false,
          status: 'ready',
        },
        flash_attn: {
          attempted: true,
          required: false,
          degradable: true,
          status: 'ready',
        },
      });
      expect(summary.native_install.cubvh.commands).toEqual([
        expect.stringContaining('--no-build-isolation'),
      ]);

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        backend: string;
        mvp_scope: string;
        output_format: string;
        weights_ready: boolean;
        required_imports_ok: boolean;
        missing_required: string[];
        missing_optional: string[];
        expected_weights: string[];
        required_checkpoint_subtrees: string[];
        install_success: boolean;
        failure_stage: string;
        failure_code: string;
      };
      expect(readiness.backend).toBe('local');
      expect(readiness.mvp_scope).toBe('mc-only');
      expect(readiness.output_format).toBe('glb');
      expect(readiness.status).toBe('blocked');
      expect(readiness.weights_ready).toBe(false);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.required_checkpoint_subtrees).toEqual(['vae', 'dit', 'conditioner']);
      expect(readiness.install_success).toBe(false);
      expect(readiness.failure_stage).toBe('weight-validation');
      expect(readiness.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(readiness.missing_required).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);

      const sentinel = join(installDir, 'venv', 'sentinel.txt');
      writeFileSync(sentinel, 'keep-me');

      const secondRun = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
      });

      expect(secondRun.status).not.toBe(0);
      expect(existsSync(sentinel)).toBe(true);
      expect(readFileSync(sentinel, 'utf8')).toBe('keep-me');
      expect(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')).toContain('"status": "blocked"');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('succeeds when the required weight is already present in ext_dir', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-ready-'));
    const installDir = join(root, 'extension-root');
    const weightPath = join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(installDir, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(weightPath, 'test-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
      });

      expect(outcome.status).toBe(0);

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        install_success: boolean;
        attempted_weight_source_kinds: string[];
        attempted_weight_sources: string[];
        resolved_weight_source_kind: string;
        resolved_weight_source: string;
      } & SetupSummaryWithNative;
      expect(summary.install_success).toBe(true);
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir']);
      expect(summary.attempted_weight_sources).toEqual([weightPath]);
      expect(summary.resolved_weight_source_kind).toBe('ext-dir');
      expect(summary.resolved_weight_source).toBe(weightPath);
      expect(summary.native_install_contract).toEqual({
        order: ['core', 'cubvh', 'flash_attn'],
        cubvh_required: true,
        flash_attn_optional: true,
      });
      expect(summary.install_surface).toEqual({
        layout: 'repo-root-python-only',
        entry: 'processor.py',
        backend_modes: ['auto', 'local'],
        resolved_backend: 'local',
        output_formats: ['glb'],
        remote_hybrid_supported: false,
      });
      expect(summary.native_install).toMatchObject({
        core: {
          attempted: true,
          status: 'ready',
        },
        cubvh: {
          attempted: true,
          required: true,
          pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
          source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
          build_isolation: false,
          status: 'ready',
          commands: [
            expect.stringContaining('--no-build-isolation'),
          ],
        },
        flash_attn: {
          attempted: true,
          required: false,
          degradable: true,
          status: 'ready',
        },
      });

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        weights_ready: boolean;
        required_imports_ok: boolean;
        missing_required: string[];
      };
      expect(readiness.status).toBe('ready');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(existsSync(weightPath)).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('marks copied extracted payload installs as blocked when ultrashape_runtime import smoke cannot close over the copied root', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-copied-root-'));
    const installDir = join(root, 'modly-UltraShape-1.0-extension');
    const weightPath = join(root, 'weight-cache', 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'weight-cache', 'models', 'ultrashape'), { recursive: true });
    writeFileSync(weightPath, 'test-weight');

    for (const relativePath of extractedPayloadPaths) {
      copyFileOrDirectory(relativePath, installDir);
    }

    try {
      const setup = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        required_weight_path: weightPath,
      })], {
        cwd: installDir,
        encoding: 'utf8',
        env: {
          ...process.env,
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        },
      });

      expect(setup.status).not.toBe(0);

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.install_success).toBe(false);
      expect(summary.failure_stage).toBe('runtime-validation');
      expect(summary.failure_code).toBe('RUNTIME_LAYOUT_INCOMPLETE');
      expect(summary.missing_required).toEqual([
        'runtime/ultrashape_runtime/utils/voxelize.py',
        'import:ultrashape_runtime',
      ]);
      expect(readiness.status).toBe('blocked');
      expect(readiness.install_success).toBe(false);
      expect(readiness.required_imports_ok).toBe(false);
      expect(readiness.failure_stage).toBe('runtime-validation');
      expect(readiness.failure_code).toBe('RUNTIME_LAYOUT_INCOMPLETE');
      expect(readiness.missing_required).toEqual([
        'runtime/ultrashape_runtime/utils/voxelize.py',
        'import:ultrashape_runtime',
      ]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('pins cubvh to the CUDA toolkit derived from the selected torch profile even when host CUDA env points elsewhere', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-cuda-toolkit-'));
    const installDir = join(root, 'extension-root');
    const weightPath = join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(installDir, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(weightPath, 'test-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_CUDA_ROOTS: ['/usr/local/cuda-12.8', '/usr/local/cuda-13.0'].join(':'),
        CUDA_HOME: '/usr/local/cuda-13.0',
        CUDA_PATH: '/usr/local/cuda-13.0',
        PATH: ['/usr/local/cuda-13.0/bin', '/usr/bin'].join(':'),
        LD_LIBRARY_PATH: '/usr/local/cuda-13.0/lib64',
        LIBRARY_PATH: '/usr/local/cuda-13.0/lib64',
      });

      expect(outcome.status).toBe(0);

      const { summary } = readSetupArtifacts(installDir);
      expect(summary.native_install.cubvh.cuda_toolkit).toMatchObject({
        torch_requirement: 'torch==2.7.0+cu128',
        cuda_tag: '128',
        expected_version: '12.8',
        candidate_roots: ['/usr/local/cuda-12.8'],
        root: '/usr/local/cuda-12.8',
        matched_profile: true,
      });
      expect(summary.native_install.cubvh.cuda_toolkit?.environment_overrides).toMatchObject({
        CUDA_HOME: '/usr/local/cuda-12.8',
        CUDA_PATH: '/usr/local/cuda-12.8',
      });
      expect(summary.native_install.cubvh.cuda_toolkit?.environment_overrides?.PATH).toBe(
        ['/usr/local/cuda-12.8/bin', '/usr/local/cuda-13.0/bin', '/usr/bin'].join(':'),
      );
      expect(summary.native_install.cubvh.cuda_toolkit?.environment_overrides?.LD_LIBRARY_PATH).toBe(
        ['/usr/local/cuda-12.8/lib64', '/usr/local/cuda-12.8/lib', '/usr/local/cuda-13.0/lib64'].join(':'),
      );
      expect(summary.native_install.cubvh.cuda_toolkit?.environment_overrides?.LIBRARY_PATH).toBe(
        ['/usr/local/cuda-12.8/lib64', '/usr/local/cuda-12.8/lib', '/usr/local/cuda-13.0/lib64'].join(':'),
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('acquires the required weight from a local source, installs required deps, and records ready metadata truthfully', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-acquire-'));
    const installDir = join(root, 'extension-root');
    const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'download-cache'), { recursive: true });
    writeFileSync(sourceWeight, 'copied-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: sourceWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
      });

      expect(outcome.status).toBe(0);
      expect(readFileSync(join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'utf8')).toBe('copied-weight');

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        install_success: boolean;
        attempted_weight_sources: string[];
      } & SetupSummaryWithNative;
      expect(summary.install_success).toBe(true);
      expect(summary.attempted_weight_sources).toContain(sourceWeight);
      expect(summary.native_install.cubvh).toMatchObject({
        attempted: true,
        required: true,
        status: 'ready',
        source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
        pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
        build_isolation: false,
      });
      expect(summary.native_install.cubvh.commands).toEqual([
        expect.stringContaining('install --no-build-isolation git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f'),
      ]);

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        weights_ready: boolean;
        required_imports_ok: boolean;
        missing_required: string[];
        required_checkpoint_subtrees: string[];
      };
      expect(readiness.status).toBe('ready');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.required_checkpoint_subtrees).toEqual(['vae', 'dit', 'conditioner']);
      expect(readiness.missing_required).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('prefers payload required_weight_path over env-local and repo-local sources', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-payload-precedence-'));
    const installDir = join(root, 'extension-root');
    const payloadWeight = join(root, 'payload-cache', 'ultrashape_v1.pt');
    const envWeight = join(root, 'env-cache', 'ultrashape_v1.pt');
    const repoWeight = join(repoRoot, 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'payload-cache'), { recursive: true });
    mkdirSync(join(root, 'env-cache'), { recursive: true });
    mkdirSync(join(repoRoot, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(payloadWeight, 'payload-weight');
    writeFileSync(envWeight, 'env-weight');
    writeFileSync(repoWeight, 'repo-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: payloadWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_WEIGHT_SOURCE_PATH: envWeight,
      });

      expect(outcome.status).toBe(0);
      expect(readFileSync(join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'utf8')).toBe('payload-weight');

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        attempted_weight_source_kinds: string[];
        attempted_weight_sources: string[];
        resolved_weight_source_kind: string;
        resolved_weight_source: string;
      };
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir', 'required_weight_path']);
      expect(summary.attempted_weight_sources).toEqual([
        join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        payloadWeight,
      ]);
      expect(summary.resolved_weight_source_kind).toBe('required_weight_path');
      expect(summary.resolved_weight_source).toBe(payloadWeight);
    } finally {
      rmSync(join(repoRoot, 'models'), { recursive: true, force: true });
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('prefers env-local over repo-local when earlier local sources are absent', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-env-precedence-'));
    const installDir = join(root, 'extension-root');
    const envWeight = join(root, 'env-cache', 'ultrashape_v1.pt');
    const repoWeight = join(repoRoot, 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'env-cache'), { recursive: true });
    mkdirSync(join(repoRoot, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(envWeight, 'env-weight');
    writeFileSync(repoWeight, 'repo-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_WEIGHT_SOURCE_PATH: envWeight,
      });

      expect(outcome.status).toBe(0);
      expect(readFileSync(join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'utf8')).toBe('env-weight');

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        attempted_weight_source_kinds: string[];
        attempted_weight_sources: string[];
        resolved_weight_source_kind: string;
        resolved_weight_source: string;
      };
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir', 'env-local']);
      expect(summary.attempted_weight_sources).toEqual([
        join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        envWeight,
      ]);
      expect(summary.resolved_weight_source_kind).toBe('env-local');
      expect(summary.resolved_weight_source).toBe(envWeight);
    } finally {
      rmSync(join(repoRoot, 'models'), { recursive: true, force: true });
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('falls back to repo-local when earlier local sources are absent and records attempted ordering', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-repo-precedence-'));
    const installDir = join(root, 'extension-root');
    const repoWeight = join(repoRoot, 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(repoRoot, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(repoWeight, 'repo-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
      });

      expect(outcome.status).toBe(0);
      expect(readFileSync(join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'utf8')).toBe('repo-weight');

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        attempted_weight_source_kinds: string[];
        attempted_weight_sources: string[];
        resolved_weight_source_kind: string;
        resolved_weight_source: string;
      };
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir', 'repo-local']);
      expect(summary.attempted_weight_sources).toEqual([
        join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        repoWeight,
      ]);
      expect(summary.resolved_weight_source_kind).toBe('repo-local');
      expect(summary.resolved_weight_source).toBe(repoWeight);
    } finally {
      rmSync(join(repoRoot, 'models'), { recursive: true, force: true });
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('succeeds with degraded readiness when the optional flash_attn stage degrades', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-degraded-'));
    const installDir = join(root, 'extension-root');
    const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'download-cache'), { recursive: true });
    writeFileSync(sourceWeight, 'copied-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: sourceWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_FLASH_ATTN_STAGE_FAIL: 'install',
      });

      expect(outcome.status).toBe(0);

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        weights_ready: boolean;
        required_imports_ok: boolean;
        missing_required: string[];
        missing_optional: string[];
      };
      expect(readiness.status).toBe('degraded');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual(['flash_attn']);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails install when an effectively required real-refinement dependency is absent even if placeholder runtime could still run', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-real-required-'));
    const installDir = join(root, 'extension-root');
    const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'download-cache'), { recursive: true });
    writeFileSync(sourceWeight, 'copied-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: sourceWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_STUB_DEPS_MISSING: 'diffusers',
      });

      expect(outcome.status).not.toBe(0);

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.dependencies).toMatchObject({
        required: expect.arrayContaining(['diffusers', 'cubvh']),
        conditional: ['rembg', 'onnxruntime'],
        degradable: ['flash_attn'],
      });
      expect(readiness.status).toBe('blocked');
      expect(readiness.required_imports_ok).toBe(false);
      expect(readiness.missing_required).toContain('import:diffusers');
      expect(readiness.missing_optional).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails install when required dependency import smoke fails after the install attempt', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-import-smoke-'));
    const installDir = join(root, 'extension-root');
    const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'download-cache'), { recursive: true });
    writeFileSync(sourceWeight, 'copied-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: sourceWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_STUB_DEPS_MISSING: 'cv2',
      });

      expect(outcome.status).not.toBe(0);

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        install_success: boolean;
        failure_stage: string;
        failure_code: string;
        missing_required: string[];
      };
      expect(summary.install_success).toBe(false);
      expect(summary.failure_stage).toBe('required-import-smoke');
      expect(summary.failure_code).toBe('REQUIRED_IMPORT_SMOKE_FAILED');
      expect(summary.missing_required).toContain('import:cv2');

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        required_imports_ok: boolean;
        missing_required: string[];
        failure_stage: string;
        failure_code: string;
      };
      expect(readiness.status).toBe('blocked');
      expect(readiness.required_imports_ok).toBe(false);
      expect(readiness.missing_required).toContain('import:cv2');
      expect(readiness.failure_stage).toBe('required-import-smoke');
      expect(readiness.failure_code).toBe('REQUIRED_IMPORT_SMOKE_FAILED');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails with explicit cubvh import-smoke text when the required native stage cannot be imported after install', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-cubvh-import-smoke-'));
    const installDir = join(root, 'extension-root');
    const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'download-cache'), { recursive: true });
    writeFileSync(sourceWeight, 'copied-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: sourceWeight,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_CUBVH_IMPORT_FAIL: '1',
      });

      expect(outcome.status).not.toBe(0);
      expect(`${outcome.stderr}${outcome.stdout}`).toContain(
        'cubvh build completed but import smoke failed; local install cannot continue without cubvh.',
      );

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.native_install.cubvh).toMatchObject({
        attempted: true,
        required: true,
        status: 'blocked',
        source: 'git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f',
        pinned_ref: '7855c000f95e43742081060d869702b2b2b33d1f',
        build_isolation: false,
        import_smoke_missing: ['cubvh'],
        failure_message: 'cubvh build completed but import smoke failed; local install cannot continue without cubvh.',
      });
      expect(summary.native_install.cubvh.commands).toEqual([
        expect.stringContaining('install --no-build-isolation git+https://github.com/ashawkey/cubvh@7855c000f95e43742081060d869702b2b2b33d1f'),
      ]);
      expect(readiness.status).toBe('blocked');
      expect(readiness.required_imports_ok).toBe(false);
      expect(readiness.failure_stage).toBe('cubvh-import-smoke');
      expect(readiness.missing_required).toEqual(['import:cubvh']);
      expect(readiness.missing_degradable).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('succeeds in degraded mode with explicit SDPA fallback metadata when flash_attn install or import fails after core and cubvh succeed', () => {
    const scenarios = [
      {
        name: 'install',
        env: {
          ULTRASHAPE_SETUP_TEST_FLASH_ATTN_STAGE_FAIL: 'install',
        },
      },
      {
        name: 'import',
        env: {
          ULTRASHAPE_SETUP_TEST_FLASH_ATTN_STAGE_FAIL: 'import',
        },
      },
    ] as const;

    for (const scenario of scenarios) {
      const root = mkdtempSync(join(tmpdir(), `ultrashape-setup-flash-attn-${scenario.name}-`));
      const installDir = join(root, 'extension-root');
      const sourceWeight = join(root, 'download-cache', 'ultrashape_v1.pt');
      mkdirSync(join(root, 'download-cache'), { recursive: true });
      writeFileSync(sourceWeight, 'copied-weight');

      try {
        const outcome = runSetupWithEnv({
          python_exe: 'python3',
          ext_dir: installDir,
          gpu_sm: 90,
          cuda_version: 12.8,
          required_weight_path: sourceWeight,
        }, {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ...scenario.env,
        });

        expect(outcome.status).toBe(0);
        expect(`${outcome.stderr}${outcome.stdout}`).toContain(
          'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
        );

        const { summary, readiness } = readSetupArtifacts(installDir);
        expect(summary.native_install.cubvh).toMatchObject({
          attempted: true,
          required: true,
          status: 'ready',
        });
        expect(summary.native_install.flash_attn).toMatchObject({
          attempted: true,
          required: false,
          degradable: true,
          status: 'degraded',
          import_smoke_missing: ['flash_attn'],
          failure_message: 'flash_attn install failed; continuing with degraded PyTorch SDPA fallback.',
        });
        expect(summary.native_install.flash_attn.commands).toEqual([
          expect.stringContaining('pip install'),
        ]);
        expect(readiness.status).toBe('degraded');
        expect(readiness.install_success).toBe(true);
        expect(readiness.required_imports_ok).toBe(true);
        expect(readiness.missing_required).toEqual([]);
        expect(readiness.missing_degradable).toEqual(['flash_attn']);
        expect(readiness.missing_optional).toEqual(['flash_attn']);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    }
  });

  it('classifies terminal acquisition failures and mirrors diagnostics in both metadata artifacts', () => {
    const hfScenarios = [
      {
        name: 'auth',
        env: {
          ULTRASHAPE_SETUP_TEST_HF_SCENARIO: 'auth',
          ULTRASHAPE_WEIGHT_HF_TOKEN: 'secret-token',
        },
        expected: {
          classification: 'auth',
          errorClass: 'GatedRepoError',
          errorMessage: '401 Unauthorized from Hugging Face',
          authUsed: true,
        },
      },
      {
        name: 'network',
        env: {
          ULTRASHAPE_SETUP_TEST_HF_SCENARIO: 'network',
        },
        expected: {
          classification: 'network',
          errorClass: 'LocalEntryNotFoundError',
          errorMessage: 'Connection error while downloading UltraShape weights',
          authUsed: false,
        },
      },
      {
        name: 'not-found',
        env: {
          ULTRASHAPE_SETUP_TEST_HF_SCENARIO: 'not-found',
        },
        expected: {
          classification: 'not-found',
          errorClass: 'RemoteEntryNotFoundError',
          errorMessage: '404 missing ultrashape_v1.pt',
          authUsed: false,
        },
      },
      {
        name: 'other',
        env: {
          ULTRASHAPE_SETUP_TEST_HF_SCENARIO: 'other',
        },
        expected: {
          classification: 'other',
          errorClass: 'RuntimeError',
          errorMessage: 'Unexpected HF failure during UltraShape acquisition',
          authUsed: false,
        },
      },
    ] as const;

    for (const scenario of hfScenarios) {
      const root = mkdtempSync(join(tmpdir(), `ultrashape-setup-hf-${scenario.name}-`));
      const installDir = join(root, 'extension-root');

      try {
        const outcome = runSetupWithEnv({
          python_exe: 'python3',
          ext_dir: installDir,
          gpu_sm: 90,
          cuda_version: 12.8,
        }, {
          ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
          ...scenario.env,
        });

        expect(outcome.status, `${scenario.name} should exit non-zero`).not.toBe(0);

        const { summary, readiness } = readSetupArtifacts(installDir);
        expect(summary.install_success).toBe(false);
        expect(summary.failure_stage).toBe('weight-validation');
        expect(summary.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
        expect(readiness.status).toBe('blocked');
        expect(readiness.install_success).toBe(false);
        expect(readiness.failure_stage).toBe('weight-validation');
        expect(readiness.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');

        for (const artifact of [summary, readiness]) {
          expect(artifact.attempted_weight_source_kinds).toEqual(['ext-dir', 'repo-local', 'hf-default']);
          expect(artifact.resolved_weight_source_kind).toBe(null);
          expect(artifact.resolved_weight_source).toBe(null);
          expect(artifact.weight_source_repo_id).toBe('infinith/UltraShape');
          expect(artifact.weight_source_filename).toBe('ultrashape_v1.pt');
          expect(artifact.weight_source_revision).toBe(null);
          expect(artifact.weight_source_auth_used).toBe(scenario.expected.authUsed);
          expect(artifact.weight_source_failure_classification).toBe(scenario.expected.classification);
          expect(artifact.weight_source_error_class).toBe(scenario.expected.errorClass);
          expect(artifact.weight_source_error_message).toBe(scenario.expected.errorMessage);
        }
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    }
  });

  it('classifies local copy failures and mirrors diagnostics in both metadata artifacts', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-local-copy-failure-'));
    const installDir = join(root, 'extension-root');
    const blockedSource = join(root, 'blocked-source', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'blocked-source'), { recursive: true });
    writeFileSync(blockedSource, 'blocked-weight');

    try {
      const outcome = runSetupWithEnv({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
        required_weight_path: blockedSource,
      }, {
        ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
        ULTRASHAPE_SETUP_TEST_COPY_SCENARIO: 'permission-error',
      });

      expect(outcome.status).not.toBe(0);

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.install_success).toBe(false);
      expect(summary.failure_stage).toBe('weight-validation');
      expect(summary.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(readiness.status).toBe('blocked');
      expect(readiness.install_success).toBe(false);
      expect(readiness.failure_stage).toBe('weight-validation');
      expect(readiness.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');

      for (const artifact of [summary, readiness]) {
        expect(artifact.attempted_weight_source_kinds).toEqual(['ext-dir', 'required_weight_path']);
        expect(artifact.attempted_weight_sources).toEqual([
          join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'),
          blockedSource,
        ]);
        expect(artifact.resolved_weight_source_kind).toBe(null);
        expect(artifact.resolved_weight_source).toBe(null);
        expect(artifact.weight_source_repo_id).toBe(null);
        expect(artifact.weight_source_filename).toBe('ultrashape_v1.pt');
        expect(artifact.weight_source_revision).toBe(null);
        expect(artifact.weight_source_auth_used).toBe(null);
        expect(artifact.weight_source_failure_classification).toBe('other');
        expect(artifact.weight_source_error_class).toBe('PermissionError');
        expect(artifact.weight_source_error_message).toBe('copy blocked for test');
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails fast when ext_dir is missing from the Modly setup payload', () => {
    const outcome = runSetup({
      python_exe: 'python3',
      gpu_sm: 90,
    });

    expect(outcome.status).not.toBe(0);
    expect(`${outcome.stderr}${outcome.stdout}`).toContain('ext_dir');
  });

  it('accepts the numeric gpu_sm/cuda_version values that Modly sends at install time', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-numeric-'));
    const installDir = join(root, 'extension-root');
    mkdirSync(join(installDir, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(join(installDir, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'test-weight');

    try {
      const outcome = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 86,
        cuda_version: 128,
      });

      expect(outcome.status).toBe(0);
      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        gpu_sm: string;
        cuda_version: string;
      };
      expect(summary.gpu_sm).toBe('86');
      expect(summary.cuda_version).toBe('128');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
