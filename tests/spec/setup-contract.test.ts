import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const setupPath = resolve(repoRoot, 'setup.py');

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

function readSetupArtifacts(installDir: string) {
  const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as Record<string, unknown>;
  const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as Record<string, unknown>;
  return { summary, readiness };
}

describe('UltraShape setup.py contract', () => {
  it('fails install when the required weight is missing but still writes truthful failure metadata', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-'));
    const installDir = join(root, 'extension-root');
    const firstRun = runSetup({
      python_exe: 'python3',
      ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
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
          optional: string[];
        };
      };
      expect(summary.torch_profile).toBe('linux-arm64-cu128-sm90+');
      expect(summary.runtime_layout_version).toBe('1');
      expect(summary.install_success).toBe(false);
      expect(summary.failure_stage).toBe('weight-validation');
      expect(summary.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(summary.dependencies.required).toContain('onnxruntime');
      expect(summary.dependencies.optional).toContain('cubvh');

      const readiness = JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as {
        status: string;
        backend: string;
        mvp_scope: string;
        weights_ready: boolean;
        required_imports_ok: boolean;
        missing_required: string[];
        missing_optional: string[];
        expected_weights: string[];
        install_success: boolean;
        failure_stage: string;
        failure_code: string;
      };
      expect(readiness.backend).toBe('local');
      expect(readiness.mvp_scope).toBe('mc-only');
      expect(readiness.status).toBe('blocked');
      expect(readiness.weights_ready).toBe(false);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.install_success).toBe(false);
      expect(readiness.failure_stage).toBe('weight-validation');
      expect(readiness.failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(readiness.missing_required).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);

      const sentinel = join(installDir, 'venv', 'sentinel.txt');
      writeFileSync(sentinel, 'keep-me');

      const secondRun = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
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
      const outcome = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
      });

      expect(outcome.status).toBe(0);

      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        install_success: boolean;
        attempted_weight_source_kinds: string[];
        attempted_weight_sources: string[];
        resolved_weight_source_kind: string;
        resolved_weight_source: string;
      };
      expect(summary.install_success).toBe(true);
      expect(summary.attempted_weight_source_kinds).toEqual(['ext-dir']);
      expect(summary.attempted_weight_sources).toEqual([weightPath]);
      expect(summary.resolved_weight_source_kind).toBe('ext-dir');
      expect(summary.resolved_weight_source).toBe(weightPath);

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
      };
      expect(summary.install_success).toBe(true);
      expect(summary.attempted_weight_sources).toContain(sourceWeight);

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

  it('succeeds with degraded readiness when only an optional dependency is absent', () => {
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
        ULTRASHAPE_SETUP_TEST_STUB_DEPS_MISSING: 'cubvh',
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
      expect(readiness.missing_optional).toEqual(['cubvh']);
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
