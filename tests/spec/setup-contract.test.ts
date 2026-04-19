import { cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const setupPath = resolve(repoRoot, 'setup.py');
const extractedPayloadPaths = [
  'manifest.json',
  'setup.py',
  'processor.py',
  'README.md',
  'runtime/configs/infer_dit_refine.yaml',
  'runtime/vendor/ultrashape_runtime/__init__.py',
  'runtime/vendor/ultrashape_runtime/local_runner.py',
  'runtime/vendor/ultrashape_runtime/pipelines.py',
  'runtime/vendor/ultrashape_runtime/preprocessors.py',
  'runtime/vendor/ultrashape_runtime/surface_loaders.py',
  'runtime/vendor/ultrashape_runtime/schedulers.py',
  'runtime/vendor/ultrashape_runtime/utils/__init__.py',
  'runtime/vendor/ultrashape_runtime/utils/checkpoint.py',
  'runtime/vendor/ultrashape_runtime/utils/voxelize.py',
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
] as const;

type InstallSurfaceSummary = {
  layout: string;
  entry: string;
  backend_modes: string[];
  resolved_backend: string;
  output_formats: string[];
  remote_hybrid_supported: boolean;
};

type SetupSummary = {
  install_success: boolean;
  install_ready: boolean;
  runtime_ready: boolean;
  runtime_closure_ready: boolean;
  runtime_closure_reason: string | null;
  failure_stage: string | null;
  failure_code: string | null;
  missing_required: string[];
  install_surface: InstallSurfaceSummary;
  native_install_contract: {
    order: string[];
    cubvh_required: boolean;
    flash_attn_optional: boolean;
  };
};

type SetupReadiness = {
  status: string;
  install_success: boolean;
  install_ready: boolean;
  runtime_ready: boolean;
  runtime_closure_ready: boolean;
  runtime_closure_reason: string | null;
  weights_ready: boolean;
  required_imports_ok: boolean;
  missing_required: string[];
  missing_optional: string[];
  missing_conditional: string[];
  missing_degradable: string[];
  failure_stage: string | null;
  failure_code: string | null;
};

function runSetup(argument: Record<string, unknown>, cwd = repoRoot) {
  return spawnSync('python3', [cwd === repoRoot ? setupPath : 'setup.py', JSON.stringify(argument)], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
    },
  });
}

function readSetupArtifacts(installDir: string) {
  return {
    summary: JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as SetupSummary,
    readiness: JSON.parse(readFileSync(join(installDir, '.runtime-readiness.json'), 'utf8')) as SetupReadiness,
  };
}

function copyFileOrDirectory(relativePath: string, installDir: string) {
  cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
}

describe('UltraShape setup.py contract', () => {
  it('marks ext_dir installs runtime-ready when the stable shell includes the vendored upstream closure', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-shell-'));
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
      expect(existsSync(join(installDir, 'runtime', 'ultrashape_runtime', '__init__.py'))).toBe(true);

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.install_success).toBe(true);
      expect(summary.install_ready).toBe(true);
      expect(summary.runtime_ready).toBe(true);
      expect(summary.runtime_closure_ready).toBe(true);
      expect(summary.runtime_closure_reason).toContain('clean-room upstream closure');
      expect(summary.failure_stage).toBeNull();
      expect(summary.failure_code).toBeNull();
      expect(summary.missing_required).toEqual([]);
      expect(summary.install_surface).toEqual({
        layout: 'repo-root-python-only',
        entry: 'processor.py',
        backend_modes: ['auto', 'local'],
        resolved_backend: 'local',
        output_formats: ['glb'],
        remote_hybrid_supported: false,
      });
      expect(summary.native_install_contract).toEqual({
        order: ['core', 'cubvh', 'flash_attn'],
        cubvh_required: true,
        flash_attn_optional: true,
      });

      expect(readiness.status).toBe('ready');
      expect(readiness.install_success).toBe(true);
      expect(readiness.install_ready).toBe(true);
      expect(readiness.runtime_ready).toBe(true);
      expect(readiness.runtime_closure_ready).toBe(true);
      expect(readiness.runtime_closure_reason).toContain('clean-room upstream closure');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);
      expect(readiness.failure_stage).toBeNull();
      expect(readiness.failure_code).toBeNull();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('marks copied payload installs runtime-ready when the copied stable shell includes the vendored upstream closure', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-copied-shell-'));
    const installDir = join(root, 'modly-UltraShape-1.0-extension');
    const weightPath = join(root, 'weight-cache', 'models', 'ultrashape', 'ultrashape_v1.pt');
    mkdirSync(join(root, 'weight-cache', 'models', 'ultrashape'), { recursive: true });
    writeFileSync(weightPath, 'test-weight');

    for (const relativePath of extractedPayloadPaths) {
      copyFileOrDirectory(relativePath, installDir);
    }

    try {
      const outcome = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
        required_weight_path: weightPath,
      }, installDir);

      expect(outcome.status).toBe(0);

      const { summary, readiness } = readSetupArtifacts(installDir);
      expect(summary.install_success).toBe(true);
      expect(summary.install_ready).toBe(true);
      expect(summary.runtime_ready).toBe(true);
      expect(summary.runtime_closure_ready).toBe(true);
      expect(summary.runtime_closure_reason).toContain('clean-room upstream closure');
      expect(summary.failure_stage).toBeNull();
      expect(summary.failure_code).toBeNull();
      expect(summary.missing_required).toEqual([]);

      expect(readiness.status).toBe('ready');
      expect(readiness.install_success).toBe(true);
      expect(readiness.install_ready).toBe(true);
      expect(readiness.runtime_ready).toBe(true);
      expect(readiness.runtime_closure_ready).toBe(true);
      expect(readiness.runtime_closure_reason).toContain('clean-room upstream closure');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);
      expect(readiness.failure_stage).toBeNull();
      expect(readiness.failure_code).toBeNull();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
