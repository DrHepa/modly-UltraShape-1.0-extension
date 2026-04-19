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

const removedAuthorityPaths = [
  'runtime/patches/README.md',
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
] as const;

type SetupSummary = {
  install_success: boolean;
  install_ready: boolean;
  runtime_ready: boolean;
  runtime_closure_ready: boolean;
  runtime_closure_reason: string | null;
  failure_stage: string | null;
  failure_code: string | null;
  missing_required: string[];
};

type Readiness = SetupSummary & {
  status: 'ready' | 'degraded' | 'blocked';
  backend: string;
  weights_ready: boolean;
  required_imports_ok: boolean;
  missing_optional: string[];
  missing_conditional: string[];
  missing_degradable: string[];
  expected_weights: string[];
};

function expectedProcessorOutcome(readiness: Readiness): 'done' | 'WEIGHTS_MISSING' | 'DEPENDENCY_MISSING' | 'LOCAL_RUNTIME_UNAVAILABLE' {
  if (!readiness.weights_ready) {
    return 'WEIGHTS_MISSING';
  }

  if (!readiness.required_imports_ok) {
    return 'DEPENDENCY_MISSING';
  }

  if (!readiness.runtime_ready || readiness.status === 'blocked' || readiness.backend !== 'local') {
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
  writeFileSync(weightPath, 'test-weight');
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
  it('keeps the copied payload limited to the stable shell surface', () => {
    for (const relativePath of removedAuthorityPaths) {
      expect(existsSync(resolve(repoRoot, relativePath)), `${relativePath} should not remain in the source payload`).toBe(false);
    }

    const simulation = copyInstallSurface();

    try {
      for (const relativePath of installSurfacePaths) {
        expect(existsSync(resolve(simulation.installDir, relativePath)), `${relativePath} should be copied`).toBe(true);
      }

      expect(existsSync(resolve(simulation.installDir, 'package.json'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'processor.js'))).toBe(false);
      expect(existsSync(resolve(simulation.installDir, 'runtime/modly'))).toBe(false);
    } finally {
      simulation.cleanup();
    }
  });

  it('reports copied payload installs as runtime-ready when the copied stable shell includes the vendored upstream closure', () => {
    const simulation = copyInstallSurface();

    try {
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

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as SetupSummary;
      expect(summary.install_success).toBe(true);
      expect(summary.install_ready).toBe(true);
      expect(summary.runtime_ready).toBe(true);
      expect(summary.runtime_closure_ready).toBe(true);
      expect(summary.runtime_closure_reason).toContain('clean-room upstream closure');
      expect(summary.failure_stage).toBeNull();
      expect(summary.failure_code).toBeNull();
      expect(summary.missing_required).toEqual([]);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect(readiness.status).toBe('ready');
      expect(readiness.backend).toBe('local');
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
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.failure_stage).toBeNull();
      expect(readiness.failure_code).toBeNull();
      expect(expectedProcessorOutcome(readiness)).toBe('done');
    } finally {
      simulation.cleanup();
    }
  });
});
