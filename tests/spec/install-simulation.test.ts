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
  expected_weights: string[];
};

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
  writeFileSync(weightPath, 'test-weight');
  return weightPath;
}

function buildSetupEnv() {
  return {
    ...process.env,
    ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
  };
}

describe('UltraShape Python install surface', () => {
  it('matches the expected GitHub-extracted payload layout for the Modly extensions directory', () => {
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

  it('fails extracted-root setup when the required weight is absent but writes truthful failure artifacts', () => {
    const simulation = copyInstallSurface();

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
        env: buildSetupEnv(),
      });

      expect(outcome.status).not.toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, '.setup-summary.json'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, '.runtime-readiness.json'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/configs/infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/__init__.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/denoisers/dit_mask.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/autoencoders/surface_extractors.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/.locks'))).toBe(true);

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as {
        torch_profile: string;
        runtime_layout_version: string;
        install_success: boolean;
      };
      expect(summary.torch_profile).toBe('linux-arm64-cu128-sm90+');
      expect(summary.runtime_layout_version).toBe('1');
      expect(summary.install_success).toBe(false);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect((readiness as Readiness & { install_success: boolean; failure_code: string }).install_success).toBe(false);
      expect((readiness as Readiness & { install_success: boolean; failure_code: string }).failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(readiness.status).toBe('blocked');
      expect(readiness.backend).toBe('local');
      expect(readiness.weights_ready).toBe(false);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);
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
        env: {
          ...process.env,
          ULTRASHAPE_TEST_ARTIFACT_PATH: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb'),
        },
      });

      const events = smoke.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(smoke.status).toBe(0);

      const expectedOutcome = expectedProcessorOutcome(readiness);
      if (expectedOutcome === 'done') {
        expect(events.at(-1)).toEqual({
          type: 'done',
          result: {
            filePath: resolve(simulation.installDir, 'smoke-output/refined.glb'),
          },
        });
        expect(existsSync(resolve(simulation.installDir, 'smoke-output/refined.glb'))).toBe(true);
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
});
