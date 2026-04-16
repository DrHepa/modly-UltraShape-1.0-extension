import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const extractedRootPaths = [
  'manifest.json',
  'setup.py',
  'processor.py',
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
  missing_optional: string[];
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

function copyExtractedRoot() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-github-install-'));
  const installDir = join(root, 'modly-UltraShape-1.0-extension');

  for (const relativePath of extractedRootPaths) {
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

describe('UltraShape GitHub install smoke', () => {
  it('fails GitHub/root install smoke when the required weight is absent', () => {
    const simulation = copyExtractedRoot();

    try {
      expect(existsSync(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/request.json'))).toBe(true);

      const manifest = JSON.parse(readFileSync(resolve(simulation.installDir, 'manifest.json'), 'utf8')) as {
        id: string;
        entry: string;
      };
      expect(manifest.id).toBe('modly.ultrashape-refiner-process');
      expect(manifest.entry).toBe('processor.py');

      const setup = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: 90,
        cuda_version: 128,
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        env: buildSetupEnv(),
      });

      expect(setup.status).not.toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/configs/infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/pipelines.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/denoisers/dit_mask.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/autoencoders/surface_extractors.py'))).toBe(true);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect((readiness as Readiness & { install_success: boolean; failure_code: string }).install_success).toBe(false);
      expect((readiness as Readiness & { install_success: boolean; failure_code: string }).failure_code).toBe('WEIGHT_ACQUISITION_FAILED');
      expect(readiness.status).toBe('blocked');
      expect(readiness.backend).toBe('local');
      expect(readiness.weights_ready).toBe(false);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual(['models/ultrashape/ultrashape_v1.pt']);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);
    } finally {
      simulation.cleanup();
    }
  });

  it('requires the staged required weight for a successful GitHub/root install smoke path', () => {
    const simulation = copyExtractedRoot();

    try {
      const sourceWeight = stageRequiredWeight(resolve(simulation.installDir, '..', 'weight-cache'));

      const setup = spawnSync('python3', ['setup.py', JSON.stringify({
        python_exe: 'python3',
        ext_dir: simulation.installDir,
        gpu_sm: 90,
        cuda_version: 128,
        required_weight_path: sourceWeight,
      })], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        env: buildSetupEnv(),
      });

      expect(setup.status).toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect(readiness.status).toBe('ready');
      expect(readiness.backend).toBe('local');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);

      const smokePayload = {
        input: {
          filePath: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/reference-image.png'),
        },
        params: {
          coarse_mesh: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb'),
        },
        workspaceDir: resolve(simulation.installDir, 'smoke-output'),
      };

      const outcome = spawnSync('python3', ['processor.py'], {
        cwd: simulation.installDir,
        encoding: 'utf8',
        input: `${JSON.stringify(smokePayload)}\n`,
        env: {
          ...process.env,
          ULTRASHAPE_TEST_ARTIFACT_PATH: resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb'),
        },
      });

      const events = outcome.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(outcome.status).toBe(0);

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
