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
  'runtime/vendor/ultrashape_runtime/local_runner.py',
  'runtime/vendor/ultrashape_runtime/pipelines.py',
  'runtime/vendor/ultrashape_runtime/preprocessors.py',
  'runtime/vendor/ultrashape_runtime/rembg.py',
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
  'fixtures/requests/refiner-bundle/request.json',
  'fixtures/requests/refiner-bundle/assets/reference-image.png',
  'fixtures/requests/refiner-bundle/assets/coarse-mesh.glb',
  'fixtures/requests/refiner-bundle/expected/output/refined-mesh.glb',
];

type Readiness = {
  status: 'ready' | 'degraded' | 'blocked';
  backend: string;
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
  expected_weights: string[];
};

type HfTrace = {
  api: string;
  repo_id: string;
  filename: string;
  revision: string | null;
  token: string | null;
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

function writeBinaryCheckpointBundle(path: string) {
  const outcome = spawnSync(
    'python3',
    [
      '-c',
      [
        'import sys, zipfile',
        'with zipfile.ZipFile(sys.argv[1], "w", compression=zipfile.ZIP_DEFLATED) as archive:',
        '    archive.writestr("checkpoint.json", sys.argv[2])',
      ].join('\n'),
      path,
      checkpointBundleText(),
    ],
    { encoding: 'utf8' },
  );

  if (outcome.status !== 0) {
    throw new Error(outcome.stderr || 'Failed to write binary checkpoint bundle.');
  }
}

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
  writeBinaryCheckpointBundle(weightPath);
  return weightPath;
}

function buildSetupEnv(extra: NodeJS.ProcessEnv = {}) {
  return {
    ...process.env,
    ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
    ...extra,
  };
}

describe('UltraShape GitHub install smoke', () => {
  it('keeps GitHub/root install smoke runtime-ready once the upstream closure is real', () => {
    const simulation = copyExtractedRoot();
    const hfTracePath = resolve(simulation.installDir, '.hf-download-trace.json');

    try {
      expect(existsSync(resolve(simulation.installDir, 'fixtures/requests/refiner-bundle/request.json'))).toBe(true);
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
        env: buildSetupEnv({
          ULTRASHAPE_SETUP_TEST_HF_HUB_DOWNLOAD_FILE: checkpointBundleText(),
          ULTRASHAPE_SETUP_TEST_HF_TRACE_PATH: hfTracePath,
        }),
      });

      expect(setup.status).toBe(0);
      expect(existsSync(resolve(simulation.installDir, 'venv'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/configs/infer_dit_refine.yaml'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/local_runner.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/pipelines.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/denoisers/dit_mask.py'))).toBe(true);
      expect(existsSync(resolve(simulation.installDir, 'runtime/ultrashape_runtime/models/autoencoders/surface_extractors.py'))).toBe(true);
      expect(readFileSync(resolve(simulation.installDir, 'models/ultrashape/ultrashape_v1.pt'), 'utf8')).toBe(checkpointBundleText());

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness & {
        install_success: boolean;
        failure_stage: string | null;
        failure_code: string | null;
        attempted_weight_source_kinds: string[];
        resolved_weight_source_kind: string;
        weight_source_repo_id: string;
        weight_source_filename: string;
      };
      expect(readiness.install_success).toBe(true);
      expect(readiness.install_ready).toBe(true);
      expect(readiness.runtime_ready).toBe(true);
      expect(readiness.runtime_closure_ready).toBe(true);
      expect(readiness.runtime_closure_reason).toContain('ported upstream MVP closure');
      expect(readiness.failure_stage).toBeNull();
      expect(readiness.failure_code).toBeNull();
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
      expect(readiness.weight_source_repo_id).toBe('infinith/UltraShape');
      expect(readiness.weight_source_filename).toBe('ultrashape_v1.pt');

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

  it('requires the staged required weight and keeps GitHub/root install smoke aligned on install-ready truth only', () => {
    const simulation = copyExtractedRoot();

    try {
      const manifest = JSON.parse(readFileSync(resolve(simulation.installDir, 'manifest.json'), 'utf8')) as {
        entry: string;
        testing_fallback_only: {
          semantic_contract: string;
          supported_execution: {
            backend_modes: string[];
            resolved_backend: string;
            output_formats: string[];
            remote_hybrid_supported: boolean;
          };
        };
      };
      expect(manifest.entry).toBe('processor.py');
      expect(manifest.testing_fallback_only.semantic_contract).toBe('reference_image + coarse_mesh -> refined.glb');
      expect(manifest.testing_fallback_only.supported_execution).toEqual({
        backend_modes: ['auto', 'local'],
        resolved_backend: 'local',
        output_formats: ['glb'],
        remote_hybrid_supported: false,
      });

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

      const summary = JSON.parse(readFileSync(resolve(simulation.installDir, '.setup-summary.json'), 'utf8')) as {
        install_surface: {
          layout: string;
          entry: string;
          backend_modes: string[];
          resolved_backend: string;
          output_formats: string[];
          remote_hybrid_supported: boolean;
        };
      };
      expect(summary.install_surface).toEqual({
        layout: 'repo-root-python-only',
        entry: 'processor.py',
        backend_modes: ['auto', 'local'],
        resolved_backend: 'local',
        output_formats: ['glb'],
        remote_hybrid_supported: false,
      });

      const readiness = JSON.parse(readFileSync(resolve(simulation.installDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
      expect(readiness.install_ready).toBe(true);
      expect(readiness.runtime_ready).toBe(true);
      expect(readiness.runtime_closure_ready).toBe(true);
      expect(readiness.runtime_closure_reason).toContain('ported upstream MVP closure');
      expect(readiness.status).toBe('ready');
      expect(readiness.backend).toBe('local');
      expect(readiness.weights_ready).toBe(true);
      expect(readiness.required_imports_ok).toBe(true);
      expect(readiness.missing_required).toEqual([]);
      expect(readiness.missing_optional).toEqual([]);
      expect(readiness.missing_conditional).toEqual([]);
      expect(readiness.missing_degradable).toEqual([]);
      expect(readiness.expected_weights).toEqual(['models/ultrashape/ultrashape_v1.pt']);

      expect(expectedProcessorOutcome(readiness)).toBe('done');
    } finally {
      simulation.cleanup();
    }
  });
});
