import { chmodSync, cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const processorPath = resolve(repoRoot, 'processor.py');
const runtimeVendorPath = resolve(repoRoot, 'runtime', 'vendor');
const runtimeConfigSourcePath = resolve(repoRoot, 'runtime', 'configs', 'infer_dit_refine.yaml');

function expectUpstreamRuntimeGraph(configText: string) {
  expect(configText).toContain('checkpoint:');
  expect(configText).toContain('preprocess:');
  expect(configText).toContain('conditioning:');
  expect(configText).toContain('scheduler:');
  expect(configText).toContain('decoder:');
  expect(configText).toContain('surface:');
  expect(configText).toContain('gate:');
  expect(configText).toContain('export:');
  expect(configText).toContain('scope: mc-only');
  expect(configText).toContain('backend: local');
  expect(configText).toContain('format: glb');
  expect(configText).toContain('required:');
  expect(configText).toContain('- cubvh');
  expect(configText).toContain('conditional:');
  expect(configText).toContain('- rembg');
  expect(configText).toContain('- onnxruntime');
  expect(configText).toContain('degradable:');
  expect(configText).toContain('- flash_attn');
}

function createBinaryGlbBytes() {
  const jsonChunk = Buffer.from('{"asset":{"version":"2.0"}}   ', 'utf8');
  const binaryChunk = Buffer.from([0x00, 0x80, 0x00, 0x00]);
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binaryChunk.length;

  return Buffer.concat([
    Buffer.from('glTF', 'ascii'),
    Buffer.from(Uint32Array.of(2, totalLength).buffer),
    Buffer.from(Uint32Array.of(jsonChunk.length, 0x4e4f534a).buffer),
    jsonChunk,
    Buffer.from(Uint32Array.of(binaryChunk.length, 0x004e4942).buffer),
    binaryChunk,
  ]);
}

function expectBinaryGlb(path: string) {
  const payload = readFileSync(path);
  expect(payload.subarray(0, 4).toString('ascii')).toBe('glTF');
  expect(payload.length).toBeGreaterThanOrEqual(24);
}

function getRunnerMetrics(result: Record<string, unknown> | null): Record<string, unknown> {
  const envelope = result?.result;
  if (!envelope || typeof envelope !== 'object' || !('metrics' in envelope)) {
    throw new Error('Expected runner success metrics to be present.');
  }

  const metrics = envelope.metrics;
  if (!metrics || typeof metrics !== 'object') {
    throw new Error('Expected runner metrics to be an object.');
  }

  return metrics as Record<string, unknown>;
}

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-runtime-'));
  const outputDir = join(root, 'output');
  const modelsDir = join(root, 'models', 'ultrashape');
  const stubDir = join(root, 'py-stubs');
  mkdirSync(outputDir);
  mkdirSync(modelsDir, { recursive: true });
  mkdirSync(stubDir, { recursive: true });

  const referenceImage = join(root, 'reference.png');
  const coarseMesh = join(root, 'coarse.glb');
  const binaryCoarseMesh = join(root, 'binary-coarse.glb');
  const packagedArtifact = join(root, 'artifact.glb');
  const checkpoint = join(modelsDir, 'ultrashape_v1.pt');
  const configPath = join(root, 'infer_dit_refine.yaml');

  writeFileSync(referenceImage, 'image');
  writeFileSync(coarseMesh, 'mesh');
  writeFileSync(binaryCoarseMesh, createBinaryGlbBytes());
  writeFileSync(packagedArtifact, 'refined-mesh');
  writeFileSync(join(stubDir, 'cubvh.py'), '__version__ = "0.0-test"\n');
  writeCheckpointBundle(checkpoint);

  return {
    root,
    outputDir,
     referenceImage,
     coarseMesh,
     binaryCoarseMesh,
      packagedArtifact,
      checkpoint,
      configPath,
      stubDir,
      cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

function checkpointBundlePayload(variant: 'a' | 'b' = 'a') {
  const values =
    variant === 'a'
      ? {
          vae: [0.11, 0.33, 0.55, 0.77],
          dit: [0.21, 0.41, 0.61, 0.81],
          conditioner: [0.14, 0.24, 0.64, 0.74],
        }
      : {
          vae: [0.77, 0.55, 0.33, 0.11],
          dit: [0.81, 0.61, 0.41, 0.21],
          conditioner: [0.74, 0.64, 0.24, 0.14],
        };

  return {
    format: 'ultrashape-checkpoint-bundle/v1',
    vae: {
      tensors: {
        latent_basis: values.vae,
      },
    },
    dit: {
      tensors: {
        attention_bias: values.dit,
      },
    },
    conditioner: {
      tensors: {
        mask_bias: values.conditioner,
      },
    },
  };
}

function writeCheckpointBundle(path: string, variant: 'a' | 'b' = 'a') {
  writeFileSync(
    path,
    JSON.stringify(checkpointBundlePayload(variant)),
  );
}

function expectRealClosureMetrics(metrics: Record<string, unknown>) {
  expect(metrics).toEqual(
    expect.objectContaining({
      chamfer: expect.any(Number),
      rms: expect.any(Number),
      topology_changed: true,
      extent_ratio: [1, 1, 1],
      execution_trace: ['preprocess', 'conditioning', 'scheduler', 'denoise', 'decode', 'extract'],
      preprocess: expect.objectContaining({
        byte_length: expect.any(Number),
        normalized_channels: 4,
      }),
      conditioning: expect.objectContaining({
        voxel_count: expect.any(Number),
        mask_tokens: expect.any(Number),
      }),
      scheduler: expect.objectContaining({
        family: 'flow-matching',
        step_count: expect.any(Number),
      }),
      denoise: expect.objectContaining({
        attention: expect.any(String),
        latent_signature: expect.any(Number),
      }),
      decode: expect.objectContaining({
        field_density: expect.any(Number),
      }),
      extract: expect.objectContaining({
        extractor: expect.any(String),
        payload_bytes: expect.any(Number),
      }),
    }),
  );
}

function installProcessorRuntime(extDir: string) {
  const runtimeDir = join(extDir, 'runtime');
  const runtimePackageDir = join(runtimeDir, 'ultrashape_runtime');
  const runtimeConfigDir = join(runtimeDir, 'configs');
  const venvBinDir = join(extDir, 'venv', 'bin');
  const modelsDir = join(extDir, 'models', 'ultrashape');
  const pythonShimPath = join(venvBinDir, 'python');

  mkdirSync(runtimeConfigDir, { recursive: true });
  mkdirSync(venvBinDir, { recursive: true });
  mkdirSync(modelsDir, { recursive: true });

  cpSync(resolve(repoRoot, 'runtime', 'vendor', 'ultrashape_runtime'), runtimePackageDir, { recursive: true });
  writeFileSync(join(runtimeDir, 'torch.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'torchvision.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'numpy.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'trimesh.py'), '__version__ = "0.0-test"\n');
  mkdirSync(join(runtimeDir, 'PIL'), { recursive: true });
  writeFileSync(join(runtimeDir, 'PIL', '__init__.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'cv2.py'), '__version__ = "0.0-test"\n');
  mkdirSync(join(runtimeDir, 'skimage'), { recursive: true });
  writeFileSync(join(runtimeDir, 'skimage', '__init__.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'yaml.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'omegaconf.py'), 'class OmegaConf:\n    pass\n');
  writeFileSync(join(runtimeDir, 'einops.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'transformers.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'huggingface_hub.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'accelerate.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'cubvh.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'safetensors.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'tqdm.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeConfigDir, 'infer_dit_refine.yaml'), readFileSync(runtimeConfigSourcePath, 'utf8'));
  writeCheckpointBundle(join(modelsDir, 'ultrashape_v1.pt'));
  writeFileSync(
    pythonShimPath,
    ['#!/usr/bin/env bash', 'set -euo pipefail', 'exec python3 "$@"', ''].join('\n'),
  );
  chmodSync(pythonShimPath, 0o755);
}

function writeRuntimeConfig(
  path: string,
  overrides: Partial<{
    scope: string;
    backend: string;
    extraction: string;
    primaryWeight: string;
    requiredImports: string[];
  }> = {},
) {
  const requiredImports = overrides.requiredImports ?? [];
  writeFileSync(
    path,
    [
      'model:',
      `  scope: ${overrides.scope ?? 'mc-only'}`,
      'runtime:',
      `  backend: ${overrides.backend ?? 'local'}`,
      'surface:',
      `  extraction: ${overrides.extraction ?? 'mc'}`,
      'weights:',
      `  primary: ${overrides.primaryWeight ?? 'models/ultrashape/ultrashape_v1.pt'}`,
      'dependencies:',
      '  required:',
      '    imports:',
      ...requiredImports.map((entry) => `      - ${entry}`),
      '',
    ].join('\n'),
  );
}

function writeReadiness(
  root: string,
  overrides: Partial<{
    status: 'ready' | 'degraded' | 'blocked';
    backend: 'local';
    mvp_scope: 'mc-only';
    weights_ready: boolean;
    required_imports_ok: boolean;
    missing_required: string[];
    missing_optional: string[];
    expected_weights: string[];
  }> = {},
) {
  writeFileSync(
    join(root, '.runtime-readiness.json'),
    JSON.stringify(
      {
        status: 'ready',
        backend: 'local',
        mvp_scope: 'mc-only',
        weights_ready: true,
        required_imports_ok: true,
        missing_required: [],
        missing_optional: [],
        expected_weights: ['models/ultrashape/ultrashape_v1.pt'],
        ...overrides,
      },
      null,
      2,
    ),
  );
}

function runProcessor(
  payload: Record<string, unknown>,
  options: { env?: NodeJS.ProcessEnv; cwd?: string; processorPath?: string } = {},
) {
  const outcome = spawnSync('python3', [options.processorPath ?? processorPath], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      ...options.env,
    },
  });

  return outcome.stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function runLocalRunner(payload: Record<string, unknown>, options: { env?: NodeJS.ProcessEnv; cwd?: string } = {}) {
  const extDir = typeof payload.ext_dir === 'string' ? payload.ext_dir : null;
  const stubDir = extDir ? join(extDir, 'py-stubs') : null;
  const pythonPath = [stubDir, runtimeVendorPath, options.env?.PYTHONPATH].filter(Boolean).join(':');

  const outcome = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      ...options.env,
    },
  });

  return outcome.stdout ? (JSON.parse(outcome.stdout) as Record<string, unknown>) : null;
}

describe('UltraShape runtime flow', () => {
  it('ships an upstream-style runtime graph that truthfully encodes the real-refinement dependency tiers', () => {
    expectUpstreamRuntimeGraph(readFileSync(runtimeConfigSourcePath, 'utf8'));
  });

  it('runs the repo-root Python boundary from the named-input contract and packages refined.<format> without any JS install artifact', () => {
    const fixture = createFixtureWorkspace();

    try {
      installProcessorRuntime(fixture.root);
      writeReadiness(fixture.root);

      const events = runProcessor(
        {
          extDir: fixture.root,
          input: {
            inputs: {
              reference_image: {
                filePath: fixture.referenceImage,
              },
              coarse_mesh: {
                filePath: fixture.coarseMesh,
              },
            },
          },
          params: {
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(existsSync(resolve(repoRoot, 'processor.js'))).toBe(false);
      expect(existsSync(resolve(repoRoot, 'runtime/modly'))).toBe(false);
      expect(events.at(-1)).toEqual({
        type: 'done',
        result: {
          filePath: join(fixture.outputDir, 'refined.glb'),
        },
      });
      expectBinaryGlb(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });

  it('keeps the fallback seam compatible when named inputs are absent and validation still succeeds', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root, {
        status: 'blocked',
        required_imports_ok: false,
        missing_required: ['onnxruntime'],
      });

      const events = runProcessor(
        {
          extDir: fixture.root,
          input: {
            filePath: fixture.referenceImage,
          },
          params: {
            coarse_mesh: fixture.coarseMesh,
            backend: 'auto',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('DEPENDENCY_MISSING'),
        code: 'DEPENDENCY_MISSING',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('prefers the installed extension directory from processor.py before cwd fallback when extDir is omitted', () => {
    const fixture = createFixtureWorkspace();
    const installedExtDir = join(fixture.root, 'installed-extension');
    const installedProcessorPath = join(installedExtDir, 'processor.py');

    try {
      mkdirSync(installedExtDir);
      writeFileSync(installedProcessorPath, readFileSync(processorPath, 'utf8'));
      installProcessorRuntime(installedExtDir);
      writeReadiness(installedExtDir);

      const events = runProcessor(
        {
          input: {
            inputs: {
              reference_image: {
                filePath: fixture.referenceImage,
              },
              coarse_mesh: {
                filePath: fixture.coarseMesh,
              },
            },
          },
          params: {
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          processorPath: installedProcessorPath,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'done',
        result: {
          filePath: join(fixture.outputDir, 'refined.glb'),
        },
      });
      expectBinaryGlb(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });

  it('maps blocked local readiness without missing deps or weights to LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root, {
        status: 'blocked',
      });

      const events = runProcessor(
        {
          extDir: fixture.root,
          input: {
            inputs: {
              reference_image: {
                filePath: fixture.referenceImage,
              },
              coarse_mesh: {
                filePath: fixture.coarseMesh,
              },
            },
          },
          params: {
            backend: 'auto',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('runs the vendored local runner to generate refined.glb inside the requested output directory', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.any(Object),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          warnings: [],
        },
      });
      expectRealClosureMetrics(getRunnerMetrics(outcome));
      expectBinaryGlb(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });

  it('accepts binary glb coarse meshes without utf8 decode failures and writes a binary refined.glb', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.binaryCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.any(Object),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          warnings: [],
        },
      });
      expectRealClosureMetrics(getRunnerMetrics(outcome));
      expectBinaryGlb(join(fixture.outputDir, 'refined.glb'));
      expect(readFileSync(join(fixture.outputDir, 'refined.glb')).includes(0x80)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-mc scope configs at the vendored runner seam as LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath, {
        scope: 'full-volume',
      });

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('mc-only'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-glb output requests at the vendored runner seam as LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'obj',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('glb-only'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing checkpoint drift after config resolution to WEIGHTS_MISSING at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      rmSync(fixture.checkpoint);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: expect.stringContaining('ultrashape_v1.pt'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects checkpoints that do not contain the required vae/dit/conditioner subtrees', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeFileSync(
        fixture.checkpoint,
        JSON.stringify({
          format: 'ultrashape-checkpoint-bundle/v1',
          vae: { weights: 'fixture-vae' },
          dit: { weights: 'fixture-dit' },
        }),
      );

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: expect.stringContaining('conditioner'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects plain JSON subtree stubs that do not satisfy the real checkpoint tensor contract', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeFileSync(
        fixture.checkpoint,
        JSON.stringify({
          format: 'ultrashape-checkpoint-bundle/v1',
          vae: { weights: 'fixture-vae' },
          dit: { weights: 'fixture-dit' },
          conditioner: { weights: 'fixture-conditioner' },
        }),
      );

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: expect.stringContaining('tensor'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing config bootstrap failures to LOCAL_RUNTIME_UNAVAILABLE at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: fixture.checkpoint,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('config_path'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing required runtime imports to DEPENDENCY_MISSING at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath, {
        requiredImports: ['module_that_does_not_exist_for_ultrashape_tests'],
      });

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'DEPENDENCY_MISSING',
        error_message: expect.stringContaining('module_that_does_not_exist_for_ultrashape_tests'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing output generation to LOCAL_RUNTIME_UNAVAILABLE at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: null,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_SKIP_OUTPUT_WRITE: '1',
          },
        },
      );

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('refined.glb'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('derives closure metrics from executed inputs instead of placeholder formulas', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      writeFileSync(fixture.referenceImage, 'image-with-real-preprocess-drift');

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.binaryCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.preprocess).not.toEqual(secondMetrics.preprocess);
      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
    } finally {
      fixture.cleanup();
    }
  });

  it('derives staged closure metrics from checkpoint tensor values, not subtree key presence alone', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeCheckpointBundle(fixture.checkpoint, 'a');

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      writeCheckpointBundle(fixture.checkpoint, 'b');

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstResult = firstOutcome?.result as { metrics?: Record<string, unknown> } | undefined;
      const secondResult = secondOutcome?.result as { metrics?: Record<string, unknown> } | undefined;
      const firstMetrics = firstResult?.metrics ?? {};
      const secondMetrics = secondResult?.metrics ?? {};

      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects aligned passthrough geometry at the vendored runner seam and does not publish refined.glb', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_PASSTHROUGH: '1',
          },
        },
      );

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('GEOMETRIC_GATE_REJECTED'),
      });
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('enforces preserve-scale bbox tolerance while allowing isotropic scale fit only when preserve_scale=false', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const rejected = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_SCALE_DRIFT: '1',
          },
        },
      );

      expect(rejected).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('GEOMETRIC_GATE_REJECTED'),
      });
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);

      const accepted = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: false,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_SCALE_DRIFT: '1',
          },
        },
      );

      expect(accepted).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.objectContaining({
            extent_ratio: expect.any(Array),
          }),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          warnings: [],
        },
      });
      expectBinaryGlb(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });
});
