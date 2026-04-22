import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  copyInstallSurface,
  createRuntimeInputs,
  runGeneratorProbe,
  stageCheckpoint,
  writeRuntimeStubModules,
} from './install-test-helpers.js';

const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==';

function runSetup(cwd: string, extDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['-S', 'setup.py', '--ext-dir', extDir], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

describe('generator lifecycle shell', () => {
  it('exposes the exact public generate signature required by the model shell contract', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-signature-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const signatureProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import inspect, json',
            'from generator import UltraShapeGenerator',
            'print(json.dumps(str(inspect.signature(UltraShapeGenerator.generate))))',
          ].join('\n'),
        ],
        { cwd: checkout, encoding: 'utf8' },
      );

      expect(signatureProbe.status).toBe(0);
      expect(JSON.parse(signatureProbe.stdout.trim())).toBe(
        '(self, image_bytes: \'bytes | None\', params: \'dict[str, Any] | None\' = None, progress_cb: \'Any | None\' = None, cancel_event: \'Any | None\' = None) -> \'str\'',
      );
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('reports download truthfully before and after staged setup assets exist', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-download-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);

    try {
      const beforeSetup = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }]);
      expect(beforeSetup.status).toBe(0);
      expect(JSON.parse(beforeSetup.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: false, loaded: false },
      ]);

      writeRuntimeStubModules(stubRoot);
      stageCheckpoint(checkout);
      const setupResult = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setupResult.status).toBe(0);

      const afterSetup = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }], { PYTHONPATH: stubRoot });
      expect(afterSetup.status).toBe(0);
      expect(JSON.parse(afterSetup.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: true, loaded: false },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('copies only the model shell authority into staged checkouts', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-surface-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const readme = readFileSync(path.join(checkout, 'README.md'), 'utf8');

      expect(existsSync(path.join(checkout, 'generator.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'setup.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'processor.py'))).toBe(false);
      expect(existsSync(path.join(checkout, 'processor.js'))).toBe(false);
      expect(readme).toContain('The model shell is the sole public authority: `manifest.json`, `setup.py`, and `generator.py`.');
      expect(readme).not.toContain('process-shell authority');
      expect(readme).not.toContain('processor.py');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('lazy-loads on generate, returns a mesh path, and unload resets the loaded state', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-lifecycle-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: {
              mesh_path: coarseMesh,
              remesh: true,
              enable_texture: true,
              texture_resolution: 1024,
            },
          },
          { method: 'unload' },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      const payload = JSON.parse(result.stdout);
      expect(payload[0]).toMatchObject({ method: 'generate', ok: true, loaded: true });
      expect(typeof payload[0].result).toBe('string');
      expect(payload[0].result).toMatch(/\.glb$/);
      expect(existsSync(payload[0].result)).toBe(true);
      expect(payload[0].result).toContain(path.join(checkout, 'outputs'));
      expect(payload[0].debug.last_result).toMatchObject({
        backend: 'local',
        subtrees_loaded: ['vae', 'dit', 'conditioner'],
      });
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('coarse_mesh');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('reference_image');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('remesh');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('enable_texture');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('texture_resolution');
      expect(payload[1]).toEqual({ method: 'unload', ok: true, result: false, loaded: false });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('resolves the reported Workflows-prefixed relative mesh_path against the real existing file', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-relative-mesh-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);

    try {
      const relativeMeshPath = path.join('Workflows', 'foo.glb');
      const existingMeshPath = path.join(checkout, 'Workflows', 'foo.glb');
      mkdirSync(path.dirname(existingMeshPath), { recursive: true });
      writeFileSync(existingMeshPath, 'mesh', 'utf8');
      const resolutionProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import json, sys',
            'from pathlib import Path',
            'from generator import UltraShapeGenerator',
            'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "Workflows")',
            'relative = str(generator._resolve_mesh_path(sys.argv[1]))',
            'print(json.dumps({"relative": relative}))',
          ].join('\n'),
          relativeMeshPath,
        ],
        {
          cwd: checkout,
          encoding: 'utf8',
          env: {
            ...process.env,
            PYTHONPATH: stubRoot,
          },
        },
      );

      expect(resolutionProbe.status).toBe(0);
      expect(JSON.parse(resolutionProbe.stdout)).toEqual({ relative: existingMeshPath });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('surfaces structured mesh resolution diagnostics when no candidate exists', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-mesh-diagnostics-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const workspaceDir = path.join(sandbox, 'workspace');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    mkdirSync(workspaceDir, { recursive: true });

    try {
      const diagnosticProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import json, sys',
            'from pathlib import Path',
            'from generator import PublicRuntimeError, UltraShapeGenerator',
            'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
            'try:',
            '    generator._resolve_mesh_path(sys.argv[1])',
            'except PublicRuntimeError as error:',
            '    print(json.dumps({"code": error.code, "message": str(error)}))',
          ].join('\n'),
          path.join('Workflows', 'missing.glb'),
        ],
        {
          cwd: checkout,
          encoding: 'utf8',
          env: {
            ...process.env,
            PYTHONPATH: stubRoot,
            WORKSPACE_DIR: workspaceDir,
          },
        },
      );

      expect(diagnosticProbe.status).toBe(0);
      expect(JSON.parse(diagnosticProbe.stdout)).toEqual({
        code: 'INVALID_INPUT',
        message: [
          'Mesh input could not be resolved.',
          'original mesh_path: Workflows/missing.glb',
          `self.outputs_dir: ${path.join(checkout, 'outputs')}`,
          `WORKSPACE_DIR: ${workspaceDir}`,
          'candidates:',
          `  1. path=${path.join(checkout, 'outputs', 'Workflows', 'missing.glb')} exists=False parent=${path.join(checkout, 'outputs', 'Workflows')} parent_exists=False`,
          `  2. path=${path.join(checkout, 'Workflows', 'missing.glb')} exists=False parent=${path.join(checkout, 'Workflows')} parent_exists=False`,
          `  3. path=${path.join(workspaceDir, 'Workflows', 'missing.glb')} exists=False parent=${path.join(workspaceDir, 'Workflows')} parent_exists=False`,
        ].join('\n'),
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('passes raw image_bytes through to the vendored runtime instead of ignoring them', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-bad-image-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: Buffer.from('not-a-png', 'utf8').toString('base64'),
            params: { mesh_path: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      const payload = JSON.parse(result.stdout);
      expect(payload[0]).toMatchObject({
        method: 'generate',
        ok: true,
        loaded: true,
        debug: {
          last_result: {
            metrics: {
              preprocess: {
                byte_length: 'not-a-png'.length,
                source_format: 'raw-bytes',
              },
              stage_evidence: {
                preprocess: {
                  source: 'reference_image',
                },
              },
            },
          },
        },
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('fails when params.mesh_path is not a readable glb because the vendored runtime actually consumes it', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-bad-mesh-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const invalidMesh = path.join(sandbox, 'invalid.glb');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    writeFileSync(invalidMesh, 'not-a-glb', 'utf8');

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: invalidMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'LOCAL_RUNTIME_UNAVAILABLE',
            message: expect.stringContaining('coarse_mesh'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('rejects legacy alias fields instead of silently accepting private runner payload keys', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-legacy-alias-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { coarse_mesh: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('legacy alias fields'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('rejects mixed public and private payload fields instead of tolerating ambiguous input', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-mixed-payload-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: coarseMesh, coarse_mesh: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('Mixed public contract fields'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('ignores Modly global generation params but still rejects other unsupported params', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-global-params-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: {
              mesh_path: coarseMesh,
              remesh: false,
              enable_texture: false,
              texture_resolution: 512,
              unsupported_flag: true,
            },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: 'Unsupported params fields: unsupported_flag.',
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('surfaces public dependency and weight failures without synthetic success', () => {
    const dependencySandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-deps-'));
    const dependencyCheckout = path.join(dependencySandbox, 'repo');
    copyInstallSurface(dependencyCheckout);

    const weightsSandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-weights-'));
    const weightsCheckout = path.join(weightsSandbox, 'repo');
    const stubRoot = path.join(weightsSandbox, 'stubs');
    copyInstallSurface(weightsCheckout);
    writeRuntimeStubModules(stubRoot);

    try {
      const dependencySetup = runSetup(dependencyCheckout, dependencyCheckout);
      expect(dependencySetup.status).toBe(1);
      const missingDependencies = runGeneratorProbe(dependencyCheckout, [{ method: 'load' }]);
      expect(missingDependencies.status).toBe(0);
      expect(JSON.parse(missingDependencies.stdout)).toEqual([
        {
          method: 'load',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'DEPENDENCY_MISSING',
            message: expect.stringContaining('Required runtime imports are unavailable'),
          },
          loaded: false,
        },
      ]);

      const setupResult = runSetup(weightsCheckout, weightsCheckout, { PYTHONPATH: stubRoot });
      expect(setupResult.status).toBe(1);
      const missingWeights = runGeneratorProbe(weightsCheckout, [{ method: 'generate', imageBase64: PNG_1X1_BASE64, params: {} }], {
        PYTHONPATH: stubRoot,
      });
      expect(missingWeights.status).toBe(0);
      expect(JSON.parse(missingWeights.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'WEIGHTS_MISSING',
            message: expect.stringContaining('Required runtime weights are unavailable'),
          },
          loaded: false,
        },
      ]);
    } finally {
      rmSync(dependencySandbox, { recursive: true, force: true });
      rmSync(weightsSandbox, { recursive: true, force: true });
    }
  });
});
