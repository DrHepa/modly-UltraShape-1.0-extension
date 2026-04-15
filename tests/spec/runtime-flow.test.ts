import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync, cpSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import type { UltraShapeExecutionClient } from '../../src/adapters/ultrashape/client.js';
import { executeUltraShapeRefiner } from '../../src/processes/ultrashape-refiner/index.js';
import type { UltraShapeProgressEvent } from '../../src/processes/ultrashape-refiner/types.js';

const repoRoot = process.cwd();
const curatedPayloadPaths = [
  'manifest.json',
  'processor.js',
  'runtime/modly/package.json',
  'runtime/modly/processes/ultrashape-refiner/index.js',
  'runtime/modly/processes/ultrashape-refiner/validate.js',
  'runtime/modly/processes/ultrashape-refiner/normalize.js',
  'runtime/modly/processes/ultrashape-refiner/preflight.js',
  'runtime/modly/processes/ultrashape-refiner/runtime.js',
  'runtime/modly/processes/ultrashape-refiner/progress.js',
  'runtime/modly/processes/ultrashape-refiner/types.js',
  'runtime/modly/adapters/ultrashape/client.js',
  'runtime/modly/adapters/ultrashape/remote.js',
  'runtime/modly/adapters/ultrashape/local.js',
];

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-runtime-'));
  const outputDir = join(root, 'output');
  mkdirSync(outputDir);

  const referenceImage = join(root, 'reference.png');
  const coarseMesh = join(root, 'coarse.glb');
  const remoteMesh = join(root, 'remote-output.obj');

  writeFileSync(referenceImage, 'image');
  writeFileSync(coarseMesh, 'mesh');
  writeFileSync(remoteMesh, 'refined-mesh');

  return {
    root,
    outputDir,
    referenceImage,
    coarseMesh,
    remoteMesh,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

function loadInstalledProcessor() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-installed-processor-'));
  const installDir = join(root, 'modly.ultrashape-refiner-process');

  for (const relativePath of curatedPayloadPaths) {
    cpSync(resolve(repoRoot, relativePath), resolve(installDir, relativePath), { recursive: true });
  }

  const requireFromInstall = createRequire(resolve(installDir, 'index.js'));

  return {
    processor: requireFromInstall('./processor.js') as (
      input: Record<string, unknown>,
      params: Record<string, unknown>,
      context: Record<string, unknown>,
    ) => Promise<{ filePath?: string }>,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

describe('UltraShape runtime flow', () => {
  it('completes a remote refinement path and packages one refined mesh output', async () => {
    const fixture = createFixtureWorkspace();
    const progress: UltraShapeProgressEvent[] = [];

    const remoteClient: UltraShapeExecutionClient = {
      async execute({ backend, onUpdate }) {
        expect(backend).toBe('remote');
        onUpdate?.({ stage: 'queued', message: 'Queued on remote worker.', progress: 35 });
        onUpdate?.({ stage: 'running', message: 'Remote refinement in progress.', progress: 70 });

        return {
          path: fixture.remoteMesh,
          format: 'obj',
          warnings: ['remote-handoff'],
        };
      },
    };

    try {
      const result = await executeUltraShapeRefiner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          params: {
            backend: 'auto',
            output_format: 'obj',
          },
        },
        {
          remoteClient,
          onProgress: (event) => progress.push(event),
          preflight: {
            hostPlatform: 'linux',
            hostArch: 'arm64',
            localAvailable: false,
            remoteAvailable: true,
          },
        },
      );

      expect(result.backendUsed).toBe('remote');
      expect(result.outputFormat).toBe('obj');
      expect(result.refinedMesh.path).toBe(join(fixture.outputDir, 'refined.obj'));
      expect(readFileSync(result.refinedMesh.path, 'utf8')).toBe('refined-mesh');
      expect(progress.map((event) => event.stage)).toEqual([
        'validating',
        'preflight',
        'running',
        'running',
        'running',
        'packaging',
        'completed',
      ]);
    } finally {
      fixture.cleanup();
    }
  });

  it('surfaces backend unavailable failures without reporting success', async () => {
    const fixture = createFixtureWorkspace();
    const progress: UltraShapeProgressEvent[] = [];

    try {
      await expect(
        executeUltraShapeRefiner(
          {
            reference_image: fixture.referenceImage,
            coarse_mesh: fixture.coarseMesh,
            output_dir: fixture.outputDir,
            params: {
              backend: 'remote',
            },
          },
          {
            onProgress: (event) => progress.push(event),
            preflight: {
              hostPlatform: 'linux',
              hostArch: 'arm64',
              localAvailable: false,
              remoteAvailable: true,
            },
          },
        ),
      ).rejects.toMatchObject({
        code: 'BACKEND_UNAVAILABLE',
      });

      expect(progress.at(-1)?.stage).toBe('failed');
      expect(progress.some((event) => event.stage === 'completed')).toBe(false);
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('emits cancelled state when the request aborts before completion', async () => {
    const fixture = createFixtureWorkspace();
    const progress: UltraShapeProgressEvent[] = [];
    const controller = new AbortController();

    const remoteClient: UltraShapeExecutionClient = {
      async execute({ signal, onUpdate }) {
        onUpdate?.({ stage: 'running', message: 'Remote refinement started.', progress: 50 });

        return await new Promise((_, reject) => {
          signal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true },
          );

          setTimeout(() => controller.abort(), 0);
        });
      },
    };

    try {
      await expect(
        executeUltraShapeRefiner(
          {
            reference_image: fixture.referenceImage,
            coarse_mesh: fixture.coarseMesh,
            output_dir: fixture.outputDir,
            abortSignal: controller.signal,
            params: {
              backend: 'remote',
            },
          },
          {
            remoteClient,
            onProgress: (event) => progress.push(event),
            preflight: {
              hostPlatform: 'linux',
              hostArch: 'arm64',
              localAvailable: false,
              remoteAvailable: true,
            },
          },
        ),
      ).rejects.toMatchObject({
        code: 'CANCELLED',
      });

      expect(progress.at(-1)?.stage).toBe('cancelled');
      expect(progress.some((event) => event.stage === 'completed')).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('maps named Modly inputs into the runtime request, merges default params, and returns ProcessResult.filePath', async () => {
    const fixture = createFixtureWorkspace();
    const installed = loadInstalledProcessor();
    const progressUpdates: Array<{ percent: number; label: string }> = [];
    const captured: Array<Record<string, unknown>> = [];

    const remoteClient = {
      async execute({ request, backend, signal }: { request: Record<string, unknown>; backend: string; signal?: AbortSignal }) {
        captured.push({ request, backend, signal });

        return {
          path: fixture.remoteMesh,
          format: 'obj',
        };
      },
    };

    try {
      const result = await installed.processor(
        {
          nodeId: 'ultrashape-refiner',
          inputs: {
            reference_image: {
              type: 'image',
              filePath: fixture.referenceImage,
              sourceNodeId: 'image-source',
            },
            coarse_mesh: {
              type: 'mesh',
              filePath: fixture.coarseMesh,
              sourceNodeId: 'mesh-source',
            },
          },
        },
        {
          output_format: 'obj',
          guidance_scale: 6.25,
        },
        {
          workspaceDir: fixture.outputDir,
          remoteClient,
          preflight: {
            hostPlatform: 'linux',
            hostArch: 'arm64',
            localAvailable: false,
            remoteAvailable: true,
          },
          progress: (percent: number, label: string) => progressUpdates.push({ percent, label }),
        },
      );

      expect(result).toEqual({
        filePath: join(fixture.outputDir, 'refined.obj'),
      });
      expect(readFileSync(join(fixture.outputDir, 'refined.obj'), 'utf8')).toBe('refined-mesh');
      expect(captured).toHaveLength(1);
      expect(captured[0]).toMatchObject({
        backend: 'remote',
        request: {
          referenceImage: {
            path: fixture.referenceImage,
            kind: 'image',
          },
          coarseMesh: {
            path: fixture.coarseMesh,
            kind: 'mesh',
          },
          outputDir: fixture.outputDir,
          params: {
            checkpoint: null,
            backend: 'auto',
            steps: 30,
            guidance_scale: 6.25,
            seed: null,
            preserve_scale: true,
            output_format: 'obj',
          },
          requestedBackend: 'auto',
        },
      });
      expect(progressUpdates.length).toBeGreaterThan(0);
    } finally {
      installed.cleanup();
      fixture.cleanup();
    }
  });

  it('falls back to tempDir for output resolution and forwards context.abortSignal to the runtime request', async () => {
    const fixture = createFixtureWorkspace();
    const installed = loadInstalledProcessor();
    const controller = new AbortController();
    const tempOutputDir = join(fixture.root, 'temp-output');
    mkdirSync(tempOutputDir);
    const seenSignals: AbortSignal[] = [];

    const remoteClient = {
      async execute({ signal }: { signal?: AbortSignal }) {
        if (signal) {
          seenSignals.push(signal);
        }

        controller.abort();
        throw new DOMException('Aborted', 'AbortError');
      },
    };

    try {
      await expect(
        installed.processor(
          {
            nodeId: 'ultrashape-refiner',
            inputs: {
              reference_image: {
                type: 'image',
                filePath: fixture.referenceImage,
                sourceNodeId: 'image-source',
              },
              coarse_mesh: {
                type: 'mesh',
                filePath: fixture.coarseMesh,
                sourceNodeId: 'mesh-source',
              },
            },
          },
          {
            backend: 'remote',
          },
          {
            tempDir: tempOutputDir,
            abortSignal: controller.signal,
            remoteClient,
            preflight: {
              hostPlatform: 'linux',
              hostArch: 'arm64',
              localAvailable: false,
              remoteAvailable: true,
            },
          },
        ),
      ).rejects.toMatchObject({
        code: 'CANCELLED',
      });

      expect(seenSignals).toEqual([controller.signal]);
      expect(existsSync(join(tempOutputDir, 'refined.glb'))).toBe(false);
    } finally {
      installed.cleanup();
      fixture.cleanup();
    }
  });

  it('surfaces backend-unavailable failures with an explicit code-bearing message for Modly runner logs', async () => {
    const fixture = createFixtureWorkspace();
    const installed = loadInstalledProcessor();

    try {
      await expect(
        installed.processor(
          {
            nodeId: 'ultrashape-refiner',
            inputs: {
              reference_image: {
                type: 'image',
                filePath: fixture.referenceImage,
                sourceNodeId: 'image-source',
              },
              coarse_mesh: {
                type: 'mesh',
                filePath: fixture.coarseMesh,
                sourceNodeId: 'mesh-source',
              },
            },
          },
          {
            backend: 'remote',
          },
          {
            workspaceDir: fixture.outputDir,
            preflight: {
              hostPlatform: 'linux',
              hostArch: 'arm64',
              localAvailable: false,
              remoteAvailable: true,
            },
          },
        ),
      ).rejects.toMatchObject({
        code: 'BACKEND_UNAVAILABLE',
        message: expect.stringContaining('BACKEND_UNAVAILABLE'),
      });
    } finally {
      installed.cleanup();
      fixture.cleanup();
    }
  });
});
