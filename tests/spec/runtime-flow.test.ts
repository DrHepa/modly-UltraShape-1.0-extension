import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import type { UltraShapeExecutionClient } from '../../src/adapters/ultrashape/client.js';
import { executeUltraShapeRefiner } from '../../src/processes/ultrashape-refiner/index.js';
import type { UltraShapeProgressEvent } from '../../src/processes/ultrashape-refiner/types.js';

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
});
