import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vitest';

import { UltraShapeLocalAdapter } from '../../src/adapters/ultrashape/local.js';
import { runRefinerRuntime } from '../../src/processes/ultrashape-refiner/runtime.js';
import type { UltraShapeNormalizedRequest, UltraShapePreflightResult } from '../../src/processes/ultrashape-refiner/types.js';

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-ts-runtime-'));
  const outputDir = join(root, 'output');
  const artifactPath = join(root, 'artifact.glb');

  mkdirSync(outputDir);
  writeFileSync(artifactPath, 'refined-mesh');

  return {
    root,
    outputDir,
    artifactPath,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

function createRequest(outputDir: string): UltraShapeNormalizedRequest {
  return {
    correlationId: 'ts-runtime',
    referenceImage: { path: '/tmp/reference.png', kind: 'image', mediaType: 'image/png' },
    coarseMesh: { path: '/tmp/coarse.glb', kind: 'mesh', mediaType: 'model/gltf-binary' },
    outputDir,
    params: {
      checkpoint: null,
      backend: 'local',
      steps: 30,
      guidance_scale: 5.5,
      seed: null,
      preserve_scale: true,
      output_format: 'glb',
    },
    requestedBackend: 'local',
  };
}

function createPreflight(selectedBackend: UltraShapePreflightResult['selectedBackend']): UltraShapePreflightResult {
  return {
    hostPlatform: 'linux',
    hostArch: 'arm64',
    localSupported: true,
    recommendedBackend: 'local',
    selectedBackend,
    fallbackApplied: false,
    reason: 'UltraShape local runtime is the only active backend in this MVP.',
  };
}

describe('UltraShape TypeScript runtime boundary', () => {
  it('packages local adapter artifacts through the local-first runtime path', async () => {
    const fixture = createFixtureWorkspace();

    try {
      const request = createRequest(fixture.outputDir);
      const result = await runRefinerRuntime(request, createPreflight('local'), {
        localAdapter: {
          backend: 'local',
          run: vi.fn(async () => ({
            path: fixture.artifactPath,
            format: 'glb' as const,
            warnings: ['compatibility seam still supported'],
          })),
        },
      });

      expect(result.backendUsed).toBe('local');
      expect(result.refinedMesh.path).toBe(join(fixture.outputDir, 'refined.glb'));
      expect(result.warnings).toEqual(['compatibility seam still supported']);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects corrupted preflight state that bypasses the local-only contract', async () => {
    const fixture = createFixtureWorkspace();

    try {
      const corruptedPreflight = {
        ...createPreflight('local'),
        selectedBackend: 'remote',
      } as unknown as UltraShapePreflightResult;

      await expect(
        runRefinerRuntime(createRequest(fixture.outputDir), corruptedPreflight),
      ).rejects.toMatchObject({
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('marks the local adapter fallback message as compatibility-only runtime unavailability', async () => {
    await expect(new UltraShapeLocalAdapter().run({ request: createRequest('/tmp'), backend: 'local' })).rejects.toMatchObject({
      code: 'LOCAL_RUNTIME_UNAVAILABLE',
    });
  });
});
