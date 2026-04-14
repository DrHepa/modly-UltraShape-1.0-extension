import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { normalizeRefinerRequest } from '../../src/processes/ultrashape-refiner/normalize.js';
import { preflightRefinerExecution } from '../../src/processes/ultrashape-refiner/preflight.js';
import { validateRefinerRequest } from '../../src/processes/ultrashape-refiner/validate.js';

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-refiner-'));
  const outputDir = join(root, 'output');
  mkdirSync(outputDir);

  const referenceImage = join(root, 'reference.png');
  const coarseMesh = join(root, 'coarse.glb');

  writeFileSync(referenceImage, 'image');
  writeFileSync(coarseMesh, 'mesh');

  return {
    root,
    outputDir,
    referenceImage,
    coarseMesh,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

describe('UltraShape request contract', () => {
  it('rejects missing assets with field-specific errors', () => {
    const fixture = createFixtureWorkspace();

    try {
      expect(() =>
        validateRefinerRequest({
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
        } as never),
      ).toThrowError(/reference_image/);

      expect(() =>
        validateRefinerRequest({
          reference_image: fixture.referenceImage,
          output_dir: fixture.outputDir,
        } as never),
      ).toThrowError(/coarse_mesh/);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects invalid params before runtime starts', () => {
    const fixture = createFixtureWorkspace();

    try {
      expect(() =>
        validateRefinerRequest({
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          params: {
            steps: 0,
          },
        }),
      ).toThrowError(/steps/);

      expect(() =>
        validateRefinerRequest({
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          params: {
            output_format: 'stl' as never,
          },
        }),
      ).toThrowError(/output_format/);
    } finally {
      fixture.cleanup();
    }
  });

  it('normalizes native and fallback requests with semantic parity', () => {
    const fixture = createFixtureWorkspace();

    try {
      const native = normalizeRefinerRequest({
        reference_image: { path: fixture.referenceImage, kind: 'image' },
        coarse_mesh: { path: fixture.coarseMesh, kind: 'mesh' },
        output_dir: fixture.outputDir,
        checkpoint: null,
        params: {
          backend: 'remote',
          steps: 40,
          guidance_scale: 6,
          preserve_scale: false,
          output_format: 'obj',
        },
        correlation_id: 'native-case',
      });

      const fallback = normalizeRefinerRequest({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        checkpoint: null,
        params: {
          backend: 'remote',
          steps: 40,
          guidance_scale: 6,
          preserve_scale: false,
          output_format: 'obj',
        },
        correlation_id: 'fallback-case',
      });

      expect(native.referenceImage).toEqual(fallback.referenceImage);
      expect(native.coarseMesh).toEqual(fallback.coarseMesh);
      expect(native.outputDir).toBe(fallback.outputDir);
      expect(native.params).toEqual(fallback.params);
      expect(native.requestedBackend).toBe('remote');
      expect(fallback.requestedBackend).toBe('remote');
    } finally {
      fixture.cleanup();
    }
  });

  it('prefers remote fallback on Linux ARM64 when local is unavailable', () => {
    const result = preflightRefinerExecution('auto', {
      hostPlatform: 'linux',
      hostArch: 'arm64',
      localAvailable: false,
      remoteAvailable: true,
    });

    expect(result.localSupported).toBe(false);
    expect(result.remoteSupported).toBe(true);
    expect(result.recommendedBackend).toBe('remote');
    expect(result.selectedBackend).toBe('remote');
    expect(result.fallbackApplied).toBe(true);
    expect(result.reason).toContain('Linux ARM64');
  });
});
