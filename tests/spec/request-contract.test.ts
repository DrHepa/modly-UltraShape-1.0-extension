import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { normalizeRefinerRequest } from '../../src/processes/ultrashape-refiner/normalize.js';
import { preflightRefinerExecution } from '../../src/processes/ultrashape-refiner/preflight.js';
import { validateRefinerRequest } from '../../src/processes/ultrashape-refiner/validate.js';

const processorPath = resolve(process.cwd(), 'processor.py');

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
  it('reads exactly one JSON object line from stdin and ignores trailing lines', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input:
          `${JSON.stringify({
            input: { filePath: fixture.referenceImage },
            params: { coarse_mesh: fixture.coarseMesh },
            workspaceDir: fixture.outputDir,
          })}\n${JSON.stringify({ should_be_ignored: true })}\n`,
      });

      const events = outcome.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(outcome.status).toBe(0);
      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('BACKEND_UNAVAILABLE'),
        code: 'BACKEND_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('fails fast when fallback payload omits params.coarse_mesh', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: { filePath: fixture.referenceImage },
          params: {},
          workspaceDir: fixture.outputDir,
        })}\n`,
      });

      const events = outcome.stdout
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(outcome.status).toBe(0);
      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('coarse_mesh'),
        code: 'MISSING_INPUT',
      });
    } finally {
      fixture.cleanup();
    }
  });

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
