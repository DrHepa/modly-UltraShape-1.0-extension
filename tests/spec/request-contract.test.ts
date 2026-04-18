import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

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

function parseEvents(stdout: string) {
  return stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe('UltraShape request contract', () => {
  it('reads exactly one JSON object line from stdin, prioritizes named inputs, and ignores trailing lines', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input:
          `${JSON.stringify({
            input: {
              filePath: join(fixture.root, 'missing-reference.png'),
              inputs: {
                reference_image: { filePath: fixture.referenceImage },
                coarse_mesh: { filePath: fixture.coarseMesh },
              },
            },
            params: { coarse_mesh: join(fixture.root, 'missing-fallback.glb') },
            workspaceDir: fixture.outputDir,
          })}\n${JSON.stringify({ should_be_ignored: true })}\n`,
      });

      const events = parseEvents(outcome.stdout);

      expect(outcome.status).toBe(0);
      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('fails fast when the secondary fallback payload omits params.coarse_mesh', () => {
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

      const events = parseEvents(outcome.stdout);

      expect(outcome.status).toBe(0);
      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('coarse_mesh'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects missing named inputs with field-specific public errors', () => {
    const fixture = createFixtureWorkspace();

    try {
      const missingReference = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });
      const missingCoarse = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            filePath: fixture.referenceImage,
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });

      expect(missingReference.status).toBe(0);
      expect(parseEvents(missingReference.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('reference_image'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });

      expect(missingCoarse.status).toBe(0);
      expect(parseEvents(missingCoarse.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('coarse_mesh'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects invalid params before runtime starts', () => {
    const fixture = createFixtureWorkspace();

    try {
      const invalidSteps = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          params: {
            steps: 0,
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });
      const invalidOutput = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          params: {
            output_format: 'obj',
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });

      expect(invalidSteps.status).toBe(0);
      expect(parseEvents(invalidSteps.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('steps must be a positive integer'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });

      expect(invalidOutput.status).toBe(0);
      expect(parseEvents(invalidOutput.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('output_format must be glb'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('accepts both named-input and temporary fallback payload shapes up to runtime readiness checks', () => {
    const fixture = createFixtureWorkspace();

    try {
      const namedOutcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          params: {
            backend: 'local',
            steps: 40,
            guidance_scale: 6,
            preserve_scale: false,
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });
      const fallbackOutcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            filePath: fixture.referenceImage,
          },
          params: {
            coarse_mesh: fixture.coarseMesh,
            backend: 'local',
            steps: 40,
            guidance_scale: 6,
            preserve_scale: false,
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });

      expect(namedOutcome.status).toBe(0);
      expect(parseEvents(namedOutcome.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });

      expect(fallbackOutcome.status).toBe(0);
      expect(parseEvents(fallbackOutcome.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects remote and hybrid backend values at the public processor boundary', () => {
    const fixture = createFixtureWorkspace();

    try {
      const remoteOutcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          params: {
            backend: 'remote',
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });
      const hybridOutcome = spawnSync('python3', [processorPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        input: `${JSON.stringify({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.coarseMesh },
            },
          },
          params: {
            backend: 'hybrid',
          },
          workspaceDir: fixture.outputDir,
        })}\n`,
      });

      expect(remoteOutcome.status).toBe(0);
      expect(parseEvents(remoteOutcome.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('backend must be auto or local'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });

      expect(hybridOutcome.status).toBe(0);
      expect(parseEvents(hybridOutcome.stdout).at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('backend must be auto or local'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });
});
