import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const processorPath = resolve(repoRoot, 'processor.py');

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-runtime-'));
  const outputDir = join(root, 'output');
  mkdirSync(outputDir);

  const referenceImage = join(root, 'reference.png');
  const coarseMesh = join(root, 'coarse.glb');
  const packagedArtifact = join(root, 'artifact.glb');

  writeFileSync(referenceImage, 'image');
  writeFileSync(coarseMesh, 'mesh');
  writeFileSync(packagedArtifact, 'refined-mesh');

  return {
    root,
    outputDir,
    referenceImage,
    coarseMesh,
    packagedArtifact,
    cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

function runProcessor(payload: Record<string, unknown>, env: NodeJS.ProcessEnv = {}) {
  const outcome = spawnSync('python3', [processorPath], {
    cwd: repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      ...env,
    },
  });

  return outcome.stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe('UltraShape runtime flow', () => {
  it('runs the repo-root Python boundary from the named-input contract and packages refined.<format> without any JS install artifact', () => {
    const fixture = createFixtureWorkspace();

    try {
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
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          ULTRASHAPE_TEST_ARTIFACT_PATH: fixture.packagedArtifact,
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
      expect(readFileSync(join(fixture.outputDir, 'refined.glb'), 'utf8')).toBe('refined-mesh');
    } finally {
      fixture.cleanup();
    }
  });

  it('keeps the fallback seam compatible when named inputs are absent and validation still succeeds', () => {
    const fixture = createFixtureWorkspace();

    try {
      const events = runProcessor({
        input: {
          filePath: fixture.referenceImage,
        },
        params: {
          coarse_mesh: fixture.coarseMesh,
          backend: 'remote',
        },
        workspaceDir: fixture.outputDir,
      });

      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('BACKEND_UNAVAILABLE'),
        code: 'BACKEND_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });
});
