import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const processorPath = resolve(repoRoot, 'processor.py');
const manifestPath = resolve(repoRoot, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
  nodes: Array<{
    id: string;
    inputs?: Array<Record<string, unknown>>;
  }>;
};

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-processor-'));
  const outputDir = join(root, 'output');
  mkdirSync(outputDir);

  const referenceImage = join(root, 'reference.png');
  const namedCoarseMesh = join(root, 'named-coarse.glb');
  const fallbackCoarseMesh = join(root, 'fallback-coarse.obj');
  const packagedArtifact = join(root, 'artifact.obj');

  writeFileSync(referenceImage, 'image');
  writeFileSync(namedCoarseMesh, 'named-mesh');
  writeFileSync(fallbackCoarseMesh, 'fallback-mesh');
  writeFileSync(packagedArtifact, 'refined-artifact');

  return {
    root,
    outputDir,
    referenceImage,
    namedCoarseMesh,
    fallbackCoarseMesh,
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

  return {
    ...outcome,
    events: outcome.stdout
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Record<string, unknown>),
  };
}

describe('UltraShape processor.py protocol', () => {
  it('treats named reference_image and coarse_mesh inputs as the primary manifest contract and only uses params.coarse_mesh as fallback metadata', () => {
    expect(manifest.nodes[0]?.inputs).toEqual([
      {
        id: 'reference_image',
        label: 'Reference Image',
        type: 'image',
        required: true,
      },
      {
        id: 'coarse_mesh',
        label: 'Coarse Mesh',
        type: 'mesh',
        required: true,
      },
    ]);

    const fixture = createFixtureWorkspace();

    try {
      const outcome = runProcessor(
        {
          nodeId: 'ultrashape-refiner',
          input: {
            filePath: fixture.referenceImage,
            inputs: {
              reference_image: {
                filePath: fixture.referenceImage,
              },
              coarse_mesh: {
                filePath: fixture.namedCoarseMesh,
              },
            },
          },
          params: {
            coarse_mesh: join(fixture.root, 'missing-fallback.glb'),
            output_format: 'obj',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          ULTRASHAPE_TEST_ARTIFACT_PATH: fixture.packagedArtifact,
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.events.map((event) => event.type)).toEqual([
        'progress',
        'log',
        'progress',
        'progress',
        'progress',
        'done',
      ]);

      const resolutionLog = outcome.events[1];
      const resolved = JSON.parse(String(resolutionLog.message)) as Record<string, string>;
      expect(resolved.reference_image).toBe(fixture.referenceImage);
      expect(resolved.coarse_mesh).toBe(fixture.namedCoarseMesh);
      expect(resolved.output_format).toBe('obj');

      const done = outcome.events.at(-1);
      expect(done).toEqual({
        type: 'done',
        result: {
          filePath: join(fixture.outputDir, 'refined.obj'),
        },
      });
      expect(existsSync(join(fixture.outputDir, 'refined.obj'))).toBe(true);
      expect(readFileSync(join(fixture.outputDir, 'refined.obj'), 'utf8')).toBe('refined-artifact');
    } finally {
      fixture.cleanup();
    }
  });

  it('uses input.filePath plus params.coarse_mesh only as a secondary fallback seam when named inputs are absent', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = runProcessor({
        input: {
          filePath: fixture.referenceImage,
        },
        params: {
          coarse_mesh: fixture.fallbackCoarseMesh,
        },
        workspaceDir: fixture.outputDir,
      });

      expect(outcome.status).toBe(0);

      const resolutionLog = outcome.events[1];
      const resolved = JSON.parse(String(resolutionLog.message)) as Record<string, string>;
      expect(resolved.reference_image).toBe(fixture.referenceImage);
      expect(resolved.coarse_mesh).toBe(fixture.fallbackCoarseMesh);
      expect(resolved.output_format).toBe('glb');

      const error = outcome.events.at(-1);
      expect(error).toEqual({
        type: 'error',
        message: expect.stringContaining('BACKEND_UNAVAILABLE'),
        code: 'BACKEND_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });
});
