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
  options: { env?: NodeJS.ProcessEnv; cwd?: string } = {},
) {
  const outcome = spawnSync('python3', [processorPath], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      ...options.env,
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
        name: 'reference_image',
        label: 'Reference Image',
        type: 'image',
        required: true,
      },
      {
        name: 'coarse_mesh',
        label: 'Coarse Mesh',
        type: 'mesh',
        required: true,
      },
    ]);

    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root);

      const outcome = runProcessor(
        {
          extDir: fixture.root,
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
            backend: 'auto',
            coarse_mesh: join(fixture.root, 'missing-fallback.glb'),
            output_format: 'obj',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_TEST_ARTIFACT_PATH: fixture.packagedArtifact,
          },
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
      expect(resolved.backend).toBe('local');
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

  it('treats degraded readiness with missing required weights as a blocked install contract, not a normal WEIGHTS_MISSING outcome', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root, {
        status: 'degraded',
        weights_ready: false,
        missing_required: ['models/ultrashape/ultrashape_v1.pt'],
      });

      const outcome = runProcessor(
        {
          extDir: fixture.root,
          input: {
            filePath: fixture.referenceImage,
          },
          params: {
            backend: 'local',
            coarse_mesh: fixture.fallbackCoarseMesh,
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(outcome.status).toBe(0);

      const resolutionLog = outcome.events[1];
      const resolved = JSON.parse(String(resolutionLog.message)) as Record<string, string>;
      expect(resolved.reference_image).toBe(fixture.referenceImage);
      expect(resolved.coarse_mesh).toBe(fixture.fallbackCoarseMesh);
      expect(resolved.backend).toBe('local');
      expect(resolved.output_format).toBe('glb');

      const error = outcome.events.at(-1);
      expect(error).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('reserves WEIGHTS_MISSING for post-install drift when readiness still claims ready', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root, {
        status: 'ready',
        weights_ready: false,
        missing_required: ['models/ultrashape/ultrashape_v1.pt'],
      });

      const outcome = runProcessor(
        {
          extDir: fixture.root,
          input: {
            filePath: fixture.referenceImage,
          },
          params: {
            backend: 'local',
            coarse_mesh: fixture.fallbackCoarseMesh,
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('WEIGHTS_MISSING'),
        code: 'WEIGHTS_MISSING',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects remote and hybrid requests explicitly because this MVP is local-only', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root);

      for (const backend of ['remote', 'hybrid'] as const) {
        const outcome = runProcessor(
          {
            extDir: fixture.root,
            input: {
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
              backend,
            },
            workspaceDir: fixture.outputDir,
          },
          {
            cwd: fixture.root,
          },
        );

        expect(outcome.events.at(-1)).toEqual({
          type: 'error',
          message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
          code: 'LOCAL_RUNTIME_UNAVAILABLE',
        });
      }
    } finally {
      fixture.cleanup();
    }
  });
});
