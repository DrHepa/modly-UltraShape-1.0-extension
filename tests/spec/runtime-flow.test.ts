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
  options: { env?: NodeJS.ProcessEnv; cwd?: string; processorPath?: string } = {},
) {
  const outcome = spawnSync('python3', [options.processorPath ?? processorPath], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      ...options.env,
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
      writeReadiness(fixture.root);

      const events = runProcessor(
        {
          extDir: fixture.root,
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
            backend: 'local',
            output_format: 'glb',
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
      writeReadiness(fixture.root, {
        status: 'blocked',
        required_imports_ok: false,
        missing_required: ['onnxruntime'],
      });

      const events = runProcessor(
        {
          extDir: fixture.root,
          input: {
            filePath: fixture.referenceImage,
          },
          params: {
            coarse_mesh: fixture.coarseMesh,
            backend: 'auto',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('DEPENDENCY_MISSING'),
        code: 'DEPENDENCY_MISSING',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('prefers the installed extension directory from processor.py before cwd fallback when extDir is omitted', () => {
    const fixture = createFixtureWorkspace();
    const installedExtDir = join(fixture.root, 'installed-extension');
    const installedProcessorPath = join(installedExtDir, 'processor.py');

    try {
      mkdirSync(installedExtDir);
      writeFileSync(installedProcessorPath, readFileSync(processorPath, 'utf8'));
      writeReadiness(installedExtDir);

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
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_TEST_ARTIFACT_PATH: fixture.packagedArtifact,
          },
          processorPath: installedProcessorPath,
        },
      );

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

  it('maps blocked local readiness without missing deps or weights to LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeReadiness(fixture.root, {
        status: 'blocked',
      });

      const events = runProcessor(
        {
          extDir: fixture.root,
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
            backend: 'auto',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });
});
