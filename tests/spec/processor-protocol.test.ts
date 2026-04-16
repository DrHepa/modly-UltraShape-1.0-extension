import { chmodSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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

function installFakeRunnerExtension(root: string) {
  const extDir = join(root, 'installed-extension');
  const venvBinDir = join(extDir, 'venv', 'bin');
  const runtimeDir = join(extDir, 'runtime');
  const runtimePackageDir = join(runtimeDir, 'ultrashape_runtime');
  const runtimeConfigDir = join(runtimeDir, 'configs');
  const modelsDir = join(extDir, 'models', 'ultrashape');
  const pythonShimPath = join(venvBinDir, 'python');
  const configPath = join(runtimeConfigDir, 'infer_dit_refine.yaml');
  const checkpointPath = join(modelsDir, 'ultrashape_v1.pt');
  const invocationArgsPath = join(extDir, 'runner-args.txt');
  const invocationEnvPath = join(extDir, 'runner-env.txt');
  const invocationInputPath = join(extDir, 'runner-input.json');

  mkdirSync(venvBinDir, { recursive: true });
  mkdirSync(runtimePackageDir, { recursive: true });
  mkdirSync(runtimeConfigDir, { recursive: true });
  mkdirSync(modelsDir, { recursive: true });

  writeFileSync(configPath, 'model:\n  scope: mc-only\nruntime:\n  backend: local\nsurface:\n  extraction: mc\n');
  writeFileSync(checkpointPath, 'weights');
  writeFileSync(join(runtimePackageDir, '__init__.py'), '');
  writeFileSync(
    join(runtimePackageDir, 'local_runner.py'),
    [
      'import json',
      'import os',
      'import sys',
      'from pathlib import Path',
      '',
      'def main():',
      '    payload = json.loads(sys.stdin.readline())',
      '    capture_path = os.environ.get("ULTRASHAPE_RUNNER_CAPTURE_PATH")',
      '    if capture_path:',
      '        Path(capture_path).write_text(json.dumps(payload), encoding="utf8")',
      '    mode = os.environ.get("ULTRASHAPE_RUNNER_STDOUT_MODE", "success")',
      '    output_dir = Path(payload["output_dir"])',
      '    output_dir.mkdir(parents=True, exist_ok=True)',
      '    inside_output = output_dir / "refined.glb"',
      '    outside_output = output_dir.parent / "outside.glb"',
      '    if mode == "invalid-json":',
      '        sys.stdout.write("not-json")',
      '        return 0',
      '    if mode == "outside-output":',
      '        outside_output.write_text("outside", encoding="utf8")',
      '        sys.stdout.write(json.dumps({"ok": True, "result": {"file_path": str(outside_output), "format": "glb", "backend": "local", "warnings": []}}))',
      '        return 0',
      '    if mode == "dependency-error":',
      '        sys.stdout.write(json.dumps({"ok": False, "error_code": "DEPENDENCY_MISSING", "error_message": "missing dependency"}))',
      '        return 1',
      '    if mode == "weights-error":',
      '        sys.stdout.write(json.dumps({"ok": False, "error_code": "WEIGHTS_MISSING", "error_message": "missing checkpoint"}))',
      '        return 1',
      '    inside_output.write_text("runner-output", encoding="utf8")',
      '    sys.stdout.write(json.dumps({"ok": True, "result": {"file_path": str(inside_output), "format": "glb", "backend": "local", "warnings": []}}))',
      '    return 0',
      '',
      'if __name__ == "__main__":',
      '    raise SystemExit(main())',
      '',
    ].join('\n'),
  );
  writeFileSync(
    pythonShimPath,
    [
      '#!/usr/bin/env bash',
      'set -euo pipefail',
      `printf '%s' "$*" > "${invocationArgsPath}"`,
      `printf '%s' "\${PYTHONPATH:-}" > "${invocationEnvPath}"`,
      'payload="$(cat)"',
      `printf '%s' "$payload" > "${invocationInputPath}"`,
      'printf "%s" "$payload" | exec python3 "$@"',
      '',
    ].join('\n'),
  );
  chmodSync(pythonShimPath, 0o755);

  return {
    extDir,
    configPath,
    checkpointPath,
    invocationArgsPath,
    invocationEnvPath,
    invocationInputPath,
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

function runLocalRunner(
  payload: Record<string, unknown>,
  options: { env?: NodeJS.ProcessEnv; cwd?: string } = {},
) {
  const outcome = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      PYTHONPATH: resolve(repoRoot, 'runtime', 'vendor'),
      ...options.env,
    },
  });

  return {
    ...outcome,
    result: outcome.stdout ? (JSON.parse(outcome.stdout) as Record<string, unknown>) : null,
  };
}

describe('UltraShape processor.py protocol', () => {
  it('treats named reference_image and coarse_mesh inputs as the primary manifest contract and rejects non-glb output requests for the local-only MVP', () => {
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
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.events.map((event) => event.type)).toEqual([
        'progress',
        'log',
        'error',
      ]);

      const resolutionLog = outcome.events[1];
      const resolved = JSON.parse(String(resolutionLog.message)) as Record<string, string>;
      expect(resolved.reference_image).toBe(fixture.referenceImage);
      expect(resolved.coarse_mesh).toBe(fixture.namedCoarseMesh);
      expect(resolved.backend).toBe('local');
      expect(resolved.output_format).toBe('obj');

      const error = outcome.events.at(-1);
      expect(error).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
      expect(existsSync(join(fixture.outputDir, 'refined.obj'))).toBe(false);
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

  it('exposes a local-only glb-only runner seam that writes refined.glb without ULTRASHAPE_TEST_ARTIFACT_PATH fallback', () => {
    const fixture = createFixtureWorkspace();
    const configPath = join(fixture.root, 'runtime-config.yaml');
    const checkpointPath = join(fixture.root, 'ultrashape_v1.pt');

    writeFileSync(configPath, 'scope: mc-only\n');
    writeFileSync(checkpointPath, 'weights');

    try {
      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.namedCoarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: checkpointPath,
          config_path: configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_TEST_ARTIFACT_PATH: fixture.packagedArtifact,
          },
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.result).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          warnings: [],
        },
      });
      expect(readFileSync(join(fixture.outputDir, 'refined.glb'), 'utf8')).not.toBe('refined-artifact');
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-glb runner jobs with a structured local runtime error envelope', () => {
    const fixture = createFixtureWorkspace();
    const configPath = join(fixture.root, 'runtime-config.yaml');

    writeFileSync(configPath, 'scope: mc-only\n');

    try {
      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.namedCoarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'obj',
          checkpoint: null,
          config_path: configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: null,
          preserve_scale: true,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(outcome.status).toBe(1);
      expect(outcome.result).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('glb-only'),
      });
      expect(existsSync(join(fixture.outputDir, 'refined.obj'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('looks up the installed venv/config runner contract and invokes python -m ultrashape_runtime.local_runner with PYTHONPATH=ext_dir/runtime', () => {
    const fixture = createFixtureWorkspace();
    const installed = installFakeRunnerExtension(fixture.root);

    try {
      writeReadiness(installed.extDir);

      const outcome = runProcessor(
        {
          extDir: installed.extDir,
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
            backend: 'auto',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_RUNNER_CAPTURE_PATH: installed.invocationInputPath,
          },
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.events.at(-1)).toEqual({
        type: 'done',
        result: {
          filePath: join(fixture.outputDir, 'refined.glb'),
        },
      });

      expect(readFileSync(installed.invocationArgsPath, 'utf8')).toBe('-m ultrashape_runtime.local_runner');
      expect(readFileSync(installed.invocationEnvPath, 'utf8')).toBe(join(installed.extDir, 'runtime'));
      expect(JSON.parse(readFileSync(installed.invocationInputPath, 'utf8'))).toEqual({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.namedCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: installed.checkpointPath,
        config_path: installed.configPath,
        ext_dir: installed.extDir,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });
      expect(readFileSync(join(fixture.outputDir, 'refined.glb'), 'utf8')).toBe('runner-output');
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing venv/config or invalid runner stdout to LOCAL_RUNTIME_UNAVAILABLE instead of fake packaging success', () => {
    const fixture = createFixtureWorkspace();
    const installed = installFakeRunnerExtension(fixture.root);

    try {
      writeReadiness(installed.extDir);
      rmSync(join(installed.extDir, 'venv'), { recursive: true, force: true });

      const missingVenvOutcome = runProcessor(
        {
          extDir: installed.extDir,
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
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(missingVenvOutcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });

      const reinstalled = installFakeRunnerExtension(fixture.root);
      writeReadiness(reinstalled.extDir);

      const invalidStdoutOutcome = runProcessor(
        {
          extDir: reinstalled.extDir,
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
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_RUNNER_STDOUT_MODE: 'invalid-json',
          },
        },
      );

      expect(invalidStdoutOutcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects runner-reported outputs outside the requested output directory', () => {
    const fixture = createFixtureWorkspace();
    const installed = installFakeRunnerExtension(fixture.root);

    try {
      writeReadiness(installed.extDir);

      const outcome = runProcessor(
        {
          extDir: installed.extDir,
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
            backend: 'local',
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
          env: {
            ULTRASHAPE_RUNNER_STDOUT_MODE: 'outside-output',
          },
        },
      );

      expect(outcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('outside the requested output directory'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
      expect(existsSync(join(fixture.root, 'outside.glb'))).toBe(true);
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('maps structured runner failures back onto the public processor error contract', () => {
    const fixture = createFixtureWorkspace();
    const installed = installFakeRunnerExtension(fixture.root);

    try {
      writeReadiness(installed.extDir);

      for (const [mode, code] of [
        ['dependency-error', 'DEPENDENCY_MISSING'],
        ['weights-error', 'WEIGHTS_MISSING'],
      ] as const) {
        const outcome = runProcessor(
          {
            extDir: installed.extDir,
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
              backend: 'local',
              output_format: 'glb',
            },
            workspaceDir: fixture.outputDir,
          },
          {
            cwd: fixture.root,
            env: {
              ULTRASHAPE_RUNNER_STDOUT_MODE: mode,
            },
          },
        );

        expect(outcome.events.at(-1)).toEqual({
          type: 'error',
          message: expect.stringContaining(code),
          code,
        });
      }
    } finally {
      fixture.cleanup();
    }
  });
});
