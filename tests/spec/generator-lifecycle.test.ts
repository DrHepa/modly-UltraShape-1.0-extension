import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  copyInstallSurface,
  createRuntimeInputs,
  runGeneratorProbe,
  stageCheckpoint,
  writeRuntimeStubModules,
} from './install-test-helpers.js';

const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==';

function runSetup(cwd: string, extDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['-S', 'setup.py', '--ext-dir', extDir], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ULTRASHAPE_SETUP_TEST_STUB_DEPS: '1',
      ULTRASHAPE_SETUP_TEST_HOST_PLATFORM: 'linux',
      ULTRASHAPE_SETUP_TEST_HOST_MACHINE: 'aarch64',
      ...env,
    },
  });
}

function writeForwardingReadiness(
  checkout: string,
  fields: {
    upstreamCheckout: string;
    upstreamConfig: string;
    checkpoint: string;
    pythonExe: string;
    venvDir: string;
    flashAttnPolicy?: Record<string, unknown>;
  },
  options: { stageAssets?: boolean } = {},
) {
  if (options.stageAssets !== false) {
    mkdirSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'), { recursive: true });
    mkdirSync(path.join(checkout, 'runtime', 'configs'), { recursive: true });
    mkdirSync(path.dirname(fields.checkpoint), { recursive: true });
    writeFileSync(path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'), 'runtime: {}\n', 'utf8');
    writeFileSync(fields.checkpoint, 'weights', 'utf8');
  }
  writeFileSync(
    path.join(checkout, '.runtime-readiness.json'),
    JSON.stringify({
      status: 'ready',
      required_imports_ok: true,
      weights_ready: true,
      config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
      checkpoint: fields.checkpoint,
      python_exe: fields.pythonExe,
      venv_dir: fields.venvDir,
      runtime_modes: {
        requested: 'auto',
        active: 'real',
        real: {
          available: true,
          checkout_path: fields.upstreamCheckout,
          upstream_config: { available: true, path: fields.upstreamConfig },
          config: { available: true, path: fields.upstreamConfig },
          checkpoint: { available: true, path: fields.checkpoint },
          attention_backend: fields.flashAttnPolicy ? 'sdpa' : undefined,
          flash_attn_policy: fields.flashAttnPolicy,
        },
        portable: { available: true },
      },
    }),
    'utf8',
  );
}

function runGeneratorDiagnosticProbe(checkout: string, source: string) {
  return spawnSync('python3', ['-S', '-c', source], {
    cwd: checkout,
    encoding: 'utf8',
    env: process.env,
  });
}

function runRunnerJobProbe(checkout: string, env: NodeJS.ProcessEnv = {}) {
  const script = [
    'import json, os, sys',
    'from pathlib import Path',
    'from generator import UltraShapeGenerator',
    'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
    'readiness = generator._require_runtime_ready()',
    'job = generator._build_runner_job(readiness=readiness, reference_image=Path.cwd() / "reference.png", coarse_mesh=Path.cwd() / "coarse.glb", output_dir=Path.cwd() / "outputs", params={"steps": 7, "guidance_scale": 2.5, "seed": 123, "preserve_scale": True})',
    'print(json.dumps(job, sort_keys=True))',
  ].join('\n');

  return spawnSync('python3', ['-S', '-c', script], {
    cwd: checkout,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

describe('generator lifecycle shell', () => {
  it('diagnoses stale installed roots when readiness is missing or staged assets disappeared', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-stale-missing-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const missingReadiness = runGeneratorDiagnosticProbe(
        checkout,
        [
          'import json',
          'from pathlib import Path',
          'from generator import PublicRuntimeError, UltraShapeGenerator',
          'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
          'try:',
          '    generator.load()',
          'except PublicRuntimeError as error:',
          '    print(json.dumps({"code": error.code, "message": str(error)}))',
        ].join('\n'),
      );
      expect(missingReadiness.status).toBe(0);
      expect(JSON.parse(missingReadiness.stdout)).toEqual({
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
        message: expect.stringContaining('installed extension root is stale or incomplete'),
      });
      expect(JSON.parse(missingReadiness.stdout).message).toContain('.runtime-readiness.json missing');

      writeForwardingReadiness(checkout, {
        upstreamCheckout: path.join(sandbox, 'upstream'),
        upstreamConfig: path.join(sandbox, 'upstream', 'configs', 'infer_dit_refine.yaml'),
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'missing-real.pt'),
        pythonExe: path.join(checkout, 'venv', 'bin', 'python'),
        venvDir: path.join(checkout, 'venv'),
      }, { stageAssets: false });
      rmSync(path.join(checkout, 'runtime', 'vendor'), { recursive: true, force: true });
      rmSync(path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'), { force: true });
      const missingAssets = runGeneratorDiagnosticProbe(
        checkout,
        [
          'import json',
          'from pathlib import Path',
          'from generator import PublicRuntimeError, UltraShapeGenerator',
          'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
          'try:',
          '    generator.load()',
          'except PublicRuntimeError as error:',
          '    print(json.dumps({"code": error.code, "message": str(error)}))',
        ].join('\n'),
      );

      expect(missingAssets.status).toBe(0);
      const assetMessage = JSON.parse(missingAssets.stdout).message as string;
      expect(assetMessage).toContain('staged vendor path missing');
      expect(assetMessage).toContain('runtime config path missing');
      expect(assetMessage).toContain('checkpoint path missing');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('diagnoses readiness recorded for a different or suspicious installed extension root', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-stale-paths-'));
    const checkout = path.join(sandbox, 'repo');
    const otherInstallRoot = path.join(sandbox, 'other-install');
    const suspiciousRoot = path.join(sandbox, 'source-repo');
    copyInstallSurface(checkout);
    mkdirSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime'), { recursive: true });
    mkdirSync(path.join(checkout, 'runtime', 'configs'), { recursive: true });
    mkdirSync(path.join(checkout, 'models', 'ultrashape'), { recursive: true });
    writeFileSync(path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'), 'runtime: {}\n', 'utf8');
    writeFileSync(path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'), 'weights', 'utf8');

    try {
      writeFileSync(
        path.join(checkout, '.runtime-readiness.json'),
        JSON.stringify({
          status: 'ready',
          required_imports_ok: true,
          weights_ready: true,
          ext_dir: otherInstallRoot,
          vendor_path: path.join(suspiciousRoot, 'runtime', 'vendor', 'ultrashape_runtime'),
          config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
          checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        }),
        'utf8',
      );

      const result = runGeneratorDiagnosticProbe(
        checkout,
        [
          'import json',
          'from pathlib import Path',
          'from generator import PublicRuntimeError, UltraShapeGenerator',
          'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
          'try:',
          '    generator.load()',
          'except PublicRuntimeError as error:',
          '    print(json.dumps({"code": error.code, "message": str(error)}))',
        ].join('\n'),
      );

      expect(result.status).toBe(0);
      const message = JSON.parse(result.stdout).message as string;
      expect(message).toContain('readiness ext_dir mismatch');
      expect(message).toContain(otherInstallRoot);
      expect(message).toContain('vendor_path points outside installed extension root');
      expect(message).toContain(suspiciousRoot);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('forwards setup-recorded real readiness fields into the runner job', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-readiness-forward-'));
    const checkout = path.join(sandbox, 'repo');
    const upstreamCheckout = path.join(sandbox, 'upstream-a');
    const upstreamConfig = path.join(upstreamCheckout, 'configs', 'infer_dit_refine.yaml');
    const checkpoint = path.join(checkout, 'models', 'ultrashape', 'real.pt');
    const venvDir = path.join(checkout, 'venv');
    copyInstallSurface(checkout);

    try {
      writeForwardingReadiness(checkout, {
        upstreamCheckout,
        upstreamConfig,
        checkpoint,
        pythonExe: path.join(venvDir, 'bin', 'python'),
        venvDir,
      });

      const result = runRunnerJobProbe(checkout);

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        runtime_mode: 'auto',
        upstream_checkout_path: upstreamCheckout,
        upstream_config_path: upstreamConfig,
        checkpoint,
        python_exe: path.join(venvDir, 'bin', 'python'),
        venv_dir: venvDir,
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('uses runtime readiness as authority while setup summary contributes diagnostics only', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-summary-diagnostics-'));
    const checkout = path.join(sandbox, 'repo');
    const readinessCheckout = path.join(sandbox, 'readiness-upstream');
    const readinessConfig = path.join(readinessCheckout, 'configs', 'infer_dit_refine.yaml');
    const summaryCheckout = path.join(sandbox, 'summary-upstream');
    const summaryConfig = path.join(summaryCheckout, 'configs', 'infer_dit_refine.yaml');
    const readinessCheckpoint = path.join(checkout, 'models', 'ultrashape', 'readiness.pt');
    const readinessVenv = path.join(checkout, 'venv-readiness');
    copyInstallSurface(checkout);

    try {
      writeForwardingReadiness(checkout, {
        upstreamCheckout: readinessCheckout,
        upstreamConfig: readinessConfig,
        checkpoint: readinessCheckpoint,
        pythonExe: path.join(readinessVenv, 'bin', 'python'),
        venvDir: readinessVenv,
      });
      writeFileSync(
        path.join(checkout, '.setup-summary.json'),
        JSON.stringify({
          diagnostics: ['summary-only diagnostic'],
          runtime_modes: {
            real: {
              checkout_path: summaryCheckout,
              upstream_config: { path: summaryConfig },
            },
          },
        }),
        'utf8',
      );

      const result = runRunnerJobProbe(checkout);

      expect(result.status).toBe(0);
      const job = JSON.parse(result.stdout);
      expect(job.upstream_checkout_path).toBe(readinessCheckout);
      expect(job.upstream_config_path).toBe(readinessConfig);
      expect(JSON.stringify(job)).not.toContain(summaryCheckout);
      expect(JSON.stringify(job)).not.toContain(summaryConfig);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('applies override precedence: explicit env beats readiness and readiness beats defaults', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-precedence-'));
    const checkout = path.join(sandbox, 'repo');
    const readinessCheckout = path.join(sandbox, 'readiness-upstream');
    const readinessConfig = path.join(readinessCheckout, 'configs', 'infer_dit_refine.yaml');
    const envCheckout = path.join(sandbox, 'env-upstream');
    const envConfig = path.join(envCheckout, 'configs', 'env.yaml');
    const readinessVenv = path.join(checkout, 'venv-readiness');
    copyInstallSurface(checkout);

    try {
      writeForwardingReadiness(checkout, {
        upstreamCheckout: readinessCheckout,
        upstreamConfig: readinessConfig,
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'readiness.pt'),
        pythonExe: path.join(readinessVenv, 'bin', 'python'),
        venvDir: readinessVenv,
      });

      const readinessResult = runRunnerJobProbe(checkout);
      expect(readinessResult.status).toBe(0);
      const readinessJob = JSON.parse(readinessResult.stdout);
      expect(readinessJob.upstream_checkout_path).toBe(readinessCheckout);
      expect(readinessJob.upstream_config_path).toBe(readinessConfig);
      expect(readinessJob.upstream_checkout_path).not.toBe(path.join(checkout, 'runtime', 'vendor', 'UltraShape-1.0'));

      const envResult = runRunnerJobProbe(checkout, {
        ULTRASHAPE_RUNTIME_MODE: 'real',
        ULTRASHAPE_UPSTREAM_CHECKOUT: envCheckout,
        ULTRASHAPE_UPSTREAM_CONFIG: envConfig,
      });
      expect(envResult.status).toBe(0);
      expect(JSON.parse(envResult.stdout)).toMatchObject({
        runtime_mode: 'real',
        upstream_checkout_path: envCheckout,
        upstream_config_path: envConfig,
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('applies override precedence for runtime mode, upstream paths, and SDPA policy fields without overriding explicit env', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-policy-precedence-'));
    const checkout = path.join(sandbox, 'repo');
    const readinessCheckout = path.join(sandbox, 'readiness-upstream');
    const readinessConfig = path.join(readinessCheckout, 'configs', 'infer_dit_refine.yaml');
    const envCheckout = path.join(sandbox, 'env-upstream');
    const envConfig = path.join(envCheckout, 'configs', 'env.yaml');
    const readinessVenv = path.join(checkout, 'venv-readiness');
    copyInstallSurface(checkout);

    try {
      writeForwardingReadiness(checkout, {
        upstreamCheckout: readinessCheckout,
        upstreamConfig: readinessConfig,
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'readiness.pt'),
        pythonExe: path.join(readinessVenv, 'bin', 'python'),
        venvDir: readinessVenv,
        flashAttnPolicy: {
          status: 'sdpa_real_allowed',
          required: false,
          available: false,
          degraded: true,
          degradation_reason: 'import:flash_attn',
          sdpa_allowed: true,
        },
      });

      const readinessResult = runRunnerJobProbe(checkout);
      expect(readinessResult.status).toBe(0);
      expect(JSON.parse(readinessResult.stdout)).toMatchObject({
        runtime_mode: 'auto',
        upstream_checkout_path: readinessCheckout,
        upstream_config_path: readinessConfig,
        attention_backend: 'sdpa',
        flash_attn_policy: {
          status: 'sdpa_real_allowed',
          sdpa_allowed: true,
          degraded: true,
        },
      });

      const envResult = runRunnerJobProbe(checkout, {
        ULTRASHAPE_RUNTIME_MODE: 'real',
        ULTRASHAPE_UPSTREAM_CHECKOUT: envCheckout,
        ULTRASHAPE_UPSTREAM_CONFIG: envConfig,
        ULTRASHAPE_ATTENTION_BACKEND: 'flash_attn',
        ULTRASHAPE_FLASH_ATTN_POLICY: 'required',
      });
      expect(envResult.status).toBe(0);
      expect(JSON.parse(envResult.stdout)).toMatchObject({
        runtime_mode: 'real',
        upstream_checkout_path: envCheckout,
        upstream_config_path: envConfig,
        attention_backend: 'flash_attn',
        flash_attn_policy: {
          status: 'required',
          required: true,
          sdpa_allowed: false,
        },
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('uses setup-recorded venv Python and forwards legacy runner environment variables', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-runner-env-'));
    const checkout = path.join(sandbox, 'repo');
    const upstreamCheckout = path.join(sandbox, 'upstream-env');
    const upstreamConfig = path.join(upstreamCheckout, 'configs', 'infer_dit_refine.yaml');
    const venvDir = path.join(checkout, 'venv');
    const pythonExe = path.join(venvDir, 'bin', 'python');
    copyInstallSurface(checkout);
    mkdirSync(path.dirname(pythonExe), { recursive: true });
    writeFileSync(pythonExe, '', 'utf8');

    try {
      writeForwardingReadiness(checkout, {
        upstreamCheckout,
        upstreamConfig,
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'readiness.pt'),
        pythonExe,
        venvDir,
      });
      const script = [
        'import json',
        'from pathlib import Path',
        'import generator as generator_module',
        'from generator import UltraShapeGenerator',
        'class Completed:',
        '    returncode = 0',
        '    stderr = ""',
        '    stdout = json.dumps({"ok": True, "result": {"file_path": "out.glb", "format": "glb", "backend": "local", "metrics": {}, "fallbacks": [], "subtrees_loaded": []}})',
        'captured = {}',
        'def fake_run(command, **kwargs):',
        '    captured["command"] = command',
        '    captured["env"] = kwargs.get("env")',
        '    return Completed()',
        'generator_module.subprocess.run = fake_run',
        'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
        'readiness = generator._require_runtime_ready()',
        'job = generator._build_runner_job(readiness=readiness, reference_image=Path.cwd() / "reference.png", coarse_mesh=Path.cwd() / "coarse.glb", output_dir=Path.cwd() / "outputs", params={"steps": 7, "guidance_scale": 2.5, "seed": 123, "preserve_scale": True})',
        'generator._run_local_runner(job)',
        'print(json.dumps({"command": captured["command"], "env": {key: captured["env"].get(key) for key in ["ULTRASHAPE_RUNTIME_MODE", "ULTRASHAPE_UPSTREAM_CHECKOUT", "ULTRASHAPE_UPSTREAM_CONFIG", "ULTRASHAPE_ATTENTION_BACKEND", "ULTRASHAPE_FLASH_ATTN_POLICY"]}}))',
      ].join('\n');

      const result = spawnSync('python3', ['-S', '-c', script], { cwd: checkout, encoding: 'utf8', env: process.env });

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual({
        command: [pythonExe, '-m', 'ultrashape_runtime.local_runner'],
        env: {
          ULTRASHAPE_RUNTIME_MODE: 'auto',
          ULTRASHAPE_UPSTREAM_CHECKOUT: upstreamCheckout,
          ULTRASHAPE_UPSTREAM_CONFIG: upstreamConfig,
          ULTRASHAPE_ATTENTION_BACKEND: null,
          ULTRASHAPE_FLASH_ATTN_POLICY: null,
        },
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('exposes the exact public generate signature required by the model shell contract', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-signature-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const signatureProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import inspect, json',
            'from generator import UltraShapeGenerator',
            'print(json.dumps(str(inspect.signature(UltraShapeGenerator.generate))))',
          ].join('\n'),
        ],
        { cwd: checkout, encoding: 'utf8' },
      );

      expect(signatureProbe.status).toBe(0);
      expect(JSON.parse(signatureProbe.stdout.trim())).toBe(
        '(self, image_bytes: \'bytes | None\', params: \'dict[str, Any] | None\' = None, progress_cb: \'Any | None\' = None, cancel_event: \'Any | None\' = None) -> \'str\'',
      );
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('reports download truthfully before and after staged setup assets exist', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-download-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);

    try {
      const beforeSetup = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }]);
      expect(beforeSetup.status).toBe(0);
      expect(JSON.parse(beforeSetup.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: false, loaded: false },
      ]);

      writeRuntimeStubModules(stubRoot);
      stageCheckpoint(checkout);
      const setupResult = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
      expect(setupResult.status).toBe(0);

      const afterSetup = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }], { PYTHONPATH: stubRoot });
      expect(afterSetup.status).toBe(0);
      expect(JSON.parse(afterSetup.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: true, loaded: false },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('copies only the model shell authority into staged checkouts', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-surface-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const readme = readFileSync(path.join(checkout, 'README.md'), 'utf8');

      expect(existsSync(path.join(checkout, 'generator.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'setup.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'processor.py'))).toBe(false);
      expect(existsSync(path.join(checkout, 'processor.js'))).toBe(false);
      expect(readme).toContain('The model shell is the sole public authority: `manifest.json`, `setup.py`, and `generator.py`.');
      expect(readme).not.toContain('process-shell authority');
      expect(readme).not.toContain('processor.py');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('lazy-loads on generate, returns a mesh path, and unload resets the loaded state', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-lifecycle-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: {
              mesh_path: coarseMesh,
              remesh: true,
              enable_texture: true,
              texture_resolution: 1024,
            },
          },
          { method: 'unload' },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      const payload = JSON.parse(result.stdout);
      expect(payload[0]).toMatchObject({ method: 'generate', ok: true, loaded: true });
      expect(typeof payload[0].result).toBe('string');
      expect(payload[0].result).toMatch(/\.glb$/);
      expect(existsSync(payload[0].result)).toBe(true);
      expect(payload[0].result).toContain(path.join(checkout, 'outputs'));
      expect(payload[0].debug.last_result).toMatchObject({
        backend: 'local',
        subtrees_loaded: ['vae', 'dit', 'conditioner'],
      });
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('coarse_mesh');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('reference_image');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('remesh');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('enable_texture');
      expect(JSON.stringify(payload[0].debug.last_job ?? {})).not.toContain('texture_resolution');
      expect(payload[1]).toEqual({ method: 'unload', ok: true, result: false, loaded: false });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('resolves the reported Workflows-prefixed relative mesh_path against the real existing file', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-relative-mesh-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);

    try {
      const relativeMeshPath = path.join('Workflows', 'foo.glb');
      const existingMeshPath = path.join(checkout, 'Workflows', 'foo.glb');
      mkdirSync(path.dirname(existingMeshPath), { recursive: true });
      writeFileSync(existingMeshPath, 'mesh', 'utf8');
      const resolutionProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import json, sys',
            'from pathlib import Path',
            'from generator import UltraShapeGenerator',
            'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "Workflows")',
            'relative = str(generator._resolve_mesh_path(sys.argv[1]))',
            'print(json.dumps({"relative": relative}))',
          ].join('\n'),
          relativeMeshPath,
        ],
        {
          cwd: checkout,
          encoding: 'utf8',
          env: {
            ...process.env,
            PYTHONPATH: stubRoot,
          },
        },
      );

      expect(resolutionProbe.status).toBe(0);
      expect(JSON.parse(resolutionProbe.stdout)).toEqual({ relative: existingMeshPath });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('surfaces structured mesh resolution diagnostics when no candidate exists', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-mesh-diagnostics-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const workspaceDir = path.join(sandbox, 'workspace');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    mkdirSync(workspaceDir, { recursive: true });

    try {
      const diagnosticProbe = spawnSync(
        'python3',
        [
          '-S',
          '-c',
          [
            'import json, sys',
            'from pathlib import Path',
            'from generator import PublicRuntimeError, UltraShapeGenerator',
            'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
            'try:',
            '    generator._resolve_mesh_path(sys.argv[1])',
            'except PublicRuntimeError as error:',
            '    print(json.dumps({"code": error.code, "message": str(error)}))',
          ].join('\n'),
          path.join('Workflows', 'missing.glb'),
        ],
        {
          cwd: checkout,
          encoding: 'utf8',
          env: {
            ...process.env,
            PYTHONPATH: stubRoot,
            WORKSPACE_DIR: workspaceDir,
          },
        },
      );

      expect(diagnosticProbe.status).toBe(0);
      expect(JSON.parse(diagnosticProbe.stdout)).toEqual({
        code: 'INVALID_INPUT',
        message: [
          'Mesh input could not be resolved.',
          'original mesh_path: Workflows/missing.glb',
          `self.outputs_dir: ${path.join(checkout, 'outputs')}`,
          `WORKSPACE_DIR: ${workspaceDir}`,
          'candidates:',
          `  1. path=${path.join(checkout, 'outputs', 'Workflows', 'missing.glb')} exists=False parent=${path.join(checkout, 'outputs', 'Workflows')} parent_exists=False`,
          `  2. path=${path.join(checkout, 'Workflows', 'missing.glb')} exists=False parent=${path.join(checkout, 'Workflows')} parent_exists=False`,
          `  3. path=${path.join(workspaceDir, 'Workflows', 'missing.glb')} exists=False parent=${path.join(workspaceDir, 'Workflows')} parent_exists=False`,
        ].join('\n'),
      });
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('fails with INVALID_INPUT when image_bytes are not a decodable runtime image payload', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-bad-image-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: Buffer.from('not-a-png', 'utf8').toString('base64'),
            params: { mesh_path: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('reference_image'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('fails when params.mesh_path is not a readable glb because the vendored runtime actually consumes it', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-bad-mesh-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    const invalidMesh = path.join(sandbox, 'invalid.glb');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    writeFileSync(invalidMesh, 'not-a-glb', 'utf8');

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: invalidMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('coarse_mesh'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('rejects legacy alias fields instead of silently accepting private runner payload keys', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-legacy-alias-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { coarse_mesh: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('legacy alias fields'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('rejects mixed public and private payload fields instead of tolerating ambiguous input', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-mixed-payload-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: { mesh_path: coarseMesh, coarse_mesh: coarseMesh },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: expect.stringContaining('Mixed public contract fields'),
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('ignores Modly global generation params but still rejects other unsupported params', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-global-params-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);
    runSetup(checkout, checkout, { PYTHONPATH: stubRoot });
    const { coarseMesh } = createRuntimeInputs(sandbox);

    try {
      const result = runGeneratorProbe(
        checkout,
        [
          {
            method: 'generate',
            imageBase64: PNG_1X1_BASE64,
            params: {
              mesh_path: coarseMesh,
              remesh: false,
              enable_texture: false,
              texture_resolution: 512,
              unsupported_flag: true,
            },
          },
        ],
        { PYTHONPATH: stubRoot },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'INVALID_INPUT',
            message: 'Unsupported params fields: unsupported_flag.',
          },
          loaded: true,
        },
      ]);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('surfaces public dependency and weight failures without reporting impossible runtime success', () => {
    const dependencySandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-deps-'));
    const dependencyCheckout = path.join(dependencySandbox, 'repo');
    copyInstallSurface(dependencyCheckout);

    const weightsSandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-generator-weights-'));
    const weightsCheckout = path.join(weightsSandbox, 'repo');
    const stubRoot = path.join(weightsSandbox, 'stubs');
    copyInstallSurface(weightsCheckout);
    writeRuntimeStubModules(stubRoot);

    try {
      const dependencySetup = runSetup(dependencyCheckout, dependencyCheckout, {
        ULTRASHAPE_SETUP_TEST_CUBVH_PREREQ_MISSING: 'compiler',
      });
      expect(dependencySetup.status).toBe(1);
      const missingDependencies = runGeneratorProbe(dependencyCheckout, [{ method: 'load' }]);
      expect(missingDependencies.status).toBe(0);
      expect(JSON.parse(missingDependencies.stdout)).toEqual([
        {
          method: 'load',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'DEPENDENCY_MISSING',
            message: expect.stringContaining('Required runtime imports are unavailable'),
          },
          loaded: false,
        },
      ]);

      const setupResult = runSetup(weightsCheckout, weightsCheckout, {
        PYTHONPATH: stubRoot,
        ULTRASHAPE_SETUP_TEST_HF_SCENARIO: 'not-found',
      });
      expect(setupResult.status).toBe(1);
      const missingWeights = runGeneratorProbe(weightsCheckout, [{ method: 'generate', imageBase64: PNG_1X1_BASE64, params: {} }], {
        PYTHONPATH: stubRoot,
      });
      expect(missingWeights.status).toBe(0);
      expect(JSON.parse(missingWeights.stdout)).toEqual([
        {
          method: 'generate',
          ok: false,
          error: {
            type: 'PublicRuntimeError',
            code: 'WEIGHTS_MISSING',
            message: expect.stringContaining('Required runtime weights are unavailable'),
          },
          loaded: false,
        },
      ]);
    } finally {
      rmSync(dependencySandbox, { recursive: true, force: true });
      rmSync(weightsSandbox, { recursive: true, force: true });
    }
  });
});
