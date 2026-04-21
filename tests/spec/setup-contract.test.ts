import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { copyInstallSurface, repoRoot, runGeneratorProbe, stageCheckpoint, writeRuntimeStubModules } from './install-test-helpers.js';

type Readiness = {
  backend: string;
  checkpoint: string;
  config_path: string;
  ext_dir: string;
  missing_required: string[];
  required_imports_ok: boolean;
  status: string;
  weights_ready: boolean;
};

function runSetup(cwd: string, extDir: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['-S', 'setup.py', '--ext-dir', extDir], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

function readReadiness(extDir: string) {
  return JSON.parse(readFileSync(path.join(extDir, '.runtime-readiness.json'), 'utf8')) as Readiness;
}

function readSetupSummary(extDir: string) {
  return readFileSync(path.join(extDir, '.setup-summary.json'), 'utf8');
}

describe('setup.py install truth', () => {
  it('blocks readiness when required imports or the checkpoint are missing', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-contract-'));
    const checkout = path.join(sandbox, 'repo');
    copyInstallSurface(checkout);

    try {
      const result = runSetup(checkout, checkout);

      expect(result.status).toBe(1);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        backend: 'local',
        ext_dir: checkout,
        config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        required_imports_ok: false,
        weights_ready: false,
        status: 'blocked',
      });
      const generator = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }, { method: 'load' }]);
      expect(generator.status).toBe(0);
      expect(JSON.parse(generator.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: false, loaded: false },
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
      expect(readiness.missing_required).toContain('import:torch');
      expect(readiness.missing_required).toContain('weight:models/ultrashape/ultrashape_v1.pt');
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).not.toContain('fallback');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).not.toContain('fallback');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).not.toContain('fallback');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('reports ready only when staged config, required imports, and weights are all present', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-setup-ready-'));
    const checkout = path.join(sandbox, 'repo');
    const stubRoot = path.join(sandbox, 'stubs');
    copyInstallSurface(checkout);
    writeRuntimeStubModules(stubRoot);
    stageCheckpoint(checkout);

    try {
      const result = runSetup(checkout, checkout, { PYTHONPATH: stubRoot });

      expect(result.status).toBe(0);

      const readiness = readReadiness(checkout);
      expect(readiness).toMatchObject({
        backend: 'local',
        ext_dir: checkout,
        config_path: path.join(checkout, 'runtime', 'configs', 'infer_dit_refine.yaml'),
        checkpoint: path.join(checkout, 'models', 'ultrashape', 'ultrashape_v1.pt'),
        required_imports_ok: true,
        weights_ready: true,
        status: 'ready',
        missing_required: [],
      });
      const generator = runGeneratorProbe(checkout, [{ method: 'is_downloaded' }], { PYTHONPATH: stubRoot });
      expect(generator.status).toBe(0);
      expect(JSON.parse(generator.stdout)).toEqual([
        { method: 'is_downloaded', ok: true, result: true, loaded: false },
      ]);
      expect(existsSync(path.join(checkout, 'runtime', 'vendor', 'ultrashape_runtime', 'local_runner.py'))).toBe(true);
      expect(existsSync(path.join(checkout, 'generator.py'))).toBe(true);
      expect(result.stdout).not.toContain('filePath');
      expect(result.stdout).not.toContain('params.coarse_mesh');
      expect(result.stdout).not.toContain('fallback');
      expect(JSON.stringify(readiness)).not.toContain('filePath');
      expect(JSON.stringify(readiness)).not.toContain('params.coarse_mesh');
      expect(JSON.stringify(readiness)).not.toContain('fallback');
      expect(readSetupSummary(checkout)).not.toContain('filePath');
      expect(readSetupSummary(checkout)).not.toContain('params.coarse_mesh');
      expect(readSetupSummary(checkout)).not.toContain('fallback');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });
});
