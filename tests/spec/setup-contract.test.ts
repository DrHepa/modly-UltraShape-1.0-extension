import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const setupPath = resolve(repoRoot, 'setup.py');

function runSetup(argument: Record<string, unknown>) {
  return spawnSync('python3', [setupPath, JSON.stringify(argument)], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
}

describe('UltraShape setup.py contract', () => {
  it('creates ext_dir/venv from Modly JSON args and preserves the venv on rerun', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-'));
    const installDir = join(root, 'extension-root');
    const firstRun = runSetup({
      python_exe: 'python3',
      ext_dir: installDir,
        gpu_sm: 90,
        cuda_version: 12.8,
    });

    try {
      expect(firstRun.status).toBe(0);
      expect(existsSync(join(installDir, 'venv'))).toBe(true);
      expect(readFileSync(join(installDir, 'venv', 'pyvenv.cfg'), 'utf8')).toContain('home');

      const sentinel = join(installDir, 'venv', 'sentinel.txt');
      writeFileSync(sentinel, 'keep-me');

      const secondRun = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 90,
      });

      expect(secondRun.status).toBe(0);
      expect(existsSync(sentinel)).toBe(true);
      expect(readFileSync(sentinel, 'utf8')).toBe('keep-me');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails fast when ext_dir is missing from the Modly setup payload', () => {
    const outcome = runSetup({
      python_exe: 'python3',
      gpu_sm: 90,
    });

    expect(outcome.status).not.toBe(0);
    expect(`${outcome.stderr}${outcome.stdout}`).toContain('ext_dir');
  });

  it('accepts the numeric gpu_sm/cuda_version values that Modly sends at install time', () => {
    const root = mkdtempSync(join(tmpdir(), 'ultrashape-setup-numeric-'));
    const installDir = join(root, 'extension-root');

    try {
      const outcome = runSetup({
        python_exe: 'python3',
        ext_dir: installDir,
        gpu_sm: 86,
        cuda_version: 128,
      });

      expect(outcome.status).toBe(0);
      const summary = JSON.parse(readFileSync(join(installDir, '.setup-summary.json'), 'utf8')) as {
        gpu_sm: string;
        cuda_version: string;
      };
      expect(summary.gpu_sm).toBe('86');
      expect(summary.cuda_version).toBe('128');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
