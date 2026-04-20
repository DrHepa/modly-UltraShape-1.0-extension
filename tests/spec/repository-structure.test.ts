import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('clean-room repository structure', () => {
  it('keeps only the clean-room harness paths for now', () => {
    expect(existsSync(repoPath('README.md'))).toBe(true);
    expect(existsSync(repoPath('package.json'))).toBe(true);
    expect(existsSync(repoPath('tsconfig.json'))).toBe(true);
    expect(existsSync(repoPath('vitest.config.ts'))).toBe(true);
    expect(existsSync(repoPath('tests', 'spec'))).toBe(true);
    expect(existsSync(repoPath('runtime', 'configs'))).toBe(true);
    expect(existsSync(repoPath('runtime', 'vendor', 'ultrashape_runtime'))).toBe(true);
  });

  it('keeps deleted authority paths absent', () => {
    expect(existsSync(repoPath('src'))).toBe(false);
    expect(existsSync(repoPath('fixtures'))).toBe(false);
    expect(existsSync(repoPath('runtime', 'patches'))).toBe(false);
    expect(existsSync(repoPath('runtime', 'modly'))).toBe(false);
    expect(existsSync(repoPath('processor.js'))).toBe(false);
    expect(existsSync(repoPath('tests', 'spec', 'fallback-fixtures.test.ts'))).toBe(false);
  });

  it('documents shell authority and current non-goals in the README', () => {
    const readme = readFileSync(repoPath('README.md'), 'utf8');

    expect(readme).toContain('Shell authority in this rewrite is limited to');
    expect(readme).toContain(
      'Fallback names are a temporary compatibility seam inside `processor.py` and are not shell authority anywhere else.',
    );
    expect(readme).toContain('Batch 1 non-goals');
    expect(readme).toContain('Do not recreate `src/`');
    expect(readme).toContain('Do not restore fallback fixture bundles');
    expect(readme).toContain('Do not restore patch-authority directories');
    expect(readme).not.toContain('required_inputs: input.filePath');
    expect(readme).not.toContain('required_inputs: params.coarse_mesh');
  });
});
