import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function repoPath(...segments: string[]) {
  return path.join(repoRoot, ...segments);
}

describe('runtime closure repository structure', () => {
  it('keeps only the runtime closure harness paths', () => {
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
    expect(existsSync(repoPath('processor.py'))).toBe(false);
    expect(existsSync(repoPath('processor.js'))).toBe(false);
    expect(existsSync(repoPath('tests', 'spec', 'fallback-fixtures.test.ts'))).toBe(false);
  });

  it('keeps shell-level manifest authority free of processor-era leakage', () => {
    const manifestText = readFileSync(repoPath('manifest.json'), 'utf8');

    expect(manifestText).not.toContain('processor');
    expect(manifestText).not.toContain('process-refiner');
    expect(manifestText).not.toContain('reference_image');
    expect(manifestText).not.toContain('coarse_mesh');
  });

  it('documents shell authority and the explicit dual-mode runtime without legacy cleanup language', () => {
    const readme = readFileSync(repoPath('README.md'), 'utf8');
    const runtimeRoot = readFileSync(repoPath('runtime', '__init__.py'), 'utf8');
    const vendoredRuntime = readFileSync(repoPath('runtime', 'vendor', 'ultrashape_runtime', '__init__.py'), 'utf8');

    expect(readme).toContain('The model shell is the sole public authority: `manifest.json`, `setup.py`, and `generator.py`.');
    expect(readme).toContain('`runtime/**` and `models/ultrashape/**` remain private implementation details.');
    expect(readme).toContain('The private runtime is an explicit dual-mode UltraShape seam: real mode targets the closest achievable upstream closure when the exact environment is available, and portable mode is the reduced-environment fallback.');
    expect(readme).toContain('Configure authoritative real mode explicitly with `ultrashape_checkout_path` or `ULTRASHAPE_UPSTREAM_CHECKOUT` pointing at a validated `PKU-YuanGroup/UltraShape-1.0` checkout.');
    expect(readme).toContain('Portable mode remains non-authoritative: it is only a reduced-environment fallback when real mode is unavailable or when `ULTRASHAPE_RUNTIME_MODE=portable` is forced.');
    expect(readme).toContain('Future work remains to durably vendor the pinned upstream inference graph with the required upstream license and notice files; this repository currently records checkout revision diagnostics but does not claim a pinned vendored graph.');
    expect(readme).not.toContain('process-shell authority');
    expect(readme).not.toContain('temporary migration seam');
    expect(readme).not.toContain('temporary compatibility seam');
    expect(readme).not.toContain('later removal batch');
    expect(readme).not.toContain('clean-room');
    expect(readme).not.toContain('synthetic success');
    expect(readme).not.toContain('fallback alias');
    expect(readme).not.toContain('Batch 1 non-goals');
    expect(readme).not.toContain('- `processor.py`');
    expect(readme).not.toContain('reference_image');
    expect(readme).not.toContain('coarse_mesh');
    expect(readme).not.toContain('required_inputs: input.filePath');
    expect(readme).not.toContain('required_inputs: params.coarse_mesh');
    expect(runtimeRoot).toContain('Runtime package root for the UltraShape dual-mode private closure.');
    expect(runtimeRoot).not.toContain('clean-room');
    expect(vendoredRuntime).toContain('UltraShape runtime package markers for the explicit dual-mode closure.');
    expect(vendoredRuntime).toContain("RUNTIME_MODE_STRATEGY = 'explicit-dual-mode'");
    expect(vendoredRuntime).toContain("UPSTREAM_CLOSURE_READY = False");
    expect(vendoredRuntime).not.toContain('clean-room');
  });
});
