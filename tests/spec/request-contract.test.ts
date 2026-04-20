import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const processorPath = path.join(repoRoot, 'processor.py');

function runProcessor(args: string[], payload: unknown) {
  return spawnSync('python3', [processorPath, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    input: JSON.stringify(payload),
  });
}

describe('processor request contract', () => {
  it('accepts the preferred top-level shell truth directly', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/reference.png',
      coarse_mesh: 'inputs/coarse.glb',
    });

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: true,
      normalized_request: {
        reference_image: 'inputs/reference.png',
        coarse_mesh: 'inputs/coarse.glb',
      },
    });
  });

  it('rejects requests missing the coarse mesh truthfully', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/reference.png',
    });

    expect(result.status).toBe(1);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: false,
      error: {
        code: 'INVALID_INPUT',
        message: 'reference_image and coarse_mesh are both required for process-refiner requests.',
      },
    });
  });

  it('allows the temporary fallback seam only inside processor.py', () => {
    const result = runProcessor(['--validate-request'], {
      input: { filePath: 'inputs/reference.png' },
      params: { coarse_mesh: 'inputs/coarse.glb' },
    });

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: true,
      normalized_request: {
        reference_image: 'inputs/reference.png',
        coarse_mesh: 'inputs/coarse.glb',
      },
    });
  });

  it('prefers top-level values over fallback aliases when both are present', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/preferred-reference.png',
      coarse_mesh: 'inputs/preferred-coarse.glb',
      input: { filePath: 'inputs/fallback-reference.png' },
      params: { coarse_mesh: 'inputs/fallback-coarse.glb' },
    });

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: true,
      normalized_request: {
        reference_image: 'inputs/preferred-reference.png',
        coarse_mesh: 'inputs/preferred-coarse.glb',
      },
    });
  });

  it('rejects partially-fallback requests instead of mixing shell truths', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/reference.png',
      params: { coarse_mesh: 'inputs/coarse.glb' },
    });

    expect(result.status).toBe(1);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: false,
      error: {
        code: 'INVALID_INPUT',
        message: 'reference_image and coarse_mesh are both required for process-refiner requests.',
      },
    });
  });

  it('rejects unsupported params keys', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/reference.png',
      coarse_mesh: 'inputs/coarse.glb',
      params: {
        backend: 'local',
        unexpected: true,
      },
    });

    expect(result.status).toBe(1);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: false,
      error: {
        code: 'INVALID_INPUT',
        message: 'Unsupported params for the stable shell: unexpected.',
      },
    });
  });

  it('rejects non-local backends', () => {
    const result = runProcessor(['--validate-request'], {
      reference_image: 'inputs/reference.png',
      coarse_mesh: 'inputs/coarse.glb',
      params: {
        backend: 'remote',
      },
    });

    expect(result.status).toBe(1);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: false,
      error: {
        code: 'INVALID_INPUT',
        message: 'Only local execution is allowed in this clean-room shell.',
      },
    });
  });
});
