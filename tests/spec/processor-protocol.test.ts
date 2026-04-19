import { chmodSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { deflateSync } from 'node:zlib';

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

function torchCheckpointStubSource() {
  return [
    '__version__ = "0.0-test"',
    'import json',
    'import zipfile',
    '',
    'int32 = "int32"',
    'float32 = "float32"',
    '',
    'class Tensor:',
    '    def __init__(self, values, dtype=None):',
    '        self._values = values',
    '        self.dtype = dtype',
    '',
    '    def int(self):',
    '        return Tensor([[int(value) for value in row] for row in self._values], int32)',
    '',
    '    def float(self):',
    '        return Tensor([[float(value) for value in row] for row in self._values], float32)',
    '',
    '    def __iter__(self):',
    '        return iter(self._values)',
    '',
    '    def __len__(self):',
    '        return len(self._values)',
    '',
    '    def __getitem__(self, index):',
    '        return self._values[index]',
    '',
    'def tensor(values, dtype=None):',
    '    normalized = [list(row) for row in values]',
    '    return Tensor(normalized, dtype)',
    '',
    'def load(path, map_location=None, weights_only=False):',
    '    with zipfile.ZipFile(path, "r") as archive:',
    '        return json.loads(archive.read("checkpoint.json").decode("utf8"))',
    '',
  ].join('\n');
}

function diffusersStubSource() {
  return [
    '__version__ = "0.0-test"',
    'class FlowMatchEulerDiscreteScheduler:',
    '    def __init__(self, **config):',
    '        self.config = dict(config)',
    '        self.timesteps = []',
    '',
    '    @classmethod',
    '    def from_config(cls, config):',
    '        payload = dict(config) if isinstance(config, dict) else {"value": config}',
    '        return cls(**payload)',
    '',
    '    def set_timesteps(self, step_count):',
    '        self.timesteps = list(range(int(step_count)))',
    '',
  ].join('\n');
}

function cubvhStubSource() {
  return [
    '__version__ = "0.0-test"',
    'def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):',
    '    vertices = []',
    '    for point, cube in zip(coords, corners):',
    '        offset = sum(float(value) for value in cube) / max(len(cube), 1)',
    '        vertices.append((',
    '            float(point[0]) + offset,',
    '            float(point[1]) - (offset / 2.0),',
    '            float(point[2]) + (offset / 3.0),',
    '        ))',
    '    faces = []',
    '    for index in range(1, max(len(vertices) - 1, 1)):',
    '        if index + 1 >= len(vertices):',
    '            break',
    '        faces.append((0, index, index + 1))',
    '    return vertices, faces',
    '',
  ].join('\n');
}

function trimeshStubSource() {
  return [
    'import json',
    'import struct',
    '',
    'def _pad(payload, pad_byte=b" "):',
    '    padding = (-len(payload)) % 4',
    '    return payload + (pad_byte * padding)',
    '',
    'class Trimesh:',
    '    def __init__(self, vertices, faces, process=False):',
    '        self.vertices = [tuple(float(axis) for axis in vertex) for vertex in vertices]',
    '        self.faces = [tuple(int(index) for index in face) for face in faces]',
    '',
    'class Scene:',
    '    def __init__(self):',
    '        self._items = []',
    '',
    '    def add_geometry(self, mesh, node_name=None):',
    '        self._items.append((mesh, node_name or "mesh"))',
    '',
    '    def export(self, file_type="glb"):',
    '        if file_type != "glb":',
    '            raise ValueError("Stub trimesh exporter supports only glb")',
    '        if not self._items:',
    '            raise ValueError("Scene has no geometry")',
    '        mesh, node_name = self._items[0]',
    '        vertex_bytes = b"".join(struct.pack("<3f", *vertex) for vertex in mesh.vertices)',
    '        index_bytes = b"".join(struct.pack("<3I", *face) for face in mesh.faces)',
    '        binary_blob = _pad(vertex_bytes, b"\\x00") + _pad(index_bytes, b"\\x00")',
    '        json_doc = {',
    '            "asset": {"version": "2.0", "generator": "trimesh-test-stub"},',
    '            "scene": 0,',
    '            "scenes": [{"nodes": [0]}],',
    '            "nodes": [{"mesh": 0, "name": node_name}],',
    '            "meshes": [{"name": node_name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],',
    '            "buffers": [{"byteLength": len(binary_blob)}],',
    '            "bufferViews": [',
    '                {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes), "target": 34962},',
    '                {"buffer": 0, "byteOffset": len(_pad(vertex_bytes, b"\\x00")), "byteLength": len(index_bytes), "target": 34963},',
    '            ],',
    '            "accessors": [',
    '                {',
    '                    "bufferView": 0,',
    '                    "componentType": 5126,',
    '                    "count": len(mesh.vertices),',
    '                    "type": "VEC3",',
    '                    "min": [min(vertex[index] for vertex in mesh.vertices) for index in range(3)],',
    '                    "max": [max(vertex[index] for vertex in mesh.vertices) for index in range(3)],',
    '                },',
    '                {"bufferView": 1, "componentType": 5125, "count": len(mesh.faces) * 3, "type": "SCALAR"},',
    '            ],',
    '        }',
    '        json_bytes = _pad(json.dumps(json_doc, separators=(",", ":")).encode("utf8"))',
    '        total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_blob)',
    '        return b"".join([',
    '            b"glTF",',
    '            struct.pack("<I", 2),',
    '            struct.pack("<I", total_length),',
    '            struct.pack("<I", len(json_bytes)),',
    '            struct.pack("<I", 0x4E4F534A),',
    '            json_bytes,',
    '            struct.pack("<I", len(binary_blob)),',
    '            struct.pack("<I", 0x004E4942),',
    '            binary_blob,',
    '        ])',
    '',
  ].join('\n');
}

function writeBinaryCheckpointBundle(path: string, payload: Record<string, unknown>) {
  const outcome = spawnSync(
    'python3',
    [
      '-c',
      [
        'import sys, json, zipfile',
        'path = sys.argv[1]',
        'payload = sys.argv[2]',
        'with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:',
        '    archive.writestr("checkpoint.json", payload)',
      ].join('\n'),
      path,
      JSON.stringify(payload),
    ],
    { encoding: 'utf8' },
  );

  if (outcome.status !== 0) {
    throw new Error(outcome.stderr || 'Failed to write binary checkpoint bundle.');
  }
}

function createBinaryGlbBytes() {
  const vertices: Array<[number, number, number]> = [
    [-0.9, -0.7, -0.5],
    [0.8, -0.6, -0.4],
    [0.9, 0.7, -0.3],
    [-0.8, 0.8, -0.2],
    [-0.6, -0.5, 0.7],
    [0.7, -0.4, 0.9],
    [0.6, 0.5, 0.8],
    [-0.7, 0.6, 0.6],
  ];
  const faces: Array<[number, number, number]> = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 5],
    [4, 7, 6],
    [0, 4, 5],
    [0, 5, 1],
    [1, 5, 6],
    [1, 6, 2],
    [2, 6, 7],
    [2, 7, 3],
    [3, 7, 4],
    [3, 4, 0],
  ];
  const vertexBytes = Buffer.alloc(vertices.length * 12);
  vertices.forEach((vertex, index) => {
    vertexBytes.writeFloatLE(vertex[0], index * 12);
    vertexBytes.writeFloatLE(vertex[1], (index * 12) + 4);
    vertexBytes.writeFloatLE(vertex[2], (index * 12) + 8);
  });

  const indexBytes = Buffer.alloc(faces.length * 12);
  faces.forEach((face, index) => {
    indexBytes.writeUInt32LE(face[0], index * 12);
    indexBytes.writeUInt32LE(face[1], (index * 12) + 4);
    indexBytes.writeUInt32LE(face[2], (index * 12) + 8);
  });

  const paddedVertexBytes = Buffer.concat([vertexBytes, Buffer.alloc((4 - (vertexBytes.length % 4)) % 4)]);
  const paddedIndexBytes = Buffer.concat([indexBytes, Buffer.alloc((4 - (indexBytes.length % 4)) % 4)]);
  const binaryChunk = Buffer.concat([paddedVertexBytes, paddedIndexBytes]);
  const jsonChunk = Buffer.from(
    JSON.stringify({
      asset: { version: '2.0', generator: 'processor-protocol-test-fixture' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0, name: 'coarse-mesh' }],
      meshes: [{ name: 'coarse-mesh', primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      buffers: [{ byteLength: binaryChunk.length }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: vertexBytes.length, target: 34962 },
        { buffer: 0, byteOffset: paddedVertexBytes.length, byteLength: indexBytes.length, target: 34963 },
      ],
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: vertices.length,
          type: 'VEC3',
          min: [-0.9, -0.7, -0.5],
          max: [0.9, 0.8, 0.9],
        },
        { bufferView: 1, componentType: 5125, count: faces.length * 3, type: 'SCALAR' },
      ],
    }),
    'utf8',
  );
  const paddedJsonChunk = Buffer.concat([jsonChunk, Buffer.alloc((4 - (jsonChunk.length % 4)) % 4, 0x20)]);
  const totalLength = 12 + 8 + paddedJsonChunk.length + 8 + binaryChunk.length;

  return Buffer.concat([
    Buffer.from('glTF', 'ascii'),
    Buffer.from(Uint32Array.of(2, totalLength).buffer),
    Buffer.from(Uint32Array.of(paddedJsonChunk.length, 0x4e4f534a).buffer),
    paddedJsonChunk,
    Buffer.from(Uint32Array.of(binaryChunk.length, 0x004e4942).buffer),
    binaryChunk,
  ]);
}

function makeCrc32Table() {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) === 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
}

const crc32Table = makeCrc32Table();

function crc32(buffer: Buffer) {
  let value = 0xffffffff;
  for (const byte of buffer) {
    value = crc32Table[(value ^ byte) & 0xff] ^ (value >>> 8);
  }
  return (value ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Buffer) {
  const typeBuffer = Buffer.from(type, 'ascii');
  const lengthBuffer = Buffer.alloc(4);
  lengthBuffer.writeUInt32BE(data.length, 0);
  const checksumBuffer = Buffer.alloc(4);
  checksumBuffer.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([lengthBuffer, typeBuffer, data, checksumBuffer]);
}

function createReferenceImageBytes() {
  const pixels = [
    [255, 32, 32, 255],
    [32, 255, 32, 255],
    [32, 32, 255, 255],
    [240, 220, 48, 160],
  ];
  const width = 2;
  const height = 2;
  const scanlines: Buffer[] = [];

  for (let rowIndex = 0; rowIndex < height; rowIndex += 1) {
    const row = Buffer.alloc(1 + width * 4);
    row[0] = 0;
    for (let columnIndex = 0; columnIndex < width; columnIndex += 1) {
      const pixel = pixels[(rowIndex * width) + columnIndex] ?? [0, 0, 0, 255];
      const offset = 1 + (columnIndex * 4);
      row[offset + 0] = pixel[0] ?? 0;
      row[offset + 1] = pixel[1] ?? 0;
      row[offset + 2] = pixel[2] ?? 0;
      row[offset + 3] = pixel[3] ?? 255;
    }
    scanlines.push(row);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(Buffer.concat(scanlines))),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function expectRenderableGlbMesh(path: string) {
  const payload = readFileSync(path);
  expect(payload.subarray(0, 4).toString('ascii')).toBe('glTF');
  const jsonLength = payload.readUInt32LE(12);
  const jsonChunkType = payload.readUInt32LE(16);
  expect(jsonChunkType).toBe(0x4e4f534a);
  const document = JSON.parse(payload.subarray(20, 20 + jsonLength).toString('utf8').trim().replace(/\u0000+$/u, '')) as {
    asset?: { version?: string };
    scenes?: unknown[];
    nodes?: unknown[];
    meshes?: Array<{ primitives?: Array<{ attributes?: { POSITION?: unknown }; indices?: unknown }> }>;
    accessors?: unknown[];
    bufferViews?: unknown[];
  };
  expect(document.asset?.version).toBe('2.0');
  expect(document.scenes?.length ?? 0).toBeGreaterThan(0);
  expect(document.nodes?.length ?? 0).toBeGreaterThan(0);
  expect(document.meshes?.length ?? 0).toBeGreaterThan(0);
  expect(document.bufferViews?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(document.accessors?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(document.meshes?.[0]?.primitives?.[0]).toEqual(
    expect.objectContaining({
      attributes: expect.objectContaining({
        POSITION: expect.any(Number),
      }),
      indices: expect.any(Number),
    }),
  );
}

function getRunnerMetrics(result: Record<string, unknown> | null): Record<string, unknown> {
  const envelope = result?.result;
  if (!envelope || typeof envelope !== 'object' || !('metrics' in envelope)) {
    throw new Error('Expected runner success metrics to be present.');
  }

  const metrics = envelope.metrics;
  if (!metrics || typeof metrics !== 'object') {
    throw new Error('Expected runner metrics to be an object.');
  }

  return metrics as Record<string, unknown>;
}

function createFixtureWorkspace() {
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-processor-'));
  const outputDir = join(root, 'output');
  const stubDir = join(root, 'py-stubs');
  mkdirSync(outputDir);
  mkdirSync(stubDir, { recursive: true });

  const referenceImage = join(root, 'reference.png');
  const namedCoarseMesh = join(root, 'named-coarse.glb');
  const fallbackCoarseMesh = join(root, 'fallback-coarse.obj');
  const packagedArtifact = join(root, 'artifact.obj');

  writeFileSync(referenceImage, createReferenceImageBytes());
  writeFileSync(namedCoarseMesh, createBinaryGlbBytes());
  writeFileSync(fallbackCoarseMesh, 'fallback-mesh');
  writeFileSync(packagedArtifact, 'refined-artifact');
  writeFileSync(join(stubDir, 'cubvh.py'), cubvhStubSource());
  writeFileSync(join(stubDir, 'torch.py'), torchCheckpointStubSource());
  writeFileSync(join(stubDir, 'trimesh.py'), trimeshStubSource());

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
      '    success_result = {"file_path": str(inside_output), "format": "glb", "backend": "local", "metrics": {"chamfer": 0.0042, "rms": 0.0125, "topology_changed": True, "extent_ratio": [1, 1, 1]}, "fallbacks": [], "subtrees_loaded": ["vae", "dit", "conditioner"], "warnings": []}',
      '    if mode == "invalid-json":',
      '        sys.stdout.write("not-json")',
      '        return 0',
      '    if mode == "outside-output":',
      '        outside_output.write_text("outside", encoding="utf8")',
      '        result = dict(success_result)',
      '        result["file_path"] = str(outside_output)',
      '        sys.stdout.write(json.dumps({"ok": True, "result": result}))',
      '        return 0',
      '    if mode == "missing-metadata":',
      '        incomplete = {"file_path": str(inside_output), "format": "glb", "backend": "local", "warnings": []}',
      '        inside_output.write_text("runner-output", encoding="utf8")',
      '        sys.stdout.write(json.dumps({"ok": True, "result": incomplete}))',
      '        return 0',
      '    if mode == "dependency-error":',
      '        sys.stdout.write(json.dumps({"ok": False, "error_code": "DEPENDENCY_MISSING", "error_message": "missing dependency"}))',
      '        return 1',
      '    if mode == "weights-error":',
      '        sys.stdout.write(json.dumps({"ok": False, "error_code": "WEIGHTS_MISSING", "error_message": "missing checkpoint"}))',
      '        return 1',
      '    if mode == "gate-error":',
      '        sys.stdout.write(json.dumps({"ok": False, "error_code": "LOCAL_RUNTIME_UNAVAILABLE", "error_message": "GEOMETRIC_GATE_REJECTED: aligned passthrough geometry does not satisfy the real-refinement gate"}))',
      '        return 1',
      '    inside_output.write_text("runner-output", encoding="utf8")',
      '    sys.stdout.write(json.dumps({"ok": True, "result": success_result}))',
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

function checkpointBundlePayload() {
  return {
    vae: {
      tensors: {
        latent_basis: [0.11, 0.33, 0.55, 0.77],
      },
    },
    dit: {
      tensors: {
        attention_bias: [0.21, 0.41, 0.61, 0.81],
      },
    },
    conditioner: {
      tensors: {
        mask_bias: [0.14, 0.24, 0.64, 0.74],
      },
    },
  };
}

function checkpointBundleWithout(...missingSubtrees: Array<'vae' | 'dit' | 'conditioner'>) {
  const payload = checkpointBundlePayload() as Record<string, unknown>;
  for (const subtree of missingSubtrees) {
    delete payload[subtree];
  }
  return payload;
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
  const extDir = typeof payload.ext_dir === 'string' ? payload.ext_dir : null;
  const stubDir = extDir ? join(extDir, 'py-stubs') : null;
  const pythonPath = [stubDir, resolve(repoRoot, 'runtime', 'vendor'), options.env?.PYTHONPATH].filter(Boolean).join(':');

  const outcome = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
    cwd: options.cwd ?? repoRoot,
    encoding: 'utf8',
    input: `${JSON.stringify(payload)}\n`,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      ...options.env,
    },
  });

  return {
    ...outcome,
    result: outcome.stdout ? (JSON.parse(outcome.stdout) as Record<string, unknown>) : null,
  };
}

function runPythonSnippet(source: string, args: string[] = [], pythonPathEntries: string[] = []) {
  return spawnSync('python3', ['-c', source, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONPATH: [...pythonPathEntries, resolve(repoRoot, 'runtime', 'vendor'), process.env.PYTHONPATH]
        .filter(Boolean)
        .join(':'),
    },
  });
}

describe('UltraShape processor.py protocol', () => {
  it('exposes tensor-first preprocess and surface state without first-class synthetic token keys', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = runPythonSnippet(
        [
          'import json, sys',
          'from ultrashape_runtime.preprocessors import ImageProcessorV2',
          'from ultrashape_runtime.surface_loaders import SharpEdgeSurfaceLoader',
          'reference_asset = ImageProcessorV2().process(sys.argv[1])',
          'surface_state = SharpEdgeSurfaceLoader().load(sys.argv[2])',
          'payload = {',
          '  "preprocess_keys": sorted(reference_asset.keys()),',
          '  "surface_keys": sorted(surface_state.keys()),',
          '  "mesh_keys": sorted(surface_state["mesh"].keys()),',
          '  "voxel_keys": sorted(surface_state["voxel_cond"].keys()),',
          '  "image_tensor_shape": reference_asset.get("image_tensor_shape"),',
          '  "mask_tensor_shape": reference_asset.get("mask_tensor_shape"),',
          '  "image_meta": reference_asset.get("image_meta"),',
          '}',
          'print(json.dumps(payload))',
        ].join('\n'),
        [fixture.referenceImage, fixture.namedCoarseMesh],
        [join(fixture.root, 'py-stubs')],
      );

      expect(outcome.status).toBe(0);
      const payload = JSON.parse(outcome.stdout) as Record<string, unknown>;
      expect(payload.preprocess_keys).toEqual(
        expect.arrayContaining(['image_meta', 'image_tensor', 'image_tensor_shape', 'mask_tensor', 'mask_tensor_shape']),
      );
      expect(payload.preprocess_keys).not.toEqual(
        expect.arrayContaining(['tokens', 'image_tokens', 'mask_tokens']),
      );
      expect(payload.surface_keys).toEqual(expect.arrayContaining(['mesh', 'sampled_surface_points', 'voxel_cond']));
      expect(payload.mesh_keys).toEqual(
        expect.arrayContaining(['bounds', 'faces', 'sampled_surface_points', 'surface_point_count', 'vertex_count', 'vertices']),
      );
      expect(payload.mesh_keys).not.toEqual(expect.arrayContaining(['tokens']));
      expect(payload.voxel_keys).toEqual(
        expect.arrayContaining(['bounds', 'coords', 'occupancies', 'occupied_ratio', 'resolution', 'surface_point_count']),
      );
      expect(payload.voxel_keys).not.toEqual(expect.arrayContaining(['tokens', 'voxel_coords', 'voxel_values']));
      expect(payload.image_tensor_shape).toEqual([1, 2, 2, 4]);
      expect(payload.mask_tensor_shape).toEqual([1, 2, 2, 1]);
      expect(payload.image_meta).toEqual(
        expect.objectContaining({
          width: 2,
          height: 2,
          pixel_count: 4,
          source_format: 'png',
        }),
      );
    } finally {
      fixture.cleanup();
    }
  });

  it('exposes a real conditioner contract with cfg-ready context and traceable source metadata', () => {
    const fixture = createFixtureWorkspace();
    const checkpointPath = join(fixture.root, 'ultrashape_v1.pt');
    writeBinaryCheckpointBundle(checkpointPath, checkpointBundlePayload());

    try {
      const outcome = runPythonSnippet(
        [
          'import json, sys',
          'from ultrashape_runtime.preprocessors import ImageProcessorV2',
          'from ultrashape_runtime.surface_loaders import SharpEdgeSurfaceLoader',
          'from ultrashape_runtime.models.conditioner_mask import SingleImageEncoder',
          'from ultrashape_runtime.utils.checkpoint import load_checkpoint_subtrees',
          'reference_asset = ImageProcessorV2().process(sys.argv[1])',
          'surface_state = SharpEdgeSurfaceLoader().load(sys.argv[2])',
          'checkpoint_bundle = load_checkpoint_subtrees(sys.argv[3], None, sys.argv[4])',
          'conditioning = SingleImageEncoder(checkpoint_state=checkpoint_bundle["bundle"]["conditioner"]).build(',
          '  reference_asset=reference_asset,',
          '  coarse_surface=surface_state,',
          ')',
          'payload = {',
          '  "conditioning_keys": sorted(conditioning.keys()),',
          '  "metadata_keys": sorted(conditioning["metadata"].keys()),',
          '  "context_length": len(conditioning["context"]),',
          '  "context_mask_length": len(conditioning["context_mask"]),',
          '  "cfg_pairing": conditioning["cfg_pairing"],',
          '  "metadata": conditioning["metadata"],',
          '}',
          'print(json.dumps(payload))',
        ].join('\n'),
        [fixture.referenceImage, fixture.namedCoarseMesh, checkpointPath, fixture.root],
        [join(fixture.root, 'py-stubs')],
      );

      expect(outcome.status).toBe(0);
      const payload = JSON.parse(outcome.stdout) as Record<string, unknown>;
      expect(payload.conditioning_keys).toEqual(
        expect.arrayContaining(['cfg_pairing', 'context', 'context_mask', 'metadata']),
      );
      expect(payload.conditioning_keys).not.toEqual(
        expect.arrayContaining(['tokens', 'mask_tokens', 'image_token_signature', 'conditioning_signature']),
      );
      expect(payload.metadata_keys).toEqual(
        expect.arrayContaining(['checkpoint_signature', 'image_signature', 'mesh_signature', 'voxel_signature']),
      );
      expect(payload.context_length).toEqual(expect.any(Number));
      expect(Number(payload.context_length)).toBeGreaterThan(0);
      expect(payload.context_mask_length).toBe(payload.context_length);
      expect(payload.cfg_pairing).toEqual(
        expect.objectContaining({
          mode: 'classifier-free-guidance',
          positive_context_tokens: expect.any(Number),
          negative_context_tokens: expect.any(Number),
        }),
      );
      expect(payload.metadata).toEqual(
        expect.objectContaining({
          image_signature: expect.any(Number),
          mesh_signature: expect.any(Number),
          voxel_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          voxel_count: expect.any(Number),
          surface_point_count: expect.any(Number),
        }),
      );
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects deferred backend requests as invalid processor params under the local-only shell', () => {
    const fixture = createFixtureWorkspace();

    try {
      for (const backend of ['remote', 'hybrid'] as const) {
        const outcome = runProcessor({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.namedCoarseMesh },
            },
          },
          params: {
            backend,
          },
          workspaceDir: fixture.outputDir,
        });

        expect(outcome.status).toBe(0);
        expect(outcome.events.at(-1)).toEqual({
          type: 'error',
          message: expect.stringContaining('backend must be auto or local'),
          code: 'INVALID_PARAMS',
        });
      }
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-glb output formats as invalid processor params under the glb-only shell', () => {
    const fixture = createFixtureWorkspace();

    try {
      for (const outputFormat of ['obj', 'fbx', 'ply'] as const) {
        const outcome = runProcessor({
          input: {
            inputs: {
              reference_image: { filePath: fixture.referenceImage },
              coarse_mesh: { filePath: fixture.namedCoarseMesh },
            },
          },
          params: {
            output_format: outputFormat,
          },
          workspaceDir: fixture.outputDir,
        });

        expect(outcome.status).toBe(0);
        expect(outcome.events.at(-1)).toEqual({
          type: 'error',
          message: expect.stringContaining('output_format must be glb'),
          code: 'INVALID_PARAMS',
        });
      }
    } finally {
      fixture.cleanup();
    }
  });

  it('treats named reference_image and coarse_mesh inputs as the primary manifest contract on the supported glb-only path', () => {
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
            output_format: 'glb',
          },
          workspaceDir: fixture.outputDir,
        },
        {
          cwd: fixture.root,
        },
      );

      expect(outcome.status).toBe(0);
      expect(outcome.events[0]?.type).toBe('progress');
      expect(outcome.events[1]?.type).toBe('log');
      expect(outcome.events.at(-1)?.type).toBe('error');

      const resolutionLog = outcome.events[1];
      const resolved = JSON.parse(String(resolutionLog.message)) as Record<string, string>;
      expect(resolved.reference_image).toBe(fixture.referenceImage);
      expect(resolved.coarse_mesh).toBe(fixture.namedCoarseMesh);
      expect(resolved.backend).toBe('local');
      expect(resolved.output_format).toBe('glb');

      const error = outcome.events.at(-1);
      expect(error).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
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
          message: expect.stringContaining('backend must be auto or local'),
          code: 'INVALID_PARAMS',
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

    writeFileSync(
      configPath,
      [
        'model:',
        '  scope: mc-only',
        'runtime:',
        '  backend: local',
        '  requires_exact_closure: true',
        'export:',
        '  format: glb',
        'public_contract:',
        '  backend_modes:',
        '    - auto',
        '    - local',
        '  success_output_formats:',
        '    - glb',
        '  public_error_codes:',
        '    - DEPENDENCY_MISSING',
        '    - WEIGHTS_MISSING',
        '    - LOCAL_RUNTIME_UNAVAILABLE',
        'checkpoint:',
        '  required_subtrees:',
        '    - vae',
        '    - dit',
        '    - conditioner',
        'dependencies:',
        '  required:',
        '    imports:',
        '      - diffusers',
        '      - cubvh',
        'vae_config:',
        '  enabled: true',
        'dit_cfg:',
        '  enabled: true',
        'conditioner_config:',
        '  enabled: true',
        'preprocess:',
        '  image_processor: ImageProcessorV2',
        'image_processor_cfg:',
        '  enabled: true',
        'conditioning:',
        '  coarse_mesh_encoder: SharpEdgeSurfaceLoader',
        '  voxelizer: voxelize_from_point',
        'surface:',
        '  extraction: mc',
        '  loader: SharpEdgeSurfaceLoader',
        'scheduler:',
        '  family: flow-matching',
        'scheduler_cfg:',
        '  family: flow-matching',
        'decoder:',
        '  vae: ShapeVAE',
        '  volume_decoder: VanillaVDMVolumeDecoding',
        'gate:',
        '  mode: geometric-hard-gate',
        '',
      ].join('\n'),
    );
    writeFileSync(join(fixture.root, 'py-stubs', 'diffusers.py'), diffusersStubSource());
    writeBinaryCheckpointBundle(checkpointPath, checkpointBundlePayload());

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
        result: expect.objectContaining({
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.any(Object),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          warnings: [],
        }),
      });
      expect((outcome.result?.result as Record<string, unknown>) ?? {}).not.toHaveProperty('runtime_contract');
      const metrics = getRunnerMetrics(outcome.result);

      expect(metrics).toEqual(
        expect.objectContaining({
          chamfer: expect.any(Number),
          rms: expect.any(Number),
          topology_changed: true,
          extent_ratio: [1, 1, 1],
          execution_trace: ['preprocess', 'conditioning', 'scheduler', 'denoise', 'decode', 'extract'],
          preprocess: expect.objectContaining({
            byte_length: expect.any(Number),
            normalized_channels: 4,
          }),
          conditioning: expect.objectContaining({
            voxel_count: expect.any(Number),
            context_token_count: expect.any(Number),
            context_mask_token_count: expect.any(Number),
            cfg_pairing: expect.objectContaining({
              mode: 'classifier-free-guidance',
              positive_context_tokens: expect.any(Number),
              negative_context_tokens: expect.any(Number),
            }),
            conditioner_metadata: expect.objectContaining({
              checkpoint_signature: expect.any(Number),
              checkpoint_tensor_count: expect.any(Number),
              checkpoint_value_count: expect.any(Number),
              image_signature: expect.any(Number),
              mesh_signature: expect.any(Number),
              voxel_signature: expect.any(Number),
            }),
            state_hydrated: true,
            hydration: expect.objectContaining({
              module: 'SingleImageEncoder',
              load_style: 'load_state_dict',
              strict: false,
            }),
          }),
          scheduler: expect.objectContaining({
            family: 'flow-matching-euler-discrete',
            step_count: 30,
            sigma_start: expect.any(Number),
            sigma_end: expect.any(Number),
          }),
          denoise: expect.objectContaining({
            model: 'RefineDiT',
            attention: expect.any(String),
            checkpoint_signature: expect.any(Number),
            inputs: expect.objectContaining({
              latents: expect.objectContaining({
                count: expect.any(Number),
                signature: expect.any(Number),
              }),
              timestep: expect.objectContaining({
                count: 30,
                signature: expect.any(Number),
              }),
              context: expect.objectContaining({
                count: expect.any(Number),
                signature: expect.any(Number),
              }),
              context_mask: expect.objectContaining({
                count: expect.any(Number),
                signature: expect.any(Number),
              }),
              voxel_cond: expect.objectContaining({
                voxel_count: expect.any(Number),
                signature: expect.any(Number),
              }),
            }),
            latent_signature: expect.any(Number),
            timestep_count: 30,
            state_hydrated: true,
            hydration: expect.objectContaining({
              module: 'RefineDiT',
              load_style: 'load_state_dict',
              strict: false,
            }),
          }),
          checkpoint: expect.objectContaining({
            hydration: expect.arrayContaining([
              expect.objectContaining({ module: 'SingleImageEncoder' }),
              expect.objectContaining({ module: 'RefineDiT' }),
              expect.objectContaining({ module: 'ShapeVAE' }),
            ]),
          }),
          decode: expect.objectContaining({
            field_density: expect.any(Number),
          }),
          extract: expect.objectContaining({
            extractor: expect.any(String),
            payload_bytes: expect.any(Number),
          }),
        }),
      );
      expect(metrics.denoise).not.toEqual(
        expect.objectContaining({
          conditioning_signature: expect.any(Number),
        }),
      );
      expect(Number(metrics.rms)).toBeGreaterThan(0.01);
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects runner jobs whose checkpoint omits any required vae/dit/conditioner subtree', () => {
    const fixture = createFixtureWorkspace();
    const configPath = join(fixture.root, 'runtime-config.yaml');
    const checkpointPath = join(fixture.root, 'ultrashape_v1.pt');

    writeFileSync(
      configPath,
      [
        'model:',
        '  scope: mc-only',
        'runtime:',
        '  backend: local',
        '  requires_exact_closure: true',
        'export:',
        '  format: glb',
        'public_contract:',
        '  backend_modes:',
        '    - auto',
        '    - local',
        '  success_output_formats:',
        '    - glb',
        '  public_error_codes:',
        '    - DEPENDENCY_MISSING',
        '    - WEIGHTS_MISSING',
        '    - LOCAL_RUNTIME_UNAVAILABLE',
        'checkpoint:',
        '  required_subtrees:',
        '    - vae',
        '    - dit',
        '    - conditioner',
        'dependencies:',
        '  required:',
        '    imports:',
        '      - diffusers',
        '      - cubvh',
        'vae_config:',
        '  enabled: true',
        'dit_cfg:',
        '  enabled: true',
        'conditioner_config:',
        '  enabled: true',
        'preprocess:',
        '  image_processor: ImageProcessorV2',
        'image_processor_cfg:',
        '  enabled: true',
        'conditioning:',
        '  coarse_mesh_encoder: SharpEdgeSurfaceLoader',
        '  voxelizer: voxelize_from_point',
        'surface:',
        '  extraction: mc',
        '  loader: SharpEdgeSurfaceLoader',
        'scheduler:',
        '  family: flow-matching',
        'scheduler_cfg:',
        '  family: flow-matching',
        'decoder:',
        '  vae: ShapeVAE',
        '  volume_decoder: VanillaVDMVolumeDecoding',
        'gate:',
        '  mode: geometric-hard-gate',
        '',
      ].join('\n'),
    );
    writeFileSync(join(fixture.root, 'py-stubs', 'diffusers.py'), diffusersStubSource());

    try {
      for (const missingSubtree of ['vae', 'dit', 'conditioner'] as const) {
        writeBinaryCheckpointBundle(checkpointPath, checkpointBundleWithout(missingSubtree));

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
          },
        );

        expect(outcome.status).toBe(1);
        expect(outcome.result).toEqual({
          ok: false,
          error_code: 'WEIGHTS_MISSING',
          error_message: expect.stringContaining(missingSubtree),
        });
        expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
      }
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
      const metadataLog = outcome.events.find((event) => {
        if (event.type !== 'log' || typeof event.message !== 'string') {
          return false;
        }

        try {
          const payload = JSON.parse(event.message) as Record<string, unknown>;
          return Array.isArray(payload.subtrees_loaded);
        } catch {
          return false;
        }
      });

      expect(metadataLog?.type).toBe('log');
      expect(JSON.parse(String(metadataLog?.message))).toEqual({
        backend: 'local',
        metrics: {
          chamfer: 0.0042,
          rms: 0.0125,
          topology_changed: true,
          extent_ratio: [1, 1, 1],
        },
        fallbacks: [],
        subtrees_loaded: ['vae', 'dit', 'conditioner'],
      });
      expect(readFileSync(join(fixture.outputDir, 'refined.glb'), 'utf8')).toBe('runner-output');
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects runner success envelopes that omit checkpoint-backed execution metadata', () => {
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
            ULTRASHAPE_RUNNER_STDOUT_MODE: 'missing-metadata',
          },
        },
      );

      expect(outcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('checkpoint-backed execution metadata'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
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

  it('blocks publish-path success when the runner rejects passthrough geometry at the gate', () => {
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
            ULTRASHAPE_RUNNER_STDOUT_MODE: 'gate-error',
          },
        },
      );

      expect(outcome.events.map((event) => event.type)).not.toContain('done');
      expect(outcome.events.at(-1)).toEqual({
        type: 'error',
        message: expect.stringContaining('LOCAL_RUNTIME_UNAVAILABLE'),
        code: 'LOCAL_RUNTIME_UNAVAILABLE',
      });
      expect(String(outcome.events.at(-1)?.message ?? '')).not.toContain('GEOMETRIC_GATE_REJECTED');
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

});
