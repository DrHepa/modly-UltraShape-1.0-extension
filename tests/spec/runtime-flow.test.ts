import { chmodSync, cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { deflateSync } from 'node:zlib';

import { describe, expect, it } from 'vitest';

const repoRoot = process.cwd();
const processorPath = resolve(repoRoot, 'processor.py');
const runtimeVendorPath = resolve(repoRoot, 'runtime', 'vendor');
const runtimeConfigSourcePath = resolve(repoRoot, 'runtime', 'configs', 'infer_dit_refine.yaml');

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
    '    def cpu(self):',
    '        return self',
    '',
    '    def tolist(self):',
    '        return [list(row) if isinstance(row, (list, tuple)) else row for row in self._values]',
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
    'import json',
    'import os',
    'from pathlib import Path',
    '',
    'def _trace(payload):',
    '    trace_path = os.environ.get("ULTRASHAPE_TEST_DIFFUSERS_TRACE_PATH")',
    '    if not trace_path:',
    '        return',
    '    Path(trace_path).write_text(json.dumps(payload), encoding="utf8")',
    '',
    'class FlowMatchEulerDiscreteScheduler:',
    '    def __init__(self, **config):',
    '        self.config = dict(config)',
    '        self.timesteps = []',
    '        _trace({"event": "init", "config": self.config})',
    '',
    '    @classmethod',
    '    def from_config(cls, config):',
    '        payload = dict(config) if isinstance(config, dict) else {"value": config}',
    '        _trace({"event": "from_config", "config": payload})',
    '        return cls(**payload)',
    '',
    '    def set_timesteps(self, step_count):',
    '        self.timesteps = list(range(int(step_count)))',
    '        _trace({"event": "set_timesteps", "step_count": int(step_count), "timesteps": self.timesteps})',
    '',
  ].join('\n');
}

function cubvhStubSource() {
  return [
    '__version__ = "0.0-test"',
    'import json',
    'import os',
    'import torch',
    'from pathlib import Path',
    '',
    'def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):',
    '    coords = coords.int()',
    '    corners = corners.float()',
    '    trace_path = os.environ.get("ULTRASHAPE_TEST_CUBVH_TRACE_PATH")',
    '    if trace_path:',
    '        Path(trace_path).write_text(json.dumps({',
    '            "coords_type": type(coords).__name__,',
    '            "corners_type": type(corners).__name__,',
    '            "coords_dtype": getattr(coords, "dtype", None),',
    '            "corners_dtype": getattr(corners, "dtype", None),',
    '            "coords": len(coords),',
    '            "corners": len(corners),',
    '            "iso": iso,',
    '            "ensure_consistency": ensure_consistency,',
    '        }), encoding="utf8")',
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
    '    if os.environ.get("ULTRASHAPE_TEST_CUBVH_RETURNS_TENSORS") == "1":',
    '        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32)',
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
    '        vertex_view_offset = 0',
    '        index_view_offset = len(vertex_bytes)',
    '        binary_blob = _pad(vertex_bytes, b"\\x00") + _pad(index_bytes, b"\\x00")',
    '        json_doc = {',
    '            "asset": {"version": "2.0", "generator": "trimesh-test-stub"},',
    '            "scene": 0,',
    '            "scenes": [{"nodes": [0]}],',
    '            "nodes": [{"mesh": 0, "name": node_name}],',
    '            "meshes": [{"name": node_name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],',
    '            "buffers": [{"byteLength": len(binary_blob)}],',
    '            "bufferViews": [',
    '                {"buffer": 0, "byteOffset": vertex_view_offset, "byteLength": len(vertex_bytes), "target": 34962},',
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

function inspectCheckpointBundle(checkpointPath: string, extDir: string, stubDir: string) {
  const outcome = spawnSync(
    'python3',
    [
      '-c',
      [
        'import json, sys',
        'from ultrashape_runtime.utils.checkpoint import load_checkpoint_subtrees',
        'result = load_checkpoint_subtrees(sys.argv[1], None, sys.argv[2])',
        'print(json.dumps(result))',
      ].join('\n'),
      checkpointPath,
      extDir,
    ],
    {
      encoding: 'utf8',
      env: {
        ...process.env,
        PYTHONPATH: [stubDir, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
      },
    },
  );

  if (outcome.status !== 0) {
    throw new Error(outcome.stderr || 'Failed to inspect checkpoint bundle.');
  }

  return JSON.parse(outcome.stdout) as Record<string, unknown>;
}

function expectUpstreamRuntimeGraph(configText: string) {
  expect(configText).toContain('checkpoint:');
  expect(configText).toContain('vae_config:');
  expect(configText).toContain('dit_cfg:');
  expect(configText).toContain('conditioner_config:');
  expect(configText).toContain('preprocess:');
  expect(configText).toContain('image_processor_cfg:');
  expect(configText).toContain('conditioning:');
  expect(configText).toContain('scheduler:');
  expect(configText).toContain('scheduler_cfg:');
  expect(configText).toContain('decoder:');
  expect(configText).toContain('surface:');
  expect(configText).toContain('gate:');
  expect(configText).toContain('export:');
  expect(configText).toContain('scope: mc-only');
  expect(configText).toContain('backend: local');
  expect(configText).toContain('format: glb');
  expect(configText).toContain('requires_exact_closure: true');
  expect(configText).toContain('required:');
  expect(configText).toContain('- diffusers');
  expect(configText).toContain('- cubvh');
  expect(configText).toContain('conditional:');
  expect(configText).toContain('- rembg');
  expect(configText).toContain('- onnxruntime');
  expect(configText).toContain('degradable:');
  expect(configText).toContain('- flash_attn');
}

function createBinaryGlbBytes() {
  const jsonChunk = Buffer.from('{"asset":{"version":"2.0"}}   ', 'utf8');
  const binaryChunk = Buffer.from([0x00, 0x80, 0x00, 0x00]);
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binaryChunk.length;

  return Buffer.concat([
    Buffer.from('glTF', 'ascii'),
    Buffer.from(Uint32Array.of(2, totalLength).buffer),
    Buffer.from(Uint32Array.of(jsonChunk.length, 0x4e4f534a).buffer),
    jsonChunk,
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

function createRgbaPngBytes(width: number, height: number, pixels: number[][]) {
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

function createReferenceImageBytes(variant: 'a' | 'b' = 'a') {
  const pixels =
    variant === 'a'
      ? [
          [255, 32, 32, 255],
          [32, 255, 32, 255],
          [32, 32, 255, 255],
          [240, 220, 48, 160],
        ]
      : [
          [20, 20, 20, 255],
          [220, 120, 40, 255],
          [180, 40, 200, 200],
          [40, 220, 240, 96],
        ];

  return createRgbaPngBytes(2, 2, pixels);
}

function createMeshGlbBytes(vertices: Array<[number, number, number]>, faces: Array<[number, number, number]>) {
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
  const binaryBlob = Buffer.concat([paddedVertexBytes, paddedIndexBytes]);
  const jsonDocument = {
    asset: { version: '2.0', generator: 'runtime-flow-test-fixture' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'coarse-mesh' }],
    meshes: [{ name: 'coarse-mesh', primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    buffers: [{ byteLength: binaryBlob.length }],
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
        min: [
          Math.min(...vertices.map((vertex) => vertex[0])),
          Math.min(...vertices.map((vertex) => vertex[1])),
          Math.min(...vertices.map((vertex) => vertex[2])),
        ],
        max: [
          Math.max(...vertices.map((vertex) => vertex[0])),
          Math.max(...vertices.map((vertex) => vertex[1])),
          Math.max(...vertices.map((vertex) => vertex[2])),
        ],
      },
      { bufferView: 1, componentType: 5125, count: faces.length * 3, type: 'SCALAR' },
    ],
  };

  const jsonBytes = Buffer.from(JSON.stringify(jsonDocument), 'utf8');
  const paddedJsonBytes = Buffer.concat([jsonBytes, Buffer.alloc((4 - (jsonBytes.length % 4)) % 4, 0x20)]);
  const totalLength = 12 + 8 + paddedJsonBytes.length + 8 + binaryBlob.length;

  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(totalLength, 8);

  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(paddedJsonBytes.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);

  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binaryBlob.length, 0);
  binHeader.writeUInt32LE(0x004e4942, 4);

  return Buffer.concat([header, jsonHeader, paddedJsonBytes, binHeader, binaryBlob]);
}

function createCoarseMeshBytes(variant: 'a' | 'b' = 'a') {
  if (variant === 'a') {
    return createMeshGlbBytes(
      [
        [-0.9, -0.7, -0.5],
        [0.8, -0.6, -0.4],
        [0.9, 0.7, -0.3],
        [-0.8, 0.8, -0.2],
        [-0.6, -0.5, 0.7],
        [0.7, -0.4, 0.9],
        [0.6, 0.5, 0.8],
        [-0.7, 0.6, 0.6],
      ],
      [
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
      ],
    );
  }

  return createMeshGlbBytes(
    [
      [-0.4, -0.9, -0.7],
      [0.9, -0.7, -0.2],
      [1.1, 0.1, -0.1],
      [0.5, 0.9, 0.3],
      [-0.3, 1.0, 0.6],
      [-1.0, 0.2, 0.4],
      [-0.8, -0.2, 0.9],
      [0.2, -0.4, 1.0],
    ],
    [
      [0, 1, 7],
      [1, 2, 7],
      [2, 3, 7],
      [3, 4, 7],
      [4, 5, 6],
      [4, 6, 7],
      [5, 0, 6],
      [0, 7, 6],
      [0, 1, 5],
      [1, 2, 3],
      [1, 3, 5],
      [3, 4, 5],
    ],
  );
}

function readGlbJson(path: string) {
  const payload = readFileSync(path);
  expect(payload.subarray(0, 4).toString('ascii')).toBe('glTF');
  expect(payload.length).toBeGreaterThanOrEqual(24);
  const jsonLength = payload.readUInt32LE(12);
  const jsonChunkType = payload.readUInt32LE(16);
  expect(jsonChunkType).toBe(0x4e4f534a);
  return JSON.parse(payload.subarray(20, 20 + jsonLength).toString('utf8').trim().replace(/\u0000+$/u, '')) as {
    asset?: { version?: string };
    scenes?: unknown[];
    nodes?: unknown[];
    meshes?: Array<{ primitives?: Array<{ attributes?: { POSITION?: unknown }; indices?: unknown }> }>;
    accessors?: unknown[];
    bufferViews?: unknown[];
    buffers?: unknown[];
  };
}

function getGlbPositionAccessorCount(path: string) {
  const document = readGlbJson(path);
  const primitive = document.meshes?.[0]?.primitives?.[0];
  const positionAccessorIndex = primitive?.attributes?.POSITION;
  expect(positionAccessorIndex).toEqual(expect.any(Number));
  const accessor = document.accessors?.[positionAccessorIndex as number] as { count?: unknown } | undefined;
  expect(accessor?.count).toEqual(expect.any(Number));
  return accessor?.count as number;
}

function expectRenderableGlbMesh(path: string) {
  const document = readGlbJson(path);
  expect(document.asset?.version).toBe('2.0');
  expect(document.scenes?.length ?? 0).toBeGreaterThan(0);
  expect(document.nodes?.length ?? 0).toBeGreaterThan(0);
  expect(document.meshes?.length ?? 0).toBeGreaterThan(0);
  expect(document.bufferViews?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(document.accessors?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(document.buffers?.length ?? 0).toBe(1);
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
  const root = mkdtempSync(join(tmpdir(), 'ultrashape-runtime-'));
  const outputDir = join(root, 'output');
  const modelsDir = join(root, 'models', 'ultrashape');
  const stubDir = join(root, 'py-stubs');
  mkdirSync(outputDir);
  mkdirSync(modelsDir, { recursive: true });
  mkdirSync(stubDir, { recursive: true });

  const referenceImage = join(root, 'reference.png');
  const coarseMesh = join(root, 'coarse.glb');
  const binaryCoarseMesh = join(root, 'binary-coarse.glb');
  const packagedArtifact = join(root, 'artifact.glb');
  const checkpoint = join(modelsDir, 'ultrashape_v1.pt');
  const configPath = join(root, 'infer_dit_refine.yaml');

  writeFileSync(referenceImage, createReferenceImageBytes('a'));
  writeFileSync(coarseMesh, createCoarseMeshBytes('a'));
  writeFileSync(binaryCoarseMesh, createCoarseMeshBytes('b'));
  writeFileSync(packagedArtifact, 'refined-mesh');
  writeFileSync(join(stubDir, 'cubvh.py'), cubvhStubSource());
  writeFileSync(join(stubDir, 'diffusers.py'), diffusersStubSource());
  writeFileSync(join(stubDir, 'torch.py'), torchCheckpointStubSource());
  writeFileSync(join(stubDir, 'trimesh.py'), trimeshStubSource());
  writeCheckpointBundle(checkpoint);

  return {
    root,
    outputDir,
     referenceImage,
     coarseMesh,
     binaryCoarseMesh,
      packagedArtifact,
      checkpoint,
      configPath,
      stubDir,
      cleanup: () => rmSync(root, { recursive: true, force: true }),
  };
}

function checkpointBundlePayload(variant: 'a' | 'b' = 'a') {
  const values =
    variant === 'a'
      ? {
          vae: [0.11, 0.33, 0.55, 0.77],
          dit: [0.21, 0.41, 0.61, 0.81],
          conditioner: [0.14, 0.24, 0.64, 0.74],
        }
      : {
          vae: [0.77, 0.55, 0.33, 0.11],
          dit: [0.81, 0.61, 0.41, 0.21],
          conditioner: [0.74, 0.64, 0.24, 0.14],
        };

  return {
    format: 'ultrashape-checkpoint-bundle/v1',
    vae: {
      tensors: {
        latent_basis: values.vae,
      },
    },
    dit: {
      tensors: {
        attention_bias: values.dit,
      },
    },
    conditioner: {
      tensors: {
        mask_bias: values.conditioner,
      },
    },
  };
}

function checkpointBundleWithout(...missingSubtrees: Array<'vae' | 'dit' | 'conditioner'>) {
  const payload = checkpointBundlePayload('a') as Record<string, unknown>;
  for (const subtree of missingSubtrees) {
    delete payload[subtree];
  }
  return payload;
}

function writeCheckpointBundle(path: string, variant: 'a' | 'b' = 'a') {
  writeBinaryCheckpointBundle(path, checkpointBundlePayload(variant));
}

function expectRealClosureMetrics(metrics: Record<string, unknown>) {
  expect(metrics).toEqual(
    expect.objectContaining({
      chamfer: expect.any(Number),
      rms: expect.any(Number),
      topology_changed: true,
      extent_ratio: [1, 1, 1],
      execution_trace: ['preprocess', 'conditioning', 'scheduler', 'denoise', 'decode', 'extract'],
      preprocess: expect.objectContaining({
        processor: 'ImageProcessorV2',
        byte_length: expect.any(Number),
        normalized_channels: 4,
        image_tensor_shape: [1, 2, 2, 4],
        image_feature_count: expect.any(Number),
        mask_feature_count: expect.any(Number),
        image_signature: expect.any(Number),
        mask_signature: expect.any(Number),
      }),
      conditioning: expect.objectContaining({
        surface_loader: 'SharpEdgeSurfaceLoader',
        encoder: 'SingleImageEncoder',
        voxelizer: 'voxelize_from_point',
        surface_vertex_count: expect.any(Number),
        surface_face_count: expect.any(Number),
        surface_point_count: expect.any(Number),
        voxel_count: expect.any(Number),
        voxel_resolution: expect.any(Number),
        mask_tokens: expect.any(Number),
        surface_signature: expect.any(Number),
        voxel_signature: expect.any(Number),
        image_token_signature: expect.any(Number),
        checkpoint_signature: expect.any(Number),
        conditioning_signature: expect.any(Number),
      }),
      scheduler: expect.objectContaining({
        family: 'flow-matching-euler-discrete',
        target: 'diffusers.FlowMatchEulerDiscreteScheduler',
        step_count: expect.any(Number),
        timestep_signature: expect.any(Number),
      }),
      denoise: expect.objectContaining({
        model: 'RefineDiT',
        attention: expect.any(String),
        checkpoint_signature: expect.any(Number),
        scheduler_signature: expect.any(Number),
        latent_signature: expect.any(Number),
      }),
      decode: expect.objectContaining({
        vae: 'ShapeVAE',
        decoder: expect.stringMatching(/VDMVolumeDecoding|VolumeDecoder/u),
        mesh_signature: expect.any(Number),
        field_density: expect.any(Number),
        cell_count: expect.any(Number),
        grid_resolution: expect.any(Number),
      }),
      extract: expect.objectContaining({
        extractor: 'cubvh.sparse_marching_cubes',
        marching_cubes: 'cubvh.sparse_marching_cubes',
        vertex_count: expect.any(Number),
        face_count: expect.any(Number),
        payload_bytes: expect.any(Number),
      }),
      gate: expect.objectContaining({
        coarse_vertex_count: expect.any(Number),
        refined_vertex_count: expect.any(Number),
        coarse_face_count: expect.any(Number),
        refined_face_count: expect.any(Number),
      }),
    }),
  );
}

function installProcessorRuntime(extDir: string) {
  const runtimeDir = join(extDir, 'runtime');
  const runtimePackageDir = join(runtimeDir, 'ultrashape_runtime');
  const runtimeConfigDir = join(runtimeDir, 'configs');
  const venvBinDir = join(extDir, 'venv', 'bin');
  const modelsDir = join(extDir, 'models', 'ultrashape');
  const pythonShimPath = join(venvBinDir, 'python');

  mkdirSync(runtimeConfigDir, { recursive: true });
  mkdirSync(venvBinDir, { recursive: true });
  mkdirSync(modelsDir, { recursive: true });

  cpSync(resolve(repoRoot, 'runtime', 'vendor', 'ultrashape_runtime'), runtimePackageDir, { recursive: true });
  writeFileSync(join(runtimeDir, 'torch.py'), torchCheckpointStubSource());
  writeFileSync(join(runtimeDir, 'torchvision.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'numpy.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'trimesh.py'), trimeshStubSource());
  mkdirSync(join(runtimeDir, 'PIL'), { recursive: true });
  writeFileSync(join(runtimeDir, 'PIL', '__init__.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'cv2.py'), '__version__ = "0.0-test"\n');
  mkdirSync(join(runtimeDir, 'skimage'), { recursive: true });
  writeFileSync(join(runtimeDir, 'skimage', '__init__.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'yaml.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'omegaconf.py'), 'class OmegaConf:\n    pass\n');
  writeFileSync(join(runtimeDir, 'einops.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'transformers.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'huggingface_hub.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'accelerate.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'cubvh.py'), cubvhStubSource());
  writeFileSync(join(runtimeDir, 'diffusers.py'), diffusersStubSource());
  writeFileSync(join(runtimeDir, 'safetensors.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeDir, 'tqdm.py'), '__version__ = "0.0-test"\n');
  writeFileSync(join(runtimeConfigDir, 'infer_dit_refine.yaml'), readFileSync(runtimeConfigSourcePath, 'utf8'));
  writeCheckpointBundle(join(modelsDir, 'ultrashape_v1.pt'));
  writeFileSync(
    pythonShimPath,
    ['#!/usr/bin/env bash', 'set -euo pipefail', 'exec python3 "$@"', ''].join('\n'),
  );
  chmodSync(pythonShimPath, 0o755);
}

function writeRuntimeConfig(
  path: string,
  overrides: Partial<{
    scope: string;
    backend: string;
    extraction: string;
    primaryWeight: string;
    requiredImports: string[];
  }> = {},
) {
  const requiredImports = overrides.requiredImports ?? [];
  writeFileSync(
    path,
    [
      'model:',
      `  scope: ${overrides.scope ?? 'mc-only'}`,
      'runtime:',
      `  backend: ${overrides.backend ?? 'local'}`,
      '  requires_exact_closure: true',
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
      '  primary: models/ultrashape/ultrashape_v1.pt',
      '  required_subtrees:',
      '    - vae',
      '    - dit',
      '    - conditioner',
      'vae_config:',
      '  target: ultrashape_runtime.models.autoencoders.model.ShapeVAE',
      'dit_cfg:',
      '  target: ultrashape_runtime.models.denoisers.dit_mask.RefineDiT',
      'conditioner_config:',
      '  target: ultrashape_runtime.models.conditioner_mask.SingleImageEncoder',
      'preprocess:',
      '  image_processor: ImageProcessorV2',
      '  require_cutout: conditional',
      'image_processor_cfg:',
      '  target: ultrashape_runtime.preprocessors.ImageProcessorV2',
      'conditioning:',
      '  coarse_mesh_encoder: SharpEdgeSurfaceLoader',
      '  voxelizer: voxelize_from_point',
      'surface:',
      `  extraction: ${overrides.extraction ?? 'mc'}`,
      '  loader: SharpEdgeSurfaceLoader',
      'scheduler:',
      '  family: flow-matching',
      'scheduler_cfg:',
      '  target: diffusers.FlowMatchEulerDiscreteScheduler',
      'decoder:',
      '  vae: ShapeVAE',
      '  volume_decoder: VanillaVDMVolumeDecoding',
      'weights:',
      `  primary: ${overrides.primaryWeight ?? 'models/ultrashape/ultrashape_v1.pt'}`,
      'gate:',
      '  mode: geometric-hard-gate',
      'export:',
      '  format: glb',
      'dependencies:',
      '  required:',
      '    imports:',
      '      - diffusers',
      ...requiredImports.map((entry) => `      - ${entry}`),
      '      - cubvh',
      '  conditional:',
      '    - rembg',
      '    - onnxruntime',
      '  degradable:',
      '    - flash_attn',
      '',
    ].join('\n'),
  );
}

function writeSyntheticSuccessConfig(path: string) {
  writeFileSync(
    path,
    [
      'model:',
      '  scope: mc-only',
      'runtime:',
      '  backend: local',
      'surface:',
      '  extraction: mc',
      'weights:',
      '  primary: models/ultrashape/ultrashape_v1.pt',
      '',
    ].join('\n'),
  );
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

function runLocalRunner(payload: Record<string, unknown>, options: { env?: NodeJS.ProcessEnv; cwd?: string } = {}) {
  const extDir = typeof payload.ext_dir === 'string' ? payload.ext_dir : null;
  const stubDir = extDir ? join(extDir, 'py-stubs') : null;
  const pythonPath = [stubDir, runtimeVendorPath, options.env?.PYTHONPATH].filter(Boolean).join(':');

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

  return outcome.stdout ? (JSON.parse(outcome.stdout) as Record<string, unknown>) : null;
}

function runPythonSnippet(source: string, args: string[] = [], pythonPathEntries: string[] = []) {
  return spawnSync('python3', ['-c', source, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONPATH: [...pythonPathEntries, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
    },
  });
}

describe('UltraShape runtime flow', () => {
  it('ships an upstream-style runtime graph that truthfully encodes the real-refinement dependency tiers', () => {
    expectUpstreamRuntimeGraph(readFileSync(runtimeConfigSourcePath, 'utf8'));
  });

  it('runs the repo-root Python boundary from the named-input contract and packages refined.<format> without any JS install artifact', () => {
    const fixture = createFixtureWorkspace();

    try {
      installProcessorRuntime(fixture.root);
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
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
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
      installProcessorRuntime(installedExtDir);
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
          processorPath: installedProcessorPath,
        },
      );

      expect(events.at(-1)).toEqual({
        type: 'done',
        result: {
          filePath: join(fixture.outputDir, 'refined.glb'),
        },
      });
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
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

  it('runs the vendored local runner to generate refined.glb inside the requested output directory', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.any(Object),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          runtime_contract: {
            backend: 'local-only',
            scope: 'mc-only',
            output_format: 'glb-only',
            requires_exact_closure: true,
            checkpoint_subtrees: ['vae', 'dit', 'conditioner'],
            public_error_codes: ['DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE'],
          },
          warnings: [],
        },
      });
      expectRealClosureMetrics(getRunnerMetrics(outcome));
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
      expect(getGlbPositionAccessorCount(join(fixture.outputDir, 'refined.glb'))).toBeGreaterThan(8);
    } finally {
      fixture.cleanup();
    }
  });

  it('accepts binary glb coarse meshes without utf8 decode failures and writes a binary refined.glb', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.binaryCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.any(Object),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          runtime_contract: {
            backend: 'local-only',
            scope: 'mc-only',
            output_format: 'glb-only',
            requires_exact_closure: true,
            checkpoint_subtrees: ['vae', 'dit', 'conditioner'],
            public_error_codes: ['DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE'],
          },
          warnings: [],
        },
      });
      expectRealClosureMetrics(getRunnerMetrics(outcome));
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
      expect(getGlbPositionAccessorCount(join(fixture.outputDir, 'refined.glb'))).toBeGreaterThan(8);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-mc scope configs at the vendored runner seam as LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath, {
        scope: 'full-volume',
      });

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('mc-only'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects shorthand synthetic-success configs that omit the exact runtime truth markers', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeSyntheticSuccessConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('exact runtime truth'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects non-glb output requests at the vendored runner seam as LOCAL_RUNTIME_UNAVAILABLE', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'obj',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('glb-only'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing checkpoint drift after config resolution to WEIGHTS_MISSING at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      rmSync(fixture.checkpoint);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: expect.stringContaining('ultrashape_v1.pt'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects checkpoints that do not contain the required vae/dit/conditioner subtrees', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      for (const missingSubtree of ['vae', 'dit', 'conditioner'] as const) {
        writeBinaryCheckpointBundle(fixture.checkpoint, checkpointBundleWithout(missingSubtree));

        const outcome = runLocalRunner({
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: null,
          preserve_scale: true,
        });

        expect(outcome).toEqual({
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

  it('rejects plain JSON subtree stubs that do not satisfy the real checkpoint tensor contract', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeBinaryCheckpointBundle(fixture.checkpoint, {
        vae: { weights: 'fixture-vae' },
        dit: { weights: 'fixture-dit' },
        conditioner: { weights: 'fixture-conditioner' },
      });

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: expect.stringContaining('tensor'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('stores compact checkpoint summaries instead of full tensor float lists', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeBinaryCheckpointBundle(fixture.checkpoint, {
        vae: { tensors: { latent_basis: Array.from({ length: 16 }, (_, index) => (index + 1) / 20) } },
        dit: { tensors: { attention_bias: Array.from({ length: 12 }, (_, index) => (index + 2) / 20) } },
        conditioner: { tensors: { mask_bias: Array.from({ length: 10 }, (_, index) => (index + 3) / 20) } },
      });

      const inspected = inspectCheckpointBundle(fixture.checkpoint, fixture.root, fixture.stubDir);
      const bundle = inspected.bundle as Record<string, unknown>;
      const vae = bundle.vae as Record<string, unknown>;
      const tensors = vae.tensors as Record<string, unknown>;
      const latentBasis = tensors.latent_basis as Record<string, unknown>;

      expect(inspected.summary).toEqual(
        expect.objectContaining({
          representation: 'tensor-summary-v1',
          tensor_count: 3,
          value_count: 38,
        }),
      );
      expect(vae).toEqual(
        expect.objectContaining({
          representation: 'checkpoint-subtree-v1',
          value_count: 16,
          tokens: expect.any(Array),
          state_dict: expect.objectContaining({
            tensors: expect.any(Object),
          }),
        }),
      );
      expect(latentBasis).toEqual(
        expect.objectContaining({
          sample_count: 8,
          value_count: 16,
          sample: expect.any(Array),
          mean: expect.any(Number),
        }),
      );
      expect((latentBasis.sample as unknown[])).toHaveLength(8);
      expect(vae.tokens).toEqual(latentBasis.sample);
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing config bootstrap failures to LOCAL_RUNTIME_UNAVAILABLE at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: fixture.checkpoint,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('config_path'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing required runtime imports to DEPENDENCY_MISSING at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath, {
        requiredImports: ['module_that_does_not_exist_for_ultrashape_tests'],
      });

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: null,
        preserve_scale: true,
      });

      expect(outcome).toEqual({
        ok: false,
        error_code: 'DEPENDENCY_MISSING',
        error_message: expect.stringContaining('module_that_does_not_exist_for_ultrashape_tests'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('maps missing output generation to LOCAL_RUNTIME_UNAVAILABLE at the vendored runner seam', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: null,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_SKIP_OUTPUT_WRITE: '1',
          },
        },
      );

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('refined.glb'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('derives closure metrics from executed inputs instead of placeholder formulas', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      writeFileSync(fixture.referenceImage, createReferenceImageBytes('b'));

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.binaryCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.preprocess).toEqual(
        expect.objectContaining({
          image_tensor_shape: [1, 2, 2, 4],
          image_feature_count: expect.any(Number),
        }),
      );
      expect(firstMetrics.conditioning).toEqual(
        expect.objectContaining({
          surface_vertex_count: 8,
          surface_face_count: 12,
          voxel_resolution: 12,
        }),
      );
      expect(firstMetrics.preprocess).not.toEqual(secondMetrics.preprocess);
      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
    } finally {
      fixture.cleanup();
    }
  });

  it('changes mesh-conditioned downstream metadata when only the coarse mesh changes', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.binaryCoarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.preprocess).toEqual(secondMetrics.preprocess);
      expect(firstMetrics.conditioning).toEqual(
        expect.objectContaining({
          surface_vertex_count: 8,
          surface_face_count: 12,
          surface_point_count: expect.any(Number),
        }),
      );
      expect(secondMetrics.conditioning).toEqual(
        expect.objectContaining({
          surface_vertex_count: 8,
          surface_face_count: 12,
          surface_point_count: expect.any(Number),
        }),
      );
      expect(firstMetrics.gate).toEqual(
        expect.objectContaining({
          reference_image_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(secondMetrics.gate).toEqual(
        expect.objectContaining({
          reference_image_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
      expect(firstMetrics.gate).not.toEqual(secondMetrics.gate);
    } finally {
      fixture.cleanup();
    }
  });

  it('changes image-conditioned downstream metadata when only the reference image changes', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      writeFileSync(fixture.referenceImage, createReferenceImageBytes('b'));

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.preprocess).toEqual(
        expect.objectContaining({
          image_tensor_shape: [1, 2, 2, 4],
          image_feature_count: expect.any(Number),
          mask_feature_count: expect.any(Number),
        }),
      );
      expect(firstMetrics.gate).toEqual(
        expect.objectContaining({
          coarse_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(secondMetrics.gate).toEqual(
        expect.objectContaining({
          coarse_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.preprocess).not.toEqual(secondMetrics.preprocess);
      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
      expect(firstMetrics.gate).not.toEqual(secondMetrics.gate);
    } finally {
      fixture.cleanup();
    }
  });

  it('publishes checkpoint-backed image conditioning metadata from SingleImageEncoder', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(getRunnerMetrics(outcome).conditioning).toEqual(
        expect.objectContaining({
          encoder: 'SingleImageEncoder',
          surface_vertex_count: 8,
          surface_face_count: 12,
          image_token_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          checkpoint_tensor_count: expect.any(Number),
          checkpoint_value_count: expect.any(Number),
          conditioning_signature: expect.any(Number),
          state_hydrated: true,
          hydration: expect.objectContaining({
            module: 'SingleImageEncoder',
            load_style: 'load_state_dict',
            strict: false,
          }),
        }),
      );
    } finally {
      fixture.cleanup();
    }
  });

  it('uses FlowMatchEulerDiscreteScheduler-guided RefineDiT denoising instead of step-agnostic placeholders', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 12,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });
      const secondOutputDir = join(fixture.root, 'output-steps-30');
      mkdirSync(secondOutputDir);
      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: secondOutputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });
      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.scheduler).toEqual(
        expect.objectContaining({
          family: 'flow-matching-euler-discrete',
          target: 'diffusers.FlowMatchEulerDiscreteScheduler',
          step_count: 12,
          sigma_start: expect.any(Number),
          sigma_end: expect.any(Number),
          timestep_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.denoise).toEqual(
        expect.objectContaining({
          model: 'RefineDiT',
          scheduler_signature: expect.any(Number),
          conditioning_signature: expect.any(Number),
          timestep_count: 12,
          state_hydrated: true,
          hydration: expect.objectContaining({
            module: 'RefineDiT',
            load_style: 'load_state_dict',
            strict: false,
          }),
        }),
      );
      expect(firstMetrics.scheduler).not.toEqual(secondMetrics.scheduler);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
    } finally {
      fixture.cleanup();
    }
  });

  it('hydrates checkpoint-backed modules and invokes real diffusers/cubvh seams in the supported path', () => {
    const fixture = createFixtureWorkspace();
    const diffusersTracePath = join(fixture.root, 'diffusers-trace.json');
    const cubvhTracePath = join(fixture.root, 'cubvh-trace.json');

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_DIFFUSERS_TRACE_PATH: diffusersTracePath,
            ULTRASHAPE_TEST_CUBVH_TRACE_PATH: cubvhTracePath,
            ULTRASHAPE_TEST_CUBVH_RETURNS_TENSORS: '1',
          },
        },
      );

      expect(outcome?.ok).toBe(true);
      const metrics = getRunnerMetrics(outcome);
      expect(metrics.checkpoint).toEqual(
        expect.objectContaining({
          load_style: 'load_state_dict',
          hydrated_modules: ['conditioner', 'dit', 'vae'],
        }),
      );
      expect(JSON.parse(readFileSync(diffusersTracePath, 'utf8'))).toEqual(
        expect.objectContaining({
          event: 'set_timesteps',
          step_count: 30,
        }),
      );
      expect(JSON.parse(readFileSync(cubvhTracePath, 'utf8'))).toEqual(
        expect.objectContaining({
          coords_type: 'Tensor',
          corners_type: 'Tensor',
          coords_dtype: 'int32',
          corners_dtype: 'float32',
          coords: expect.any(Number),
          corners: expect.any(Number),
          iso: 0,
          ensure_consistency: false,
        }),
      );
      const cubvhTrace = JSON.parse(readFileSync(cubvhTracePath, 'utf8')) as { coords: number; corners: number };
      expect(cubvhTrace.coords).toBeGreaterThanOrEqual(64);
      expect(cubvhTrace.corners).toBe(cubvhTrace.coords);
      expect((metrics.decode as Record<string, unknown>).cell_count).toBeGreaterThanOrEqual(64);
    } finally {
      fixture.cleanup();
    }
  });

  it('exports refined.glb from the extracted mesh geometry instead of synthetic payload bytes', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      expect(outcome?.ok).toBe(true);
      const metrics = getRunnerMetrics(outcome);
      const extractMetrics = metrics.extract as Record<string, unknown>;
      const refinedPath = join(fixture.outputDir, 'refined.glb');
      const refinedBytes = readFileSync(refinedPath);
      const refinedDocument = readGlbJson(refinedPath);
      const primitive = refinedDocument.meshes?.[0]?.primitives?.[0];
      const indexAccessorIndex = primitive?.indices;
      expect(indexAccessorIndex).toEqual(expect.any(Number));
      const indexAccessor = refinedDocument.accessors?.[indexAccessorIndex as number] as { count?: unknown } | undefined;

      expect(extractMetrics).toEqual(
        expect.objectContaining({
          vertex_count: getGlbPositionAccessorCount(refinedPath),
          face_count: Math.floor(Number(indexAccessor?.count ?? 0) / 3),
          payload_bytes: refinedBytes.length,
        }),
      );
      expect(Number(extractMetrics.vertex_count)).toBeGreaterThanOrEqual(24);
      expect(Number(extractMetrics.face_count)).toBeGreaterThanOrEqual(16);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects byte-derived refined mesh payloads instead of fabricating fallback geometry', () => {
    const fixture = createFixtureWorkspace();

    try {
      const outcome = runPythonSnippet(
        [
          'import json',
          'from ultrashape_runtime.utils.mesh import build_renderable_mesh_payload',
          'try:',
          '    build_renderable_mesh_payload({',
          '        "kind": "refined-mesh",',
          '        "mesh_name": "synthetic-refined",',
          '        "bytes": b"synthetic-refined-payload",',
          '    })',
          'except Exception as error:',
          '    print(json.dumps({"ok": False, "error": str(error)}))',
          'else:',
          '    print(json.dumps({"ok": True}))',
        ].join('\n'),
        [],
        [fixture.stubDir],
      );

      expect(outcome.status).toBe(0);
      expect(JSON.parse(outcome.stdout)).toEqual({
        ok: false,
        error: expect.stringContaining('renderable vertices and faces'),
      });
    } finally {
      fixture.cleanup();
    }
  });

  it('derives staged closure metrics from checkpoint tensor values, not subtree key presence alone', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeCheckpointBundle(fixture.checkpoint, 'a');

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      writeCheckpointBundle(fixture.checkpoint, 'b');

      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });

      const firstResult = firstOutcome?.result as { metrics?: Record<string, unknown> } | undefined;
      const secondResult = secondOutcome?.result as { metrics?: Record<string, unknown> } | undefined;
      const firstMetrics = firstResult?.metrics ?? {};
      const secondMetrics = secondResult?.metrics ?? {};

      expect(firstMetrics.conditioning).not.toEqual(secondMetrics.conditioning);
      expect(firstMetrics.denoise).not.toEqual(secondMetrics.denoise);
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
    } finally {
      fixture.cleanup();
    }
  });

  it('changes checkpoint-backed refine metrics and output bytes when only checkpoint subtree content changes', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);
      writeCheckpointBundle(fixture.checkpoint, 'a');

      const firstOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: fixture.outputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });
      writeCheckpointBundle(fixture.checkpoint, 'b');

      const secondOutputDir = join(fixture.root, 'output-checkpoint-b');
      mkdirSync(secondOutputDir);
      const secondOutcome = runLocalRunner({
        reference_image: fixture.referenceImage,
        coarse_mesh: fixture.coarseMesh,
        output_dir: secondOutputDir,
        output_format: 'glb',
        checkpoint: null,
        config_path: fixture.configPath,
        ext_dir: fixture.root,
        backend: 'local',
        steps: 30,
        guidance_scale: 5.5,
        seed: 7,
        preserve_scale: true,
      });
      const firstMetrics = getRunnerMetrics(firstOutcome);
      const secondMetrics = getRunnerMetrics(secondOutcome);

      expect(firstMetrics.conditioning).toEqual(
        expect.objectContaining({
          checkpoint_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.denoise).toEqual(
        expect.objectContaining({
          checkpoint_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.gate).toEqual(
        expect.objectContaining({
          coarse_signature: expect.any(Number),
          reference_image_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(secondMetrics.gate).toEqual(
        expect.objectContaining({
          coarse_signature: expect.any(Number),
          reference_image_signature: expect.any(Number),
          checkpoint_signature: expect.any(Number),
          attribution_signature: expect.any(Number),
        }),
      );
      expect(firstMetrics.decode).not.toEqual(secondMetrics.decode);
      expect(firstMetrics.extract).not.toEqual(secondMetrics.extract);
      expect(firstMetrics.gate).not.toEqual(secondMetrics.gate);
    } finally {
      fixture.cleanup();
    }
  });

  it('rejects aligned passthrough geometry at the vendored runner seam and does not publish refined.glb', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const outcome = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_PASSTHROUGH: '1',
          },
        },
      );

      expect(outcome).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('passthrough-like'),
      });
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it('enforces preserve-scale bbox tolerance while allowing isotropic scale fit only when preserve_scale=false', () => {
    const fixture = createFixtureWorkspace();

    try {
      writeRuntimeConfig(fixture.configPath);

      const rejected = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: true,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_SCALE_DRIFT: '1',
          },
        },
      );

      expect(rejected).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('preserve-scale bbox tolerance failed'),
      });
      expect(existsSync(join(fixture.outputDir, 'refined.glb'))).toBe(false);

      const accepted = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: fixture.outputDir,
          output_format: 'glb',
          checkpoint: null,
          config_path: fixture.configPath,
          ext_dir: fixture.root,
          backend: 'local',
          steps: 30,
          guidance_scale: 5.5,
          seed: 7,
          preserve_scale: false,
        },
        {
          env: {
            ULTRASHAPE_TEST_FORCE_SCALE_DRIFT: '1',
          },
        },
      );

      expect(accepted).toEqual({
        ok: true,
        result: {
          file_path: join(fixture.outputDir, 'refined.glb'),
          format: 'glb',
          backend: 'local',
          metrics: expect.objectContaining({
            extent_ratio: expect.any(Array),
            gate: expect.objectContaining({
              preserve_scale: false,
              scale_fit_applied: true,
              attribution_signature: expect.any(Number),
            }),
          }),
          fallbacks: ['flash_attn->sdpa'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
          runtime_contract: {
            backend: 'local-only',
            scope: 'mc-only',
            output_format: 'glb-only',
            requires_exact_closure: true,
            checkpoint_subtrees: ['vae', 'dit', 'conditioner'],
            public_error_codes: ['DEPENDENCY_MISSING', 'WEIGHTS_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE'],
          },
          warnings: [],
        },
      });
      expectRenderableGlbMesh(join(fixture.outputDir, 'refined.glb'));
    } finally {
      fixture.cleanup();
    }
  });
});
