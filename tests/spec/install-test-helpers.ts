import { cpSync, mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const INSTALL_SURFACE_PATHS = [
  'README.md',
  'manifest.json',
  'generator.py',
  'setup.py',
  'runtime/configs/infer_dit_refine.yaml',
  'runtime/vendor/ultrashape_runtime',
] as const;

const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==';

export function copyInstallSurface(targetRoot: string) {
  for (const relativePath of INSTALL_SURFACE_PATHS) {
    cpSync(path.join(repoRoot, relativePath), path.join(targetRoot, relativePath), { recursive: true });
  }
}

export function stageCheckpoint(extDir: string) {
  const checkpointPath = path.join(extDir, 'models', 'ultrashape', 'ultrashape_v1.pt');
  mkdirSync(path.dirname(checkpointPath), { recursive: true });
  writeFileSync(
    checkpointPath,
    JSON.stringify({
      vae: { tensors: { weights: [0.1, 0.2, 0.3, 0.4] } },
      dit: { tensors: { weights: [0.5, 0.6, 0.7, 0.8] } },
      conditioner: { tensors: { weights: [0.2, 0.4, 0.6, 0.8] } },
    }),
    'utf8',
  );
  return checkpointPath;
}

export function createRuntimeInputs(root: string) {
  const referenceImage = path.join(root, 'reference.png');
  const coarseMesh = path.join(root, 'coarse.glb');
  writeFileSync(referenceImage, Buffer.from(PNG_1X1_BASE64, 'base64'));
  writeFileSync(coarseMesh, createBinaryGlb());
  return { referenceImage, coarseMesh };
}

export function writeRuntimeStubModules(root: string) {
  const modules = [
    'torchvision.py',
    'numpy.py',
    'cv2.py',
    'yaml.py',
    'omegaconf.py',
    'einops.py',
    'transformers.py',
    'huggingface_hub.py',
    'accelerate.py',
    'safetensors.py',
    'tqdm.py',
    'onnxruntime.py',
    'flash_attn.py',
  ];

  mkdirSync(root, { recursive: true });
  mkdirSync(path.join(root, 'skimage'), { recursive: true });
  mkdirSync(path.join(root, 'PIL'), { recursive: true });
  writeFileSync(path.join(root, 'skimage', '__init__.py'), '', 'utf8');
  writeFileSync(path.join(root, 'PIL', '__init__.py'), '', 'utf8');

  for (const modulePath of modules) {
    writeFileSync(path.join(root, modulePath), '\n', 'utf8');
  }

  writeFileSync(path.join(root, 'rembg.py'), 'def remove(payload):\n    return payload\n', 'utf8');

  writeFileSync(
    path.join(root, 'diffusers.py'),
    `class FlowMatchEulerDiscreteScheduler:\n    def __init__(self, **config):\n        self.config = config\n        self.timesteps = []\n        self.sigmas = []\n\n    @classmethod\n    def from_config(cls, config):\n        return cls(**config)\n\n    def set_timesteps(self, step_count):\n        self.timesteps = [float(index) for index in range(step_count)]\n        if step_count <= 1:\n            self.sigmas = [1.0]\n            return\n        self.sigmas = [round(1.0 - (index / (step_count - 1)), 6) for index in range(step_count)]\n`,
    'utf8',
  );

  writeFileSync(
    path.join(root, 'torch.py'),
    `import json\n\nint32 = 'int32'\nfloat32 = 'float32'\n\nclass Tensor:\n    def __init__(self, values, dtype=None):\n        self._values = values\n        self.dtype = dtype\n        self.shape = _shape(values)\n\n    def cpu(self):\n        return self\n\n    def tolist(self):\n        return self._values\n\n    def reshape(self, *_shape_args):\n        return Tensor(_flatten(self._values), dtype=self.dtype)\n\n    def flatten(self):\n        return Tensor(_flatten(self._values), dtype=self.dtype)\n\n    def numel(self):\n        return len(_flatten(self._values))\n\n    def __getitem__(self, index):\n        return _flatten(self._values)[index]\n\n    def min(self):\n        return min(_flatten(self._values))\n\n    def max(self):\n        return max(_flatten(self._values))\n\n    def mean(self):\n        flat = _flatten(self._values)\n        return sum(flat) / len(flat) if flat else 0.0\n\ndef tensor(values, dtype=None):\n    return Tensor(values, dtype=dtype)\n\ndef load(path, map_location=None):\n    del map_location\n    with open(path, 'r', encoding='utf8') as handle:\n        return json.load(handle)\n\ndef _flatten(values):\n    if isinstance(values, (list, tuple)):\n        flattened = []\n        for value in values:\n            flattened.extend(_flatten(value))\n        return flattened\n    return [values]\n\ndef _shape(values):\n    if isinstance(values, (list, tuple)) and values:\n        return (len(values), *_shape(values[0]))\n    if isinstance(values, (list, tuple)):\n        return (0,)\n    return ()\n`,
    'utf8',
  );

  writeFileSync(
    path.join(root, 'cubvh.py'),
    `def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):\n    del coords, corners, iso, ensure_consistency\n    vertices = [\n        [0.0, 0.0, 0.0],\n        [1.0, 0.0, 0.0],\n        [0.0, 1.0, 0.0],\n        [0.0, 0.0, 1.0],\n        [1.0, 1.0, 0.0],\n        [1.0, 0.0, 1.0],\n        [0.0, 1.0, 1.0],\n        [1.0, 1.0, 1.0],\n        [0.5, 0.5, 1.02],\n    ]\n    faces = [\n        [0, 1, 4],\n        [0, 4, 2],\n        [0, 1, 5],\n        [0, 5, 3],\n        [2, 4, 7],\n        [2, 7, 6],\n        [3, 5, 8],\n        [3, 8, 6],\n    ]\n    return vertices, faces\n`,
    'utf8',
  );

  writeFileSync(
    path.join(root, 'trimesh.py'),
    `import json\nimport struct\n\nclass Trimesh:\n    def __init__(self, vertices, faces, process=False):\n        del process\n        self.vertices = vertices\n        self.faces = faces\n\nclass Scene:\n    def __init__(self):\n        self.mesh = None\n\n    def add_geometry(self, mesh, node_name=None):\n        del node_name\n        self.mesh = mesh\n\n    def export(self, file_type='glb'):\n        if file_type != 'glb':\n            raise ValueError('Only glb export is supported in tests.')\n        if self.mesh is None:\n            raise ValueError('Scene has no mesh.')\n        return _build_glb(self.mesh.vertices, self.mesh.faces)\n\ndef _build_glb(vertices, faces):\n    document = {\n        'asset': {'version': '2.0'},\n        'scenes': [{'nodes': [0]}],\n        'scene': 0,\n        'nodes': [{'mesh': 0}],\n        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],\n        'accessors': [\n            {'bufferView': 0, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'},\n            {'bufferView': 1, 'componentType': 5125, 'count': len(faces) * 3, 'type': 'SCALAR'},\n        ],\n        'bufferViews': [\n            {'buffer': 0, 'byteOffset': 0, 'byteLength': len(vertices) * 12},\n            {'buffer': 0, 'byteOffset': len(vertices) * 12, 'byteLength': len(faces) * 12},\n        ],\n        'buffers': [{'byteLength': (len(vertices) * 12) + (len(faces) * 12)}],\n    }\n    json_chunk = json.dumps(document).encode('utf8')\n    json_chunk += b' ' * ((4 - (len(json_chunk) % 4)) % 4)\n    binary = bytearray()\n    for vertex in vertices:\n        binary.extend(struct.pack('<3f', float(vertex[0]), float(vertex[1]), float(vertex[2])))\n    for face in faces:\n        binary.extend(struct.pack('<3I', int(face[0]), int(face[1]), int(face[2])))\n    header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_chunk) + 8 + len(binary))\n    json_header = struct.pack('<II', len(json_chunk), 0x4E4F534A)\n    bin_header = struct.pack('<II', len(binary), 0x004E4942)\n    return header + json_header + json_chunk + bin_header + bytes(binary)\n`,
    'utf8',
  );
}

type GeneratorProbeAction =
  | { method: 'is_downloaded' }
  | { method: 'load' }
  | { method: 'unload' }
  | {
      method: 'generate';
      imageBase64?: string;
      params?: Record<string, unknown>;
    };

export function runGeneratorProbe(
  checkout: string,
  actions: GeneratorProbeAction[],
  env: NodeJS.ProcessEnv = {},
) {
  const script = [
    'import base64, json, sys',
    'from pathlib import Path',
    'from generator import UltraShapeGenerator',
    'generator = UltraShapeGenerator(Path.cwd() / "models", Path.cwd() / "outputs")',
    'actions = json.loads(sys.argv[1])',
    'results = []',
    'for action in actions:',
    '    method = action["method"]',
    '    try:',
    '        if method == "is_downloaded":',
    '            value = generator.is_downloaded()',
    '        elif method == "load":',
    '            value = generator.load()',
    '        elif method == "unload":',
    '            value = generator.unload()',
    '        elif method == "generate":',
    '            image_base64 = action.get("imageBase64")',
    '            image_bytes = base64.b64decode(image_base64) if isinstance(image_base64, str) else None',
    '            value = generator.generate(image_bytes, action.get("params") or {})',
    '        else:',
    '            raise ValueError(f"Unsupported probe method: {method}")',
    '        item = {"method": method, "ok": True, "result": value, "loaded": getattr(generator, "_loaded", None)}',
    '        if method == "generate":',
    '            item["debug"] = {"last_job": getattr(generator, "_last_job", None), "last_pythonpath": getattr(generator, "_last_pythonpath", None), "last_result": getattr(generator, "_last_result", None)}',
    '        results.append(item)',
    '    except Exception as error:',
    '        results.append({',
    '            "method": method,',
    '            "ok": False,',
    '            "error": {',
    '                "type": error.__class__.__name__,',
    '                "code": getattr(error, "code", None),',
    '                "message": str(error),',
    '            },',
    '            "loaded": getattr(generator, "_loaded", None),',
    '        })',
    'print(json.dumps(results))',
  ].join('\n');

  return spawnSync('python3', ['-S', '-c', script, JSON.stringify(actions)], {
    cwd: checkout,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env,
    },
  });
}

function createBinaryGlb() {
  const jsonChunk = Buffer.from(
    JSON.stringify({
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' },
        { bufferView: 1, componentType: 5125, count: 6, type: 'SCALAR' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 48 },
        { buffer: 0, byteOffset: 48, byteLength: 24 },
      ],
      buffers: [{ byteLength: 72 }],
    }),
    'utf8',
  );
  const paddedJson = Buffer.concat([jsonChunk, Buffer.alloc((4 - (jsonChunk.length % 4)) % 4, 0x20)]);
  const binaryChunk = Buffer.alloc(72);
  const vertices = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const faces = [0, 1, 2, 0, 1, 3];
  vertices.forEach((vertex, vertexIndex) => {
    vertex.forEach((axis, axisIndex) => {
      binaryChunk.writeFloatLE(axis, vertexIndex * 12 + axisIndex * 4);
    });
  });
  faces.forEach((index, faceIndex) => {
    binaryChunk.writeUInt32LE(index, 48 + faceIndex * 4);
  });
  const totalLength = 12 + 8 + paddedJson.length + 8 + binaryChunk.length;
  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(totalLength, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(paddedJson.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binaryChunk.length, 0);
  binHeader.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, jsonHeader, paddedJson, binHeader, binaryChunk]);
}
