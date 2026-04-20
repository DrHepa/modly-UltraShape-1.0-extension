import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const configPath = path.join(repoRoot, 'runtime/configs/infer_dit_refine.yaml');
const runtimeVendorPath = path.join(repoRoot, 'runtime/vendor');
const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==';

function createBinaryGlb() {
  const jsonChunk = Buffer.from(
    JSON.stringify({
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [
        {
          primitives: [
            {
              attributes: { POSITION: 0 },
              indices: 1,
            },
          ],
        },
      ],
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

function writeRuntimeStubModules(root: string) {
  const modules = [
    'torchvision.py',
    'cv2.py',
    'omegaconf.py',
    'einops.py',
    'transformers.py',
    'accelerate.py',
    'safetensors.py',
  ];

  mkdirSync(path.join(root, 'skimage'), { recursive: true });
  writeFileSync(path.join(root, 'skimage', '__init__.py'), '');
  modules.forEach((modulePath) => writeFileSync(path.join(root, modulePath), '\n', 'utf8'));

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

function createRuntimeFixture() {
  const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-runtime-flow-'));
  const stubRoot = path.join(sandbox, 'stubs');
  const extDir = path.join(sandbox, 'ext');
  const modelsDir = path.join(extDir, 'models', 'ultrashape');
  mkdirSync(stubRoot, { recursive: true });
  mkdirSync(modelsDir, { recursive: true });
  writeRuntimeStubModules(stubRoot);

  const referenceImage = path.join(sandbox, 'reference.png');
  const coarseMesh = path.join(sandbox, 'coarse.glb');
  const checkpoint = path.join(modelsDir, 'ultrashape_v1.pt');
  writeFileSync(referenceImage, Buffer.from(PNG_1X1_BASE64, 'base64'));
  writeFileSync(coarseMesh, createBinaryGlb());
  writeFileSync(
    checkpoint,
    JSON.stringify({
      vae: { tensors: { weights: [0.1, 0.2, 0.3, 0.4] } },
      dit: { tensors: { weights: [0.5, 0.6, 0.7, 0.8] } },
      conditioner: { tensors: { weights: [0.2, 0.4, 0.6, 0.8] } },
    }),
    'utf8',
  );

  return { sandbox, stubRoot, extDir, referenceImage, coarseMesh, checkpoint };
}

function runPythonSnippet(source: string, args: string[] = []) {
  return spawnSync('python3', ['-c', source, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONPATH: [runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
    },
  });
}

function runLocalRunner(job: Record<string, unknown>, stubRoot: string) {
  return spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
    cwd: repoRoot,
    encoding: 'utf8',
    input: JSON.stringify(job),
    env: {
      ...process.env,
      PYTHONPATH: [stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
    },
  });
}

describe('runtime closure authority', () => {
  it('publishes upstream-style stage evidence from the vendored closure', () => {
    const result = runPythonSnippet(
      [
        'import json, sys',
        'from ultrashape_runtime import CHECKPOINT_REQUIRED_SUBTREES, RUNTIME_LAYOUT, RUNTIME_SCOPE, UPSTREAM_CLOSURE_READY',
        'from ultrashape_runtime.local_runner import PUBLIC_ERROR_CODES',
        'from ultrashape_runtime.pipelines import build_refine_pipeline, load_runtime_config',
        'from ultrashape_runtime.preprocessors import ImageProcessorV2',
        'from ultrashape_runtime.surface_loaders import SharpEdgeSurfaceLoader',
        'from ultrashape_runtime.schedulers import default_scheduler_name',
        'from ultrashape_runtime.models.conditioner_mask import SingleImageEncoder',
        'from ultrashape_runtime.models.denoisers.dit_mask import RefineDiT',
        'from ultrashape_runtime.models.autoencoders.model import ShapeVAE',
        'config = load_runtime_config(sys.argv[1])',
        'payload = {',
        '  "runtime": {',
        '    "scope": RUNTIME_SCOPE,',
        '    "layout": RUNTIME_LAYOUT,',
        '    "ready": UPSTREAM_CLOSURE_READY,',
        '    "checkpoint_required_subtrees": list(CHECKPOINT_REQUIRED_SUBTREES),',
        '    "public_error_codes": sorted(PUBLIC_ERROR_CODES),',
        '  },',
        '  "config": {',
        '    "scope": config["model"]["scope"],',
        '    "requires_exact_closure": config["runtime"]["requires_exact_closure"],',
        '    "scheduler_target": config["scheduler"]["target"],',
        '    "conditioner_target": config["conditioner_config"]["target"],',
        '    "dit_target": config["dit_cfg"]["target"],',
        '    "vae_target": config["vae_config"]["target"],',
        '    "surface_extraction": config["surface"]["extraction"],',
        '    "export_format": config["export"]["format"],',
        '  },',
        '  "stages": {',
        '    "pipeline": build_refine_pipeline(),',
        '    "preprocess": {"class": ImageProcessorV2.__name__, "method": "process"},',
        '    "conditioning": {"class": SingleImageEncoder.__name__, "method": "build"},',
        '    "surface": {"class": SharpEdgeSurfaceLoader.__name__, "method": "load"},',
        '    "scheduler": {"family": default_scheduler_name()},',
        '    "denoise": {"class": RefineDiT.__name__, "method": "denoise"},',
        '    "decode": {"class": ShapeVAE.__name__, "method": "decode_latents"},',
        '  },',
        '}',
        'print(json.dumps(payload))',
      ].join('\n'),
      [configPath],
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      runtime: {
        scope: 'mc-only',
        layout: 'vendored-upstream-closure',
        ready: true,
        checkpoint_required_subtrees: ['vae', 'dit', 'conditioner'],
        public_error_codes: ['DEPENDENCY_MISSING', 'LOCAL_RUNTIME_UNAVAILABLE', 'WEIGHTS_MISSING'],
      },
      config: {
        scope: 'mc-only',
        requires_exact_closure: true,
        scheduler_target: 'diffusers.FlowMatchEulerDiscreteScheduler',
        conditioner_target: 'ultrashape_runtime.models.conditioner_mask.SingleImageEncoder',
        dit_target: 'ultrashape_runtime.models.denoisers.dit_mask.RefineDiT',
        vae_target: 'ultrashape_runtime.models.autoencoders.model.ShapeVAE',
        surface_extraction: 'mc',
        export_format: 'glb',
      },
      stages: {
        pipeline: { name: 'ultrashape-refine', scope: 'mc-only' },
        preprocess: { class: 'ImageProcessorV2', method: 'process' },
        conditioning: { class: 'SingleImageEncoder', method: 'build' },
        surface: { class: 'SharpEdgeSurfaceLoader', method: 'load' },
        scheduler: { family: 'flow-matching-euler-discrete' },
        denoise: { class: 'RefineDiT', method: 'denoise' },
        decode: { class: 'ShapeVAE', method: 'decode_latents' },
      },
    });
  });

  it('rejects shorthand closure configs instead of re-authorizing synthetic runtime allowances', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-runtime-config-'));
    const shorthandConfigPath = path.join(sandbox, 'shorthand.yaml');
    writeFileSync(
      shorthandConfigPath,
      ['model:', '  scope: mc-only', 'runtime:', '  backend: local', '  requires_exact_closure: false'].join('\n'),
      'utf8',
    );

    const result = runPythonSnippet(
      [
        'import json, sys',
        'from ultrashape_runtime.pipelines import load_runtime_config',
        'try:',
        '    load_runtime_config(sys.argv[1])',
        'except Exception as error:',
        '    print(json.dumps({"ok": False, "code": getattr(error, "code", None), "message": str(error)}))',
        'else:',
        '    print(json.dumps({"ok": True}))',
      ].join('\n'),
      [shorthandConfigPath],
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      ok: false,
      code: 'LOCAL_RUNTIME_UNAVAILABLE',
      message: 'Runtime config requires_exact_closure: true for the upstream closure path.',
    });
  });

  it('executes the vendored local runner and writes output_dir/refined.glb', () => {
    const fixture = createRuntimeFixture();
    const outputDir = path.join(fixture.sandbox, 'output');

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: outputDir,
          output_format: 'glb',
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          backend: 'local',
          format: 'glb',
          file_path: path.join(outputDir, 'refined.glb'),
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
        },
      });
      expect(existsSync(path.join(outputDir, 'refined.glb'))).toBe(true);
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('maps unreadable checkpoints to WEIGHTS_MISSING through the real runner path', () => {
    const fixture = createRuntimeFixture();

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.referenceImage,
          coarse_mesh: fixture.coarseMesh,
          output_dir: path.join(fixture.sandbox, 'output'),
          output_format: 'glb',
          checkpoint: path.join(fixture.sandbox, 'missing.pt'),
          config_path: configPath,
          ext_dir: fixture.extDir,
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
      );

      expect(result.status).toBe(1);
      expect(JSON.parse(result.stdout)).toEqual({
        ok: false,
        error_code: 'WEIGHTS_MISSING',
        error_message: `Required checkpoint is not readable: ${path.join(fixture.sandbox, 'missing.pt')}.`,
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });
});
