import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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

function readGlbVertices(glbPath: string) {
  const payload = readFileSync(glbPath);
  const jsonChunkLength = payload.readUInt32LE(12);
  const jsonChunk = payload.subarray(20, 20 + jsonChunkLength);
  const document = JSON.parse(jsonChunk.toString('utf8').trimEnd()) as {
    accessors: Array<{ bufferView: number; count: number; byteOffset?: number }>;
    bufferViews: Array<{ byteOffset?: number; byteStride?: number }>;
    meshes: Array<{ primitives: Array<{ attributes: { POSITION: number } }> }>;
  };
  const binaryOffset = 20 + jsonChunkLength + 8;
  const positionAccessorIndex = document.meshes[0]?.primitives[0]?.attributes.POSITION;
  const accessor = document.accessors[positionAccessorIndex];
  const bufferView = document.bufferViews[accessor.bufferView];
  const byteOffset = binaryOffset + (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = bufferView.byteStride ?? 12;
  const vertices: Array<[number, number, number]> = [];

  for (let index = 0; index < accessor.count; index += 1) {
    const start = byteOffset + index * stride;
    vertices.push([
      payload.readFloatLE(start),
      payload.readFloatLE(start + 4),
      payload.readFloatLE(start + 8),
    ]);
  }

  return vertices;
}

function meshExtents(vertices: Array<[number, number, number]>) {
  const xs = vertices.map((vertex) => vertex[0]);
  const ys = vertices.map((vertex) => vertex[1]);
  const zs = vertices.map((vertex) => vertex[2]);
  return [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), Math.max(...zs) - Math.min(...zs)];
}

const COHESIVE_CUBVH_STUB = `def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):\n    del coords, corners, iso, ensure_consistency\n    vertices = [\n        [0.0, 0.0, 0.0],\n        [1.0, 0.0, 0.0],\n        [1.0, 1.0, 0.0],\n        [0.0, 1.0, 0.0],\n        [0.0, 0.0, 1.0],\n        [1.0, 0.0, 1.0],\n        [1.0, 1.0, 1.0],\n        [0.0, 1.0, 1.0],\n        [0.5, 0.5, 0.5],\n    ]\n    faces = [\n        [0, 2, 1], [0, 3, 2],\n        [4, 5, 6], [4, 6, 7],\n        [0, 1, 5], [0, 5, 4],\n        [1, 2, 6], [1, 6, 5],\n        [2, 3, 7], [2, 7, 6],\n        [3, 0, 4], [3, 4, 7],\n    ]\n    return vertices, faces\n`;

const LAMELLAR_CUBVH_STUB = `def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):\n    del coords, corners, iso, ensure_consistency\n    vertices = []\n    faces = []\n    for z in [0.0, 0.5, 1.0]:\n        base = len(vertices)\n        vertices.extend([\n            [0.0, 0.0, z],\n            [1.0, 0.0, z],\n            [1.0, 1.0, z],\n            [0.0, 1.0, z],\n            [0.5, 0.5, z],\n        ])\n        faces.extend([\n            [base + 0, base + 1, base + 4],\n            [base + 1, base + 2, base + 4],\n            [base + 2, base + 3, base + 4],\n            [base + 3, base + 0, base + 4],\n        ])\n    return vertices, faces\n`;

function writeCubvhStub(root: string, source: string) {
  writeFileSync(path.join(root, 'cubvh.py'), source, 'utf8');
}

function writeRuntimeStubModules(root: string, options: { cubvhSource?: string } = {}) {
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

  writeCubvhStub(root, options.cubvhSource ?? COHESIVE_CUBVH_STUB);

  writeFileSync(
    path.join(root, 'trimesh.py'),
    `import json\nimport struct\n\nclass Trimesh:\n    def __init__(self, vertices, faces, process=False):\n        del process\n        self.vertices = vertices\n        self.faces = faces\n\nclass Scene:\n    def __init__(self):\n        self.mesh = None\n\n    def add_geometry(self, mesh, node_name=None):\n        del node_name\n        self.mesh = mesh\n\n    def export(self, file_type='glb'):\n        if file_type != 'glb':\n            raise ValueError('Only glb export is supported in tests.')\n        if self.mesh is None:\n            raise ValueError('Scene has no mesh.')\n        return _build_glb(self.mesh.vertices, self.mesh.faces)\n\ndef _build_glb(vertices, faces):\n    document = {\n        'asset': {'version': '2.0'},\n        'scenes': [{'nodes': [0]}],\n        'scene': 0,\n        'nodes': [{'mesh': 0}],\n        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],\n        'accessors': [\n            {'bufferView': 0, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'},\n            {'bufferView': 1, 'componentType': 5125, 'count': len(faces) * 3, 'type': 'SCALAR'},\n        ],\n        'bufferViews': [\n            {'buffer': 0, 'byteOffset': 0, 'byteLength': len(vertices) * 12},\n            {'buffer': 0, 'byteOffset': len(vertices) * 12, 'byteLength': len(faces) * 12},\n        ],\n        'buffers': [{'byteLength': (len(vertices) * 12) + (len(faces) * 12)}],\n    }\n    json_chunk = json.dumps(document).encode('utf8')\n    json_chunk += b' ' * ((4 - (len(json_chunk) % 4)) % 4)\n    binary = bytearray()\n    for vertex in vertices:\n        binary.extend(struct.pack('<3f', float(vertex[0]), float(vertex[1]), float(vertex[2])))\n    for face in faces:\n        binary.extend(struct.pack('<3I', int(face[0]), int(face[1]), int(face[2])))\n    header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_chunk) + 8 + len(binary))\n    json_header = struct.pack('<II', len(json_chunk), 0x4E4F534A)\n    bin_header = struct.pack('<II', len(binary), 0x004E4942)\n    return header + json_header + json_chunk + bin_header + bytes(binary)\n`,
    'utf8',
  );
}

function createRuntimeFixture(options: { cubvhSource?: string } = {}) {
  const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-runtime-flow-'));
  const stubRoot = path.join(sandbox, 'stubs');
  const extDir = path.join(sandbox, 'ext');
  const modelsDir = path.join(extDir, 'models', 'ultrashape');
  mkdirSync(stubRoot, { recursive: true });
  mkdirSync(modelsDir, { recursive: true });
  writeRuntimeStubModules(stubRoot, options);

  const imageInputPath = path.join(sandbox, 'reference.png');
  const meshInputPath = path.join(sandbox, 'coarse.glb');
  const checkpoint = path.join(modelsDir, 'ultrashape_v1.pt');
  writeFileSync(imageInputPath, Buffer.from(PNG_1X1_BASE64, 'base64'));
  writeFileSync(meshInputPath, createBinaryGlb());
  writeFileSync(
    checkpoint,
    JSON.stringify({
      vae: {
        state_dict: {
          'post_kl.weight': [0.11, 0.12, 0.13, 0.14],
          'post_kl.bias': [0.21, 0.22, 0.23, 0.24],
          'transformer.resblocks.0.attn.c_qkv.weight': [0.31, 0.32, 0.33, 0.34],
          'transformer.resblocks.0.attn.c_proj.bias': [0.41, 0.42, 0.43, 0.44],
          'geo_decoder.query_proj.weight': [0.51, 0.52, 0.53, 0.54],
        },
      },
      dit: {
        state_dict: {
          'x_embedder.weight': [0.15, 0.25, 0.35, 0.45],
          'x_embedder.bias': [0.55, 0.65, 0.75, 0.85],
          't_embedder.mlp.0.weight': [0.19, 0.29, 0.39, 0.49],
          't_embedder.mlp.0.bias': [0.59, 0.69, 0.79, 0.89],
          't_embedder.mlp.2.weight': [0.14, 0.24, 0.34, 0.44],
          'final_layer.linear.weight': [0.54, 0.64, 0.74, 0.84],
          'final_layer.linear.bias': [0.18, 0.28, 0.38, 0.48],
        },
      },
      conditioner: {
        state_dict: {
          'main_image_encoder.model.embeddings.patch_embedding.weight': [0.12, 0.24, 0.36, 0.48],
          'main_image_encoder.model.embeddings.patch_embedding.bias': [0.16, 0.26, 0.36, 0.46],
          'main_image_encoder.model.post_layernorm.weight': [0.22, 0.32, 0.42, 0.52],
          'main_image_encoder.model.post_layernorm.bias': [0.28, 0.38, 0.48, 0.58],
        },
      },
    }),
    'utf8',
  );

  return { sandbox, stubRoot, extDir, imageInputPath, meshInputPath, checkpoint };
}

function createFakeUpstreamCheckout(
  root: string,
  options: { writesOutput?: boolean; checkoutName?: string; checkoutMarker?: string } = {},
) {
  const checkout = path.join(root, options.checkoutName ?? 'fake-ultrashape-checkout');
  const scriptsDir = path.join(checkout, 'scripts');
  const configsDir = path.join(checkout, 'configs');
  const packageDir = path.join(checkout, 'ultrashape');
  mkdirSync(scriptsDir, { recursive: true });
  mkdirSync(configsDir, { recursive: true });
  mkdirSync(packageDir, { recursive: true });
  writeFileSync(path.join(checkout, 'LICENSE'), 'Fake UltraShape checkout for contract tests.\n', 'utf8');
  writeFileSync(path.join(scriptsDir, '__init__.py'), '', 'utf8');
  writeFileSync(path.join(packageDir, '__init__.py'), `CHECKOUT_MARKER = ${JSON.stringify(options.checkoutMarker ?? 'default')}\n`, 'utf8');
  writeFileSync(
    path.join(configsDir, 'infer_dit_refine.yaml'),
    'model:\n  params:\n    upstream_schema_marker: fake-ultrashape-checkout\n',
    'utf8',
  );
  writeFileSync(
    path.join(scriptsDir, 'infer_dit_refine.py'),
    [
      'import json',
      'from pathlib import Path',
      'from ultrashape import CHECKOUT_MARKER',
      'def run_inference(args):',
      '    expected_config = Path(__file__).resolve().parents[1] / "configs" / "infer_dit_refine.yaml"',
      '    if Path(args.config).resolve() != expected_config.resolve():',
      '        raise AssertionError(f"upstream args.config must point at checkout config, got {args.config}")',
      '    config_text = expected_config.read_text(encoding="utf8")',
      '    if "upstream_schema_marker" not in config_text:',
      '        raise AssertionError("upstream config fixture must expose an upstream-compatible schema marker")',
      '    record = Path(args.output_dir) / "entrypoint-called.json"',
      '    required = {',
      '        "image": args.image,',
      '        "mesh": args.mesh,',
      '        "ckpt": args.ckpt,',
      '        "config": args.config,',
      '        "output_dir": args.output_dir,',
      '        "steps": args.steps,',
      '        "scale": args.scale,',
      '        "num_latents": args.num_latents,',
      '        "chunk_size": args.chunk_size,',
      '        "octree_res": args.octree_res,',
      '        "seed": args.seed,',
      '        "remove_bg": args.remove_bg,',
      '        "low_vram": args.low_vram,',
      '        "checkout_marker": CHECKOUT_MARKER,',
      '    }',
      '    record.write_text(json.dumps(required, sort_keys=True), encoding="utf8")',
      options.writesOutput === false
        ? '    return {"ok": True, "output_written": False}'
        : '    (Path(args.output_dir) / f"{Path(args.image).stem}_refined.glb").write_bytes(record.read_bytes())',
      '    return {"ok": True, "output_written": True}',
    ].join('\n'),
    'utf8',
  );

  return checkout;
}

function writeRealDependencyStubs(root: string) {
  mkdirSync(path.join(root, 'flash_attn'), { recursive: true });
  writeFileSync(path.join(root, 'flash_attn', '__init__.py'), '__version__ = "0.0-test"\n', 'utf8');
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

function runLocalRunner(job: Record<string, unknown>, stubRoot: string, env: NodeJS.ProcessEnv = {}) {
  return spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
    cwd: repoRoot,
    encoding: 'utf8',
    input: JSON.stringify(job),
    env: {
      ...process.env,
      ...env,
      PYTHONPATH: [stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
    },
  });
}

describe('private runtime flow behind the model shell', () => {
  it('adapts generator state into the private local runner payload through production code', () => {
    const result = runPythonSnippet(
      [
        'import json, pathlib, tempfile',
        'from generator import UltraShapeGenerator',
        'generator = UltraShapeGenerator(pathlib.Path.cwd() / "models", pathlib.Path.cwd() / "outputs")',
        'output_dir = pathlib.Path(tempfile.mkdtemp(prefix="ultrashape-private-runner-output-"))',
        'job = generator._build_runner_job(',
        '    readiness={"checkpoint": "/tmp/ultrashape_v1.pt", "config_path": "", "ext_dir": "/tmp/ext"},',
        '    reference_image=pathlib.Path("/tmp/reference.png"),',
        '    coarse_mesh=pathlib.Path("/tmp/coarse.glb"),',
        '    output_dir=output_dir,',
        '    params={"steps": 4, "guidance_scale": 6, "seed": 7, "preserve_scale": True},',
        ')',
        'print(json.dumps(job))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      reference_image: '/tmp/reference.png',
      coarse_mesh: '/tmp/coarse.glb',
      output_dir: expect.stringContaining('ultrashape-private-runner-output-'),
      output_format: 'glb',
      checkpoint: '/tmp/ultrashape_v1.pt',
      config_path: configPath,
      ext_dir: '/tmp/ext',
      backend: 'local',
      steps: 4,
      guidance_scale: 6,
      seed: 7,
      preserve_scale: true,
    });
  });

  it('publishes upstream-style stage evidence from the vendored closure', () => {
    const result = runPythonSnippet(
      [
        'import json, sys',
        'from ultrashape_runtime import CHECKPOINT_REQUIRED_SUBTREES, DEFAULT_RUNTIME_MODE, RUNTIME_LAYOUT, RUNTIME_MODE_STRATEGY, RUNTIME_SCOPE, SUPPORTED_RUNTIME_MODES, UPSTREAM_CLOSURE_READY, UPSTREAM_CLOSURE_REASON',
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
        '    "mode_strategy": RUNTIME_MODE_STRATEGY,',
        '    "default_mode": DEFAULT_RUNTIME_MODE,',
        '    "supported_modes": list(SUPPORTED_RUNTIME_MODES),',
        '    "real_closure_ready": UPSTREAM_CLOSURE_READY,',
        '    "real_closure_reason": UPSTREAM_CLOSURE_REASON,',
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
        layout: 'vendored-dual-mode-closure',
        mode_strategy: 'explicit-dual-mode',
        default_mode: 'auto',
        supported_modes: ['auto', 'real', 'portable'],
        real_closure_ready: false,
        real_closure_reason:
          'Authoritative real mode is optional and remains unavailable until the exact upstream torch module graph adapter is vendored and the required runtime dependencies are present.',
        checkpoint_required_subtrees: ['vae', 'dit', 'conditioner'],
        public_error_codes: ['DEPENDENCY_MISSING', 'INVALID_INPUT', 'LOCAL_RUNTIME_UNAVAILABLE', 'WEIGHTS_MISSING'],
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
        pipeline: {
          name: 'ultrashape-refine',
          scope: 'mc-only',
          entrypoint: 'scripts.infer_dit_refine.run_inference',
          loader: 'load_models',
          class: 'UltraShapePipeline',
        },
        preprocess: { class: 'ImageProcessorV2', method: 'process' },
        conditioning: { class: 'SingleImageEncoder', method: 'build' },
        surface: { class: 'SharpEdgeSurfaceLoader', method: 'load' },
        scheduler: { family: 'flow-matching-euler-discrete' },
        denoise: { class: 'RefineDiT', method: 'denoise' },
        decode: { class: 'ShapeVAE', method: 'decode_latents' },
      },
    });
  });

  it('exposes upstream closure markers for OmegaConf loading, strict hydration, and preserve_scale restoration', () => {
    const result = runPythonSnippet(
      [
        'import json, sys',
        'from ultrashape_runtime.pipelines import describe_upstream_closure_markers',
        'print(json.dumps(describe_upstream_closure_markers(sys.argv[1])))',
      ].join('\n'),
      [configPath],
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      config_loader: 'OmegaConf.load',
      checkpoint_hydration: 'strict',
      preserve_scale_restore: 'coarse-normalization-inverse',
      required_subtrees: ['vae', 'dit', 'conditioner'],
    });
  });

  it('accepts upstream RefineDiT block parameters during strict portable hydration while rejecting unknown roots', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.denoisers.dit_mask import RefineDiT',
        'accepted = RefineDiT()',
        'state_dict = {',
        '    "x_embedder.weight": [0.1, 0.2],',
        '    "t_embedder.mlp.0.weight": [0.3, 0.4],',
        '    "final_layer.linear.weight": [0.5, 0.6],',
        '    "blocks.0.norm1.weight": [0.7, 0.8],',
        '    "blocks.0.attn.qkv.weight": [0.9, 1.0],',
        '}',
        'accepted_result = accepted.load_state_dict(state_dict, strict=True)',
        'try:',
        '    RefineDiT().load_state_dict({**state_dict, "unknown_root.weight": [0.11]}, strict=True)',
        'except ValueError as error:',
        '    rejected = {"message": str(error)}',
        'else:',
        '    rejected = None',
        'print(json.dumps({',
        '    "accepted": accepted_result,',
        '    "hydrated_keys": sorted(accepted.state_dict.keys()),',
        '    "module_roots": accepted.checkpoint_state["state_dict_metadata"]["module_roots"],',
        '    "rejected": rejected,',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      accepted: { missing_keys: [], unexpected_keys: [], strict: true },
      hydrated_keys: [
        'blocks.0.attn.qkv.weight',
        'blocks.0.norm1.weight',
        'final_layer.linear.weight',
        't_embedder.mlp.0.weight',
        'x_embedder.weight',
      ],
      module_roots: ['blocks', 'final_layer', 't_embedder', 'x_embedder'],
      rejected: {
        message:
          "RefineDiT strict hydration requires upstream module-family keys for ('x_embedder', 't_embedder', 'final_layer', 'blocks'); missing=[], unexpected=['unknown_root.weight'].",
      },
    });
  });

  it('accepts upstream ShapeVAE encoder and pre_kl parameters during strict portable hydration while rejecting unknown roots', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.autoencoders.model import ShapeVAE',
        'accepted = ShapeVAE()',
        'state_dict = {',
        '    "encoder.input_proj_q.weight": [0.1, 0.2],',
        '    "encoder.input_proj_v.bias": [0.3, 0.4],',
        '    "pre_kl.weight": [0.5, 0.6],',
        '    "pre_kl.bias": [0.7, 0.8],',
        '    "post_kl.weight": [0.11, 0.12],',
        '    "transformer.resblocks.0.attn.c_qkv.weight": [0.21, 0.22],',
        '    "geo_decoder.query_proj.weight": [0.31, 0.32],',
        '}',
        'accepted_result = accepted.load_state_dict(state_dict, strict=True)',
        'try:',
        '    ShapeVAE().load_state_dict({**state_dict, "unknown_root.weight": [0.41]}, strict=True)',
        'except ValueError as error:',
        '    rejected = {"message": str(error)}',
        'else:',
        '    rejected = None',
        'print(json.dumps({',
        '    "accepted": accepted_result,',
        '    "hydrated_keys": sorted(accepted.state_dict.keys()),',
        '    "module_roots": accepted.checkpoint_state["state_dict_metadata"]["module_roots"],',
        '    "rejected": rejected,',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      accepted: { missing_keys: [], unexpected_keys: [], strict: true },
      hydrated_keys: [
        'encoder.input_proj_q.weight',
        'encoder.input_proj_v.bias',
        'geo_decoder.query_proj.weight',
        'post_kl.weight',
        'pre_kl.bias',
        'pre_kl.weight',
        'transformer.resblocks.0.attn.c_qkv.weight',
      ],
      module_roots: ['encoder', 'geo_decoder', 'post_kl', 'pre_kl', 'transformer'],
      rejected: {
        message:
          "ShapeVAE strict hydration requires upstream module-family keys for ('post_kl', 'transformer', 'geo_decoder', 'encoder', 'pre_kl'); missing=[], unexpected=['unknown_root.weight'].",
      },
    });
  });

  it('rejects shorthand closure configs instead of re-authorizing non-upstream runtime allowances', () => {
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
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
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
          metrics: {
            pipeline: {
              entrypoint: 'scripts.infer_dit_refine.run_inference',
              loader: 'load_models',
              class: 'UltraShapePipeline',
              returns_latents: true,
            },
            runtime_mode: {
              selection: 'portable-only',
              requested: 'auto',
              active: 'portable',
              real: {
                available: false,
                adapter: 'ultrashape_runtime.real_mode.run_real_refine_pipeline',
              },
              portable: {
                available: true,
              },
            },
            checkpoint: {
              strict: true,
              representation: 'module-state-dict-v2',
              hydrated_modules: ['conditioner', 'dit', 'vae'],
              hydration: [
                {
                  module: 'SingleImageEncoder',
                  module_family: 'SingleImageEncoder',
                  parameter_count: 4,
                  module_roots: ['main_image_encoder'],
                },
                {
                  module: 'RefineDiT',
                  module_family: 'RefineDiT',
                  parameter_count: 7,
                  module_roots: ['final_layer', 't_embedder', 'x_embedder'],
                },
                {
                  module: 'ShapeVAE',
                  module_family: 'ShapeVAE',
                  parameter_count: 5,
                  module_roots: ['geo_decoder', 'post_kl', 'transformer'],
                },
              ],
            },
            gate: {
              preserve_scale: true,
              portable_quality: {
                passed: true,
                reason: 'ok',
                component_count: 1,
                boundary_edge_count: 0,
                non_manifold_edge_count: 0,
              },
            },
            conditioning: {
              hydration: {
                module_family: 'SingleImageEncoder',
                parameter_count: 4,
              },
            },
            denoise: {
              hydration: {
                module_family: 'RefineDiT',
                parameter_count: 7,
              },
            },
            decode: {
              state_hydrated: true,
            },
          },
          warnings: ['PORTABLE_FALLBACK_NON_AUTHORITATIVE'],
          subtrees_loaded: ['vae', 'dit', 'conditioner'],
        },
      });
      expect(existsSync(path.join(outputDir, 'refined.glb'))).toBe(true);
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('rejects lamellar portable output before exporting output_dir/refined.glb', () => {
    const fixture = createRuntimeFixture({ cubvhSource: LAMELLAR_CUBVH_STUB });
    const outputDir = path.join(fixture.sandbox, 'output-lamellar');

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
        { ULTRASHAPE_RUNTIME_MODE: 'portable' },
      );

      expect(result.status).toBe(1);
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: expect.stringContaining('portable lamellar geometry rejected'),
      });
      expect(JSON.parse(result.stdout).error_message).toContain('boundary_edge_count=');
      expect(JSON.parse(result.stdout).error_message).toContain('component_count=');
      expect(existsSync(path.join(outputDir, 'refined.glb'))).toBe(false);
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('routes the portable closure through an UltraShapePipeline-compatible call graph', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.conditioner_mask import SingleImageEncoder',
        'from ultrashape_runtime.models.denoisers.dit_mask import RefineDiT',
        'from ultrashape_runtime.models.autoencoders.model import ShapeVAE',
        'from ultrashape_runtime.pipelines import UltraShapePipeline',
        'class StubProcessor:',
        '    def __call__(self, image, mask=None):',
        '        del mask',
        '        return {"image": image, "mask": [[[ [1.0] ]]]}',
        'reference = [[[[1.0, 0.5, 0.25, 1.0], [0.2, 0.4, 0.6, 1.0]]]]',
        'voxel_cond = {"coords": [[0, 0, 0], [1, 1, 1]], "occupancies": [0.25, 0.75], "resolution": 4, "voxel_count": 2}',
        'coarse_surface = {',
        '    "mesh": {"bounds": {"min": (0.0, 0.0, 0.0), "extents": (1.0, 1.0, 1.0)}, "signature": 11, "vertex_count": 4, "face_count": 2, "surface_point_count": 4},',
        '    "voxel_cond": voxel_cond,',
        '    "sampled_surface_points": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],',
        '}',
        'pipeline = UltraShapePipeline(',
        '    vae=ShapeVAE(checkpoint_state={"tensors": {"weights": [0.7, 0.5, 0.3, 0.1]}}),',
        '    model=RefineDiT(checkpoint_state={"tensors": {"weights": [0.3, 0.6, 0.9, 0.1]}}),',
        '    scheduler={"consumed_timesteps": [2.0, 1.0, 0.0], "object_type": "FlowMatchEulerDiscreteScheduler"},',
        '    conditioner=SingleImageEncoder(checkpoint_state={"tensors": {"weights": [0.2, 0.4, 0.6, 0.8]}}),',
        '    image_processor=StubProcessor(),',
        ')',
        'mesh, latents = pipeline(image=reference, voxel_cond=voxel_cond, coarse_surface=coarse_surface, num_inference_steps=3, guidance_scale=5.5, seed=7, output_type="latent")',
        'print(json.dumps({',
        '    "pipeline": {',
        '        "class": pipeline.__class__.__name__,',
        '        "returns_latents": isinstance(latents, dict),',
        '        "execution_trace": mesh["execution_trace"],',
        '    },',
        '    "mesh": {',
        '        "authority": mesh["authority"],',
        '        "extractor": mesh["decoder"],',
        '    },',
        '    "latents": {',
        '        "model": latents["model"],',
        '        "latent_count": latents["latent_count"],',
        '        "timestep_count": latents["timestep_count"],',
        '    },',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      pipeline: {
        class: 'UltraShapePipeline',
        returns_latents: true,
        execution_trace: ['prepare_image', 'encode_cond', 'prepare_latents', 'diffusion_sampling', 'decode', 'extract'],
      },
      mesh: {
        authority: 'UltraShapePipeline._export',
        extractor: 'VanillaVolumeDecoder',
      },
      latents: {
        model: 'RefineDiT',
        latent_count: expect.any(Number),
        timestep_count: 3,
      },
    });
  });

  it('decodes field grids and extracts sparse mc surfaces through upstream-style decoder families', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'import sys, types',
        'cubvh = types.ModuleType("cubvh")',
        'cubvh.sparse_marching_cubes = lambda coords, corners, iso, ensure_consistency=False: ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, 1.02]], [[0, 1, 4], [0, 4, 2], [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6], [3, 5, 8], [3, 8, 6]])',
        'torch = types.ModuleType("torch")',
        'class Tensor:',
        '    def __init__(self, values, dtype=None):',
        '        self._values = values',
        '        self.dtype = dtype',
        '        self.shape = (len(values), len(values[0]) if values and isinstance(values[0], (list, tuple)) else 0)',
        '    def cpu(self):',
        '        return self',
        '    def tolist(self):',
        '        return self._values',
        'torch.int32 = "int32"',
        'torch.float32 = "float32"',
        'torch.tensor = lambda values, dtype=None: Tensor(values, dtype=dtype)',
        'sys.modules["cubvh"] = cubvh',
        'sys.modules["torch"] = torch',
        'from ultrashape_runtime.models.autoencoders.volume_decoders import VanillaVolumeDecoder, HierarchicalVolumeDecoding, get_sparse_valid_voxels',
        'from ultrashape_runtime.models.autoencoders.surface_extractors import SurfaceExtractors, extract_surface',
        'decoded_latents = {',
        '    "field_grid": [[[-1.0, -0.5], [-0.5, 0.25]], [[-0.25, 0.5], [0.5, 1.0]]],',
        '    "field_signature": 321,',
        '    "spatial_context": {"voxel_coords": [[0, 0, 0]]},',
        '}',
        'decoded_volume = VanillaVolumeDecoder().decode(decoded_latents)',
        'grid_only_volume = dict(decoded_volume)',
        'grid_only_volume.pop("coords", None)',
        'grid_only_volume.pop("corners", None)',
        'hierarchical = HierarchicalVolumeDecoding().decode(decoded_latents)',
        'coords, corners = get_sparse_valid_voxels(decoded_volume["grid_logits"] if "grid_logits" in decoded_volume else decoded_volume["field_grid"])',
        'surface = extract_surface(',
        '    extraction="mc",',
        '    coarse_surface={',
        '        "mesh": {',
        '            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],',
        '            "faces": [(0, 1, 2), (0, 1, 3)],',
        '            "signature": 17,',
        '        },',
        '    },',
        '    reference_asset={"byte_length": 68, "signature": 23},',
        '    decoded_volume=grid_only_volume,',
        '    preserve_scale=False,',
        ')',
        'print(json.dumps({',
        '    "decoder": {',
        '        "class": decoded_volume["decoder"],',
        '        "authority": decoded_volume["authority"],',
        '        "has_grid_logits": "grid_logits" in decoded_volume,',
        '        "sparse_cell_count": len(coords),',
        '        "corner_count": len(corners),',
        '    },',
        '    "hierarchical": {',
        '        "class": hierarchical["decoder"],',
        '        "hierarchy": hierarchical["hierarchy"],',
        '    },',
        '    "extractor": {',
        '        "family_keys": sorted(SurfaceExtractors.keys()),',
        '        "marching_cubes": surface["marching_cubes"],',
        '        "authority": surface["authority"],',
        '        "vertex_count": surface["vertex_count"],',
        '    },',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      decoder: {
        class: 'VanillaVolumeDecoder',
        authority: 'geo_decoder(query_grid)',
        has_grid_logits: true,
        sparse_cell_count: expect.any(Number),
        corner_count: expect.any(Number),
      },
      hierarchical: {
        class: 'HierarchicalVolumeDecoding',
        hierarchy: 'octree-near-surface',
      },
      extractor: {
        family_keys: ['dmc', 'mc'],
        marching_cubes: 'cubvh.sparse_marching_cubes',
        authority: 'grid_logits',
        vertex_count: expect.any(Number),
      },
    });
    const payload = JSON.parse(result.stdout) as {
      decoder: { sparse_cell_count: number; corner_count: number };
      extractor: { vertex_count: number };
    };
    expect(payload.decoder.sparse_cell_count).toBeGreaterThan(0);
    expect(payload.decoder.corner_count).toBe(payload.decoder.sparse_cell_count);
    expect(payload.extractor.vertex_count).toBeGreaterThan(0);
  });

  it('reports lamellar mesh-quality metrics without relying on screenshots', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.autoencoders.surface_extractors import evaluate_portable_mesh_quality_gate',
        'vertices = []',
        'faces = []',
        'for z in [0.0, 0.5, 1.0]:',
        '    base = len(vertices)',
        '    vertices.extend([(0.0, 0.0, z), (1.0, 0.0, z), (1.0, 1.0, z), (0.0, 1.0, z), (0.5, 0.5, z)])',
        '    faces.extend([(base + 0, base + 1, base + 4), (base + 1, base + 2, base + 4), (base + 2, base + 3, base + 4), (base + 3, base + 0, base + 4)])',
        'try:',
        '    evaluate_portable_mesh_quality_gate({"vertices": vertices, "faces": faces})',
        'except Exception as error:',
        '    print(json.dumps({"ok": False, "code": getattr(error, "code", None), "message": str(error)}))',
        'else:',
        '    print(json.dumps({"ok": True}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      ok: false,
      code: 'LOCAL_RUNTIME_UNAVAILABLE',
      message: expect.stringContaining('portable lamellar geometry rejected'),
    });
    expect(JSON.parse(result.stdout).message).toContain('boundary_edge_count=');
    expect(JSON.parse(result.stdout).message).toContain('slab_concentration=');
  });

  it('passes cohesive valid-ish portable geometry while labeling it non-authoritative fallback', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.autoencoders.surface_extractors import evaluate_portable_mesh_quality_gate',
        'vertices = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1),(0.5,0.5,0.5)]',
        'faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]',
        'metrics = evaluate_portable_mesh_quality_gate({"vertices": vertices, "faces": faces})',
        'print(json.dumps(metrics, sort_keys=True))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      passed: true,
      reason: 'ok',
      component_count: 1,
      largest_component_ratio: 1,
      boundary_edge_count: 0,
      non_manifold_edge_count: 0,
    });
  });

  it('maps unsupported backend requests to INVALID_INPUT through the public runner envelope', () => {
    const fixture = createRuntimeFixture();

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: path.join(fixture.sandbox, 'output'),
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'remote',
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
        error_code: 'INVALID_INPUT',
        error_message: 'UltraShape local runner is local-only in this MVP.',
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('fails honestly when real mode is explicitly requested but unavailable', () => {
    const fixture = createRuntimeFixture();

    try {
      const result = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
        cwd: repoRoot,
        encoding: 'utf8',
        input: JSON.stringify({
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: path.join(fixture.sandbox, 'output-real'),
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        }),
        env: {
          ...process.env,
          ULTRASHAPE_RUNTIME_MODE: 'real',
          PYTHONPATH: [fixture.stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
        },
      });

      expect(result.status).toBe(1);
      expect(JSON.parse(result.stdout)).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: 'Real runtime mode requested but unavailable: checkout-config:ultrashape_upstream_checkout.',
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('accepts only explicitly configured upstream checkouts with required real-mode markers', () => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'ultrashape-real-checkout-'));
    const checkout = createFakeUpstreamCheckout(sandbox);
    const driftedCheckout = path.join(sandbox, 'drifted-checkout');
    mkdirSync(driftedCheckout, { recursive: true });
    writeFileSync(path.join(driftedCheckout, 'LICENSE'), 'not enough markers\n', 'utf8');

    try {
      const result = runPythonSnippet(
        [
          'import json, sys',
          'from ultrashape_runtime.real_mode import describe_real_mode, validate_upstream_checkout',
          'valid = validate_upstream_checkout(sys.argv[1])',
          'missing = describe_real_mode(checkout_path=sys.argv[2])',
          'print(json.dumps({"valid": valid, "missing": missing}))',
        ].join('\n'),
        [checkout, driftedCheckout],
      );

      expect(result.status).toBe(0);
      const payload = JSON.parse(result.stdout) as {
        valid: { available: boolean; source: string; entrypoint: string; blockers: string[]; checkout_path: string };
        missing: { available: boolean; blockers: string[] };
      };
      expect(payload.valid).toMatchObject({
        available: true,
        source: 'checkout',
        checkout_path: checkout,
        entrypoint: 'scripts.infer_dit_refine.run_inference',
        blockers: [],
      });
      expect(payload.missing.available).toBe(false);
      expect(payload.missing.blockers).toContain('checkout-marker:scripts/infer_dit_refine.py');
      expect(payload.missing.blockers).toContain('checkout-marker:configs/infer_dit_refine.yaml');
      expect(payload.missing.blockers).toContain('checkout-marker:ultrashape');
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('invokes a configured upstream-like real entrypoint and normalizes output_dir/refined.glb', () => {
    const fixture = createRuntimeFixture();
    const checkout = createFakeUpstreamCheckout(fixture.sandbox);
    const upstreamConfigPath = path.join(checkout, 'configs', 'infer_dit_refine.yaml');
    writeRealDependencyStubs(fixture.stubRoot);
    const outputDir = path.join(fixture.sandbox, 'output-real-success');

    try {
      const result = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
        cwd: repoRoot,
        encoding: 'utf8',
        input: JSON.stringify({
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        }),
        env: {
          ...process.env,
          ULTRASHAPE_RUNTIME_MODE: 'real',
          ULTRASHAPE_UPSTREAM_CHECKOUT: checkout,
          PYTHONPATH: [fixture.stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
        },
      });

      expect(result.status).toBe(0);
      expect(JSON.parse(readFileSync(path.join(outputDir, 'refined.glb'), 'utf8'))).toMatchObject({
        image: fixture.imageInputPath,
        mesh: fixture.meshInputPath,
        ckpt: fixture.checkpoint,
        config: upstreamConfigPath,
        steps: 4,
        scale: 0.99,
        num_latents: 32768,
        chunk_size: 8000,
        octree_res: 1024,
        seed: 7,
        remove_bg: false,
        low_vram: false,
        checkout_marker: 'default',
      });
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          backend: 'local',
          format: 'glb',
          file_path: path.join(outputDir, 'refined.glb'),
          metrics: {
            runtime_mode: {
              selection: 'real-available',
              requested: 'real',
              active: 'real',
              real: {
                available: true,
                source: 'checkout',
                checkout_path: checkout,
                entrypoint: 'scripts.infer_dit_refine.run_inference',
                runtime_config: { path: configPath, available: true },
                upstream_config: { path: upstreamConfigPath, available: true },
              },
            },
            upstream: {
              output_name: 'reference_refined.glb',
            },
          },
          subtrees_loaded: ['upstream-real'],
        },
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('restores checkout-loaded scripts and ultrashape modules between real executions', () => {
    const fixture = createRuntimeFixture();
    const firstCheckout = createFakeUpstreamCheckout(fixture.sandbox, {
      checkoutName: 'fake-ultrashape-checkout-one',
      checkoutMarker: 'first-checkout',
    });
    const secondCheckout = createFakeUpstreamCheckout(fixture.sandbox, {
      checkoutName: 'fake-ultrashape-checkout-two',
      checkoutMarker: 'second-checkout',
    });
    writeRealDependencyStubs(fixture.stubRoot);
    const firstOutput = path.join(fixture.sandbox, 'output-real-first');
    const secondOutput = path.join(fixture.sandbox, 'output-real-second');

    try {
      const result = spawnSync(
        'python3',
        [
          '-c',
          [
            'import json, os, sys',
            'from pathlib import Path',
            'from ultrashape_runtime.real_mode import run_real_refine_pipeline',
            'def run_once(checkout, output_dir):',
            '    os.environ["ULTRASHAPE_UPSTREAM_CHECKOUT"] = checkout',
            '    run_real_refine_pipeline(',
            '        reference_image=sys.argv[5],',
            '        coarse_mesh=sys.argv[6],',
            '        output_dir=output_dir,',
            '        output_format="glb",',
            '        checkpoint=sys.argv[3],',
            '        config_path=sys.argv[4],',
            '        ext_dir="unused",',
            '        backend="local",',
            '        steps=4,',
            '        guidance_scale=6,',
            '        seed=7,',
            '        preserve_scale=True,',
            '    )',
            'run_once(sys.argv[1], sys.argv[7])',
            'run_once(sys.argv[2], sys.argv[8])',
            'print(json.dumps({',
            '    "first": json.loads((Path(sys.argv[7]) / "refined.glb").read_text(encoding="utf8"))["checkout_marker"],',
            '    "second": json.loads((Path(sys.argv[8]) / "refined.glb").read_text(encoding="utf8"))["checkout_marker"],',
            '    "checkout_modules": sorted(name for name in sys.modules if name == "scripts" or name.startswith("scripts.") or name == "ultrashape" or name.startswith("ultrashape.")),',
            '}))',
          ].join('\n'),
          firstCheckout,
          secondCheckout,
          fixture.checkpoint,
          configPath,
          fixture.imageInputPath,
          fixture.meshInputPath,
          firstOutput,
          secondOutput,
        ],
        {
          cwd: repoRoot,
          encoding: 'utf8',
          env: {
            ...process.env,
            PYTHONPATH: [fixture.stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
          },
        },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual({
        first: 'first-checkout',
        second: 'second-checkout',
        checkout_modules: [],
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('fails forced real honestly when the upstream entrypoint does not produce the expected refined mesh', () => {
    const fixture = createRuntimeFixture();
    const checkout = createFakeUpstreamCheckout(fixture.sandbox, { writesOutput: false });
    writeRealDependencyStubs(fixture.stubRoot);

    try {
      const result = spawnSync('python3', ['-m', 'ultrashape_runtime.local_runner'], {
        cwd: repoRoot,
        encoding: 'utf8',
        input: JSON.stringify({
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: path.join(fixture.sandbox, 'output-real-missing'),
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        }),
        env: {
          ...process.env,
          ULTRASHAPE_RUNTIME_MODE: 'real',
          ULTRASHAPE_UPSTREAM_CHECKOUT: checkout,
          PYTHONPATH: [fixture.stubRoot, runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
        },
      });

      expect(result.status).toBe(1);
      expect(JSON.parse(result.stdout)).toEqual({
        ok: false,
        error_code: 'LOCAL_RUNTIME_UNAVAILABLE',
        error_message: 'Upstream real mode did not produce expected refined output: reference_refined.glb.',
      });
      expect(existsSync(path.join(fixture.sandbox, 'output-real-missing', 'refined.glb'))).toBe(false);
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('selects authoritative real mode in auto only when checkout and real-only dependencies are ready', () => {
    const fixture = createRuntimeFixture();
    const checkout = createFakeUpstreamCheckout(fixture.sandbox);
    const upstreamConfigPath = path.join(checkout, 'configs', 'infer_dit_refine.yaml');
    writeRealDependencyStubs(fixture.stubRoot);
    const outputDir = path.join(fixture.sandbox, 'output-auto-real');

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
        { ULTRASHAPE_UPSTREAM_CHECKOUT: checkout },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(readFileSync(path.join(outputDir, 'refined.glb'), 'utf8'))).toMatchObject({
        ckpt: fixture.checkpoint,
        config: upstreamConfigPath,
        checkout_marker: 'default',
      });
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          metrics: {
            runtime_mode: {
              selection: 'real-available',
              requested: 'auto',
              active: 'real',
              real: {
                available: true,
                dependencies: {
                  flash_attn: { available: true, required: true },
                },
              },
            },
          },
        },
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('falls back to portable in auto when real is blocked by real-only dependencies', () => {
    const fixture = createRuntimeFixture();
    const checkout = createFakeUpstreamCheckout(fixture.sandbox);
    const outputDir = path.join(fixture.sandbox, 'output-auto-portable');

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
        { ULTRASHAPE_UPSTREAM_CHECKOUT: checkout },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          metrics: {
            runtime_mode: {
              selection: 'portable-only',
              requested: 'auto',
              active: 'portable',
              real: {
                available: false,
                blockers: expect.arrayContaining(['dependency:flash_attn']),
              },
              portable: {
                available: true,
                authoritative: false,
              },
            },
          },
          fallbacks: ['real->portable', 'flash_attn->sdpa'],
        },
      });
      expect(readFileSync(path.join(outputDir, 'refined.glb')).subarray(0, 4).toString('ascii')).toBe('glTF');
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('bypasses ready real mode when portable is forced and keeps portable non-authoritative', () => {
    const fixture = createRuntimeFixture();
    const checkout = createFakeUpstreamCheckout(fixture.sandbox);
    writeRealDependencyStubs(fixture.stubRoot);
    const outputDir = path.join(fixture.sandbox, 'output-forced-portable');

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputDir,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
        { ULTRASHAPE_RUNTIME_MODE: 'portable', ULTRASHAPE_UPSTREAM_CHECKOUT: checkout },
      );

      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        ok: true,
        result: {
          metrics: {
            runtime_mode: {
              selection: 'portable-only',
              requested: 'portable',
              active: 'portable',
              real: {
                available: false,
                reason: expect.stringContaining('bypassed because portable mode was forced'),
              },
              portable: {
                available: true,
                authoritative: false,
                reason: expect.stringContaining('not the authoritative upstream closure'),
              },
            },
          },
        },
      });
      expect(readFileSync(path.join(outputDir, 'refined.glb')).subarray(0, 4).toString('ascii')).toBe('glTF');
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('exposes a portable upstream-shaped model subset API for conditioning, denoising, and decoding', () => {
    const result = runPythonSnippet(
      [
        'import json',
        'from ultrashape_runtime.models.conditioner_mask import SingleImageEncoder',
        'from ultrashape_runtime.models.denoisers.dit_mask import RefineDiT',
        'from ultrashape_runtime.models.autoencoders.model import ShapeVAE',
        'reference_asset = {',
        '    "image_tensor": [[[[1.0, 0.5, 0.25, 1.0], [0.1, 0.2, 0.3, 1.0]]]],',
        '    "mask_tensor": [[[[1.0], [0.0]]]],',
        '}',
        'coarse_surface = {',
        '    "mesh": {"bounds": {"extents": (1.0, 2.0, 3.0)}, "signature": 41},',
        '    "voxel_cond": {',
        '        "coords": [[0, 0, 0], [1, 1, 1]],',
        '        "occupancies": [0.25, 0.75],',
        '        "resolution": 4,',
        '        "voxel_count": 2,',
        '    },',
        '}',
        'conditioner = SingleImageEncoder(checkpoint_state={"tensors": {"weights": [0.2, 0.4, 0.6, 0.8]}})',
        'conditioning = conditioner.build(reference_asset=reference_asset, coarse_surface=coarse_surface)',
        'forward_tokens = conditioner.forward(reference_asset["image_tensor"], mask=reference_asset["mask_tensor"])',
        'unconditional = conditioner.unconditional_embedding(batch_size=1, num_tokens=len(forward_tokens["main"]))',
        'dit = RefineDiT(checkpoint_state={"tensors": {"weights": [0.3, 0.6, 0.9, 0.1]}})',
        'forward_latents = dit.forward([[0.1, 0.2, 0.3, 0.4]], [1.0], forward_tokens, voxel_cond=coarse_surface["voxel_cond"], guidance_cond=[4.0])',
        'guided = dit.denoise(latents=[0.1, 0.2, 0.3, 0.4], timesteps=[3.0, 1.0], context=conditioning["context"], context_mask=conditioning["context_mask"], voxel_cond=coarse_surface["voxel_cond"], guidance_scale=8.0, schedule={"step_count": 2, "object_type": "FlowMatchEulerDiscreteScheduler"}, seed=7)',
        'unguided = dit.denoise(latents=[0.1, 0.2, 0.3, 0.4], timesteps=[3.0, 1.0], context=conditioning["context"], context_mask=conditioning["context_mask"], voxel_cond=coarse_surface["voxel_cond"], guidance_scale=1.0, schedule={"step_count": 2, "object_type": "FlowMatchEulerDiscreteScheduler"}, seed=7)',
        'vae = ShapeVAE(checkpoint_state={"tensors": {"weights": [0.7, 0.5, 0.3, 0.1]}})',
        'decoded = vae.decode([[0.2, 0.4, 0.6, 0.8]])',
        'query_logits = vae.query(decoded, [[-1.0, -1.0, -1.0], [0.0, 0.5, 1.0]])',
        'latent_mesh, latent_grid = vae.latents2mesh(decoded)',
        'print(json.dumps({',
        '    "conditioning": {',
        '        "token_count": len(forward_tokens["main"]),',
        '        "feature_count": len(forward_tokens["main"][0]),',
        '        "mask_count": len(forward_tokens["main_mask"]),',
        '        "unconditional_sum": round(sum(sum(row) for row in unconditional["main"]), 6),',
        '        "conditioning_signature": conditioning["conditioning_signature"],',
        '        "has_main_image_encoder": hasattr(conditioner, "main_image_encoder"),',
        '    },',
        '    "dit": {',
        '        "forward_token_count": len(forward_latents),',
        '        "forward_feature_count": len(forward_latents[0]),',
        '        "guided_signature": guided["latent_signature"],',
        '        "unguided_signature": unguided["latent_signature"],',
        '        "has_timestep_embedder": hasattr(dit, "t_embedder"),',
        '    },',
        '    "vae": {',
        '        "decoded_token_count": len(decoded),',
        '        "decoded_feature_count": len(decoded[0]),',
        '        "query_count": len(query_logits),',
        '        "query_signature": round(sum(query_logits), 6),',
        '        "latent_mesh_count": len(latent_mesh),',
        '        "latent_grid_decoder": latent_grid["decoder"],',
        '    },',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      conditioning: {
        token_count: 1,
        feature_count: 4,
        mask_count: 1,
        unconditional_sum: 0,
        conditioning_signature: expect.any(Number),
        has_main_image_encoder: true,
      },
      dit: {
        forward_token_count: 1,
        forward_feature_count: 4,
        guided_signature: expect.any(Number),
        unguided_signature: expect.any(Number),
        has_timestep_embedder: true,
      },
      vae: {
        decoded_token_count: 1,
        decoded_feature_count: 4,
        query_count: 2,
        query_signature: expect.any(Number),
        latent_mesh_count: 1,
        latent_grid_decoder: 'VanillaVolumeDecoder',
      },
    });
    const payload = JSON.parse(result.stdout) as {
      dit: { guided_signature: number; unguided_signature: number };
    };
    expect(payload.dit.guided_signature).not.toBe(payload.dit.unguided_signature);
  });

  it('restores coarse-mesh scale only when preserve_scale is enabled', () => {
    const fixture = createRuntimeFixture();
    const outputPreserved = path.join(fixture.sandbox, 'output-preserved');
    const outputNormalized = path.join(fixture.sandbox, 'output-normalized');

    try {
      const preserved = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputPreserved,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: true,
        },
        fixture.stubRoot,
      );
      const normalized = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: outputNormalized,
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
          backend: 'local',
          steps: 4,
          guidance_scale: 6,
          seed: 7,
          preserve_scale: false,
        },
        fixture.stubRoot,
      );

      expect(preserved.status).toBe(0);
      expect(normalized.status).toBe(0);

      const coarseExtents = meshExtents(readGlbVertices(fixture.meshInputPath));
      const preservedExtents = meshExtents(readGlbVertices(path.join(outputPreserved, 'refined.glb')));
      const normalizedExtents = meshExtents(readGlbVertices(path.join(outputNormalized, 'refined.glb')));

      expect(preservedExtents[0]).toBeCloseTo(coarseExtents[0], 5);
      expect(preservedExtents[1]).toBeCloseTo(coarseExtents[1], 5);
      expect(preservedExtents[2]).toBeCloseTo(coarseExtents[2], 5);
      expect(normalizedExtents[0]).toBeCloseTo(1.9998, 3);
      expect(normalizedExtents[1]).toBeCloseTo(1.9998, 3);
      expect(normalizedExtents[2]).toBeCloseTo(1.9998, 3);
      expect(normalizedExtents[0]).not.toBeCloseTo(coarseExtents[0], 3);

      expect(JSON.parse(preserved.stdout)).toMatchObject({
        ok: true,
        result: {
          metrics: {
            conditioning: {
              normalization_transform: {
                max_extent: 1,
                scale_factor: 1.9998,
              },
            },
            gate: {
              preserve_scale: true,
            },
          },
        },
      });
      expect(JSON.parse(normalized.stdout)).toMatchObject({
        ok: true,
        result: {
          metrics: {
            gate: {
              preserve_scale: false,
            },
          },
        },
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('rejects JPEG payloads until true JPEG preprocessing exists', () => {
    const fixture = createRuntimeFixture();
    const jpegPath = path.join(fixture.sandbox, 'reference.jpg');
    writeFileSync(jpegPath, Buffer.from([0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x01]));

    try {
      const result = runLocalRunner(
        {
          reference_image: jpegPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: path.join(fixture.sandbox, 'output-jpeg'),
          checkpoint: fixture.checkpoint,
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
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
        error_code: 'INVALID_INPUT',
        error_message: `reference_image must be a decodable PNG payload: ${jpegPath}.`,
      });
    } finally {
      rmSync(fixture.sandbox, { recursive: true, force: true });
    }
  });

  it('maps unreadable checkpoints to WEIGHTS_MISSING through the real runner path', () => {
    const fixture = createRuntimeFixture();

    try {
      const result = runLocalRunner(
        {
          reference_image: fixture.imageInputPath,
          coarse_mesh: fixture.meshInputPath,
          output_dir: path.join(fixture.sandbox, 'output'),
          checkpoint: path.join(fixture.sandbox, 'missing.pt'),
          config_path: configPath,
          ext_dir: fixture.extDir,
          output_format: 'glb',
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
