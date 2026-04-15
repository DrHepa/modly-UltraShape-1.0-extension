const { accessSync, constants } = require('node:fs');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const MESH_EXTENSIONS = new Set(['.glb', '.obj', '.fbx', '.ply']);
const IMAGE_MEDIA_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const MESH_MEDIA_TYPES = new Set([
  'model/gltf-binary',
  'model/obj',
  'model/fbx',
  'model/ply',
  'application/octet-stream',
]);

const DEFAULT_REFINER_PARAMS = {
  checkpoint: null,
  backend: 'auto',
  steps: 30,
  guidance_scale: 5.5,
  seed: null,
  preserve_scale: true,
  output_format: 'glb',
};

function validateRefinerRequest(input) {
  if (!input || typeof input !== 'object') {
    throw createProcessError('INVALID_PARAMS', 'Request must be an object.');
  }

  const request = input;
  const referenceImage = ensureAssetPresent(request.reference_image, 'reference_image');
  const coarseMesh = ensureAssetPresent(request.coarse_mesh, 'coarse_mesh');
  const outputDir = ensureNonEmptyPath(request.output_dir, 'output_dir');
  const checkpoint = normalizeCheckpoint(request.checkpoint);
  const params = validateParams(request.params || {}, checkpoint);

  validateAsset(referenceImage, 'reference_image', 'image');
  validateAsset(coarseMesh, 'coarse_mesh', 'mesh');

  accessOrThrow(outputDir, 'output_dir');
  if (checkpoint) {
    accessOrThrow(checkpoint, 'checkpoint');
  }

  return {
    reference_image: referenceImage,
    coarse_mesh: coarseMesh,
    output_dir: outputDir,
    checkpoint,
    params,
    correlation_id: request.correlation_id,
    abortSignal: request.abortSignal,
  };
}

function validateParams(input, checkpointOverride) {
  const backend = validateBackend(input.backend ?? DEFAULT_REFINER_PARAMS.backend);
  const steps = validatePositiveInteger(input.steps ?? DEFAULT_REFINER_PARAMS.steps, 'steps');
  const guidanceScale = validatePositiveNumber(
    input.guidance_scale ?? DEFAULT_REFINER_PARAMS.guidance_scale,
    'guidance_scale',
  );
  const seed = validateNullableInteger(input.seed ?? DEFAULT_REFINER_PARAMS.seed, 'seed');
  const preserveScale = validateBoolean(
    input.preserve_scale ?? DEFAULT_REFINER_PARAMS.preserve_scale,
    'preserve_scale',
  );
  const outputFormat = validateOutputFormat(
    input.output_format ?? DEFAULT_REFINER_PARAMS.output_format,
  );
  const checkpoint = normalizeCheckpoint(checkpointOverride ?? input.checkpoint ?? null);

  return {
    checkpoint,
    backend,
    steps,
    guidance_scale: guidanceScale,
    seed,
    preserve_scale: preserveScale,
    output_format: outputFormat,
  };
}

function inferAssetRef(input, expectedKind) {
  if (typeof input === 'string') {
    return { path: input, kind: expectedKind, mediaType: inferMediaType(input, expectedKind) };
  }

  return {
    ...input,
    kind: expectedKind,
    mediaType: input.mediaType ?? inferMediaType(input.path, expectedKind),
  };
}

function ensureAssetPresent(asset, field) {
  if (typeof asset === 'string' && asset.trim()) {
    return asset.trim();
  }

  if (asset && typeof asset === 'object' && typeof asset.path === 'string' && asset.path.trim()) {
    return { ...asset, path: asset.path.trim() };
  }

  throw createProcessError('MISSING_INPUT', `Missing required field: ${field}.`, field);
}

function ensureNonEmptyPath(value, field) {
  if (typeof value !== 'string' || !value.trim()) {
    throw createProcessError('MISSING_INPUT', `Missing required field: ${field}.`, field);
  }

  return value.trim();
}

function normalizeCheckpoint(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }

  if (typeof value !== 'string' || !value.trim()) {
    throw createProcessError('INVALID_PARAMS', 'checkpoint must be a string or null.', 'checkpoint');
  }

  return value.trim();
}

function validateAsset(input, field, expectedKind) {
  const asset = inferAssetRef(input, expectedKind);

  if (typeof input !== 'string' && input.kind !== expectedKind) {
    throw createProcessError(
      'UNSUPPORTED_ASSET_TYPE',
      `${field} must be provided as a ${expectedKind} asset.`,
      field,
    );
  }

  accessOrThrow(asset.path, field);

  const allowedExtensions = expectedKind === 'image' ? IMAGE_EXTENSIONS : MESH_EXTENSIONS;
  const allowedMediaTypes = expectedKind === 'image' ? IMAGE_MEDIA_TYPES : MESH_MEDIA_TYPES;
  const extension = extensionOf(asset.path);
  const mediaType = asset.mediaType && asset.mediaType.toLowerCase();

  if (!allowedExtensions.has(extension) && (!mediaType || !allowedMediaTypes.has(mediaType))) {
    throw createProcessError(
      'UNSUPPORTED_ASSET_TYPE',
      `${field} must use an allowed ${expectedKind} format.`,
      field,
    );
  }
}

function accessOrThrow(path, field) {
  try {
    accessSync(path, constants.R_OK);
  } catch {
    throw createProcessError('UNREADABLE_ASSET', `${String(field)} is not readable: ${path}.`, field);
  }
}

function validateBackend(value) {
  const allowed = ['auto', 'local', 'remote', 'hybrid'];
  if (typeof value !== 'string' || !allowed.includes(value)) {
    throw createProcessError('INVALID_PARAMS', 'backend must be auto, local, remote, or hybrid.', 'backend');
  }

  return value;
}

function validateOutputFormat(value) {
  const allowed = ['glb', 'obj', 'fbx', 'ply'];
  if (typeof value !== 'string' || !allowed.includes(value)) {
    throw createProcessError('INVALID_PARAMS', 'output_format must be glb, obj, fbx, or ply.', 'output_format');
  }

  return value;
}

function validatePositiveInteger(value, field) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw createProcessError('INVALID_PARAMS', `${field} must be a positive integer.`, field);
  }

  return value;
}

function validatePositiveNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw createProcessError('INVALID_PARAMS', `${field} must be a positive number.`, field);
  }

  return value;
}

function validateNullableInteger(value, field) {
  if (value === null) {
    return null;
  }

  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw createProcessError('INVALID_PARAMS', `${field} must be an integer or null.`, field);
  }

  return value;
}

function validateBoolean(value, field) {
  if (typeof value !== 'boolean') {
    throw createProcessError('INVALID_PARAMS', `${field} must be a boolean.`, field);
  }

  return value;
}

function inferMediaType(path, kind) {
  const extension = extensionOf(path);

  if (kind === 'image') {
    if (extension === '.png') return 'image/png';
    if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg';
    if (extension === '.webp') return 'image/webp';
    return undefined;
  }

  if (extension === '.glb') return 'model/gltf-binary';
  if (extension === '.obj') return 'model/obj';
  if (extension === '.fbx') return 'model/fbx';
  if (extension === '.ply') return 'model/ply';
  return undefined;
}

function extensionOf(path) {
  const index = path.lastIndexOf('.');
  return index >= 0 ? path.slice(index).toLowerCase() : '';
}

function createProcessError(code, message, field) {
  const error = new Error(message);
  error.code = code;
  error.field = field;
  error.recoverable = code !== 'BACKEND_UNAVAILABLE';
  return error;
}

module.exports = {
  DEFAULT_REFINER_PARAMS,
  validateRefinerRequest,
  validateParams,
  inferAssetRef,
  createProcessError,
};
