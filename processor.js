const manifest = require('./manifest.json');
const { executeUltraShapeRefiner } = require('./runtime/modly/processes/ultrashape-refiner/index.js');

module.exports = async function processor(input = {}, params = {}, context = {}) {
  const referenceImage = resolveNamedInput(input, 'reference_image', 'image');
  const coarseMesh = resolveNamedInput(input, 'coarse_mesh', 'mesh');
  const outputDir = resolveOutputDir(context);
  const mergedParams = mergeManifestDefaults(params);
  const progress = typeof context.progress === 'function' ? context.progress : null;

  const result = await executeUltraShapeRefiner(
    {
      reference_image: referenceImage.filePath,
      coarse_mesh: coarseMesh.filePath,
      output_dir: outputDir,
      checkpoint: mergedParams.checkpoint,
      params: mergedParams,
      abortSignal: context.abortSignal,
    },
    {
      remoteClient: context.remoteClient,
      localAdapter: context.localAdapter,
      preflight: context.preflight,
      onProgress: progress
        ? (event) => progress(event.progress ?? 0, event.message)
        : undefined,
    },
  );

  return { filePath: result.refinedMesh.path };
};

function resolveNamedInput(input, key, expectedType) {
  const named = input && input.inputs ? input.inputs[key] : undefined;

  if (!named || typeof named !== 'object') {
    throw createInputError(`Missing required named input: ${key}.`);
  }

  if (named.type !== expectedType) {
    throw createInputError(`Input ${key} must be of type ${expectedType}.`);
  }

  if (typeof named.filePath !== 'string' || !named.filePath.trim()) {
    throw createInputError(`Input ${key} must include a filePath.`);
  }

  return named;
}

function resolveOutputDir(context) {
  const candidate = context.workspaceDir ?? context.tempDir;

  if (typeof candidate !== 'string' || !candidate.trim()) {
    throw createInputError('Modly context must provide workspaceDir or tempDir for output packaging.');
  }

  return candidate;
}

function mergeManifestDefaults(params) {
  const properties = manifest.nodes[0].params_schema.properties;
  const defaults = Object.fromEntries(
    Object.entries(properties).map(([key, schema]) => [key, schema.default ?? null]),
  );

  return {
    ...defaults,
    ...(params && typeof params === 'object' ? params : {}),
  };
}

function createInputError(message) {
  const error = new Error(message);
  error.code = 'MISSING_INPUT';
  return error;
}
