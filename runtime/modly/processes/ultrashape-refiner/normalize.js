const { randomUUID } = require('node:crypto');

const { inferAssetRef, validateRefinerRequest } = require('./validate.js');

function normalizeRefinerRequest(input) {
  const validated = isValidatedRequest(input) ? input : validateRefinerRequest(input);

  return {
    correlationId: validated.correlation_id && validated.correlation_id.trim() ? validated.correlation_id.trim() : randomUUID(),
    referenceImage: inferAssetRef(validated.reference_image, 'image'),
    coarseMesh: inferAssetRef(validated.coarse_mesh, 'mesh'),
    outputDir: validated.output_dir,
    params: validated.params,
    abortSignal: validated.abortSignal,
    requestedBackend: validated.params.backend,
  };
}

function isValidatedRequest(input) {
  return Boolean(input && typeof input === 'object' && 'params' in input && 'reference_image' in input && 'coarse_mesh' in input && 'output_dir' in input);
}

module.exports = {
  normalizeRefinerRequest,
};
