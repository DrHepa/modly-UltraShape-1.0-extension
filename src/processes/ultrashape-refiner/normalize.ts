import { randomUUID } from 'node:crypto';

import type {
  UltraShapeNormalizedRequest,
  UltraShapeRequestInput,
} from './types.js';
import type { UltraShapeValidatedRequest } from './validate.js';
import { inferAssetRef, validateRefinerRequest } from './validate.js';

export function normalizeRefinerRequest(
  input: UltraShapeRequestInput | UltraShapeValidatedRequest,
): UltraShapeNormalizedRequest {
  const validated = isValidatedRequest(input) ? input : validateRefinerRequest(input);

  return {
    correlationId: validated.correlation_id?.trim() || randomUUID(),
    referenceImage: inferAssetRef(validated.reference_image, 'image'),
    coarseMesh: inferAssetRef(validated.coarse_mesh, 'mesh'),
    outputDir: validated.output_dir,
    params: validated.params,
    abortSignal: validated.abortSignal,
    requestedBackend: validated.params.backend,
  };
}

function isValidatedRequest(
  input: UltraShapeRequestInput | UltraShapeValidatedRequest,
): input is UltraShapeValidatedRequest {
  return 'params' in input && 'reference_image' in input && 'coarse_mesh' in input && 'output_dir' in input;
}
