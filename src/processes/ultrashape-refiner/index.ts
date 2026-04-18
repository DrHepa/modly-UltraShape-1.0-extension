import { preflightProgressEvent, validatingProgressEvent } from './progress.js';
import { normalizeRefinerRequest } from './normalize.js';
import { preflightRefinerExecution, type UltraShapePreflightOptions } from './preflight.js';
import { runRefinerRuntime, type UltraShapeRuntimeOptions } from './runtime.js';
import type {
  UltraShapeRequestInput,
  UltraShapeRefinerResult,
} from './types.js';
import { validateRefinerRequest } from './validate.js';

export interface ExecuteUltraShapeRefinerOptions extends UltraShapeRuntimeOptions {
  preflight?: UltraShapePreflightOptions;
}

export async function executeUltraShapeRefiner(
  input: UltraShapeRequestInput,
  options: ExecuteUltraShapeRefinerOptions = {},
): Promise<UltraShapeRefinerResult> {
  const emit = options.onProgress ?? (() => undefined);

  emit(validatingProgressEvent());
  const normalized = normalizeRefinerRequest(validateRefinerRequest(input));
  const preflight = preflightRefinerExecution(normalized.requestedBackend ?? 'auto', {
    ...options.preflight,
    requestedOutputFormat: normalized.params.output_format,
  });
  emit(preflightProgressEvent(preflight));

  return runRefinerRuntime(normalized, preflight, options);
}
