const { preflightProgressEvent, validatingProgressEvent } = require('./progress.js');
const { normalizeRefinerRequest } = require('./normalize.js');
const { preflightRefinerExecution } = require('./preflight.js');
const { runRefinerRuntime } = require('./runtime.js');
const { validateRefinerRequest } = require('./validate.js');

async function executeUltraShapeRefiner(input, options = {}) {
  const emit = options.onProgress || (() => undefined);

  emit(validatingProgressEvent());
  const validated = validateRefinerRequest(input);
  const normalized = normalizeRefinerRequest(validated);
  const preflight = preflightRefinerExecution(normalized.requestedBackend || 'auto', options.preflight);
  emit(preflightProgressEvent(preflight));

  return runRefinerRuntime(normalized, preflight, options);
}

module.exports = {
  executeUltraShapeRefiner,
};
