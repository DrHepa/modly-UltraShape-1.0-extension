const { createProcessError } = require('./validate.js');

function validatingProgressEvent() {
  return {
    stage: 'validating',
    message: 'Validating UltraShape refiner inputs and parameters.',
    progress: 5,
  };
}

function preflightProgressEvent(result) {
  return {
    stage: 'preflight',
    message: result.reason || `Preflight selected ${result.selectedBackend} execution for UltraShape refinement.`,
    progress: 20,
    backend: result.selectedBackend,
  };
}

function runningProgressEvent(backend) {
  return {
    stage: 'running',
    message: `Running UltraShape refinement via ${backend} backend.`,
    progress: 60,
    backend,
  };
}

function packagingProgressEvent(backend) {
  return {
    stage: 'packaging',
    message: 'Packaging refined mesh output.',
    progress: 90,
    backend,
  };
}

function completedProgressEvent(result) {
  return {
    stage: 'completed',
    message: `Refined mesh ready at ${result.refinedMesh.path}.`,
    progress: 100,
    backend: result.backendUsed,
  };
}

function terminalErrorProgressEvent(error, backend) {
  const normalized = normalizeRuntimeError(error);
  if (normalized.code === 'CANCELLED') {
    return cancelledProgressEvent(backend);
  }

  return {
    stage: 'failed',
    message: normalized.message,
    progress: 100,
    backend,
  };
}

function cancelledProgressEvent(backend) {
  return {
    stage: 'cancelled',
    message: 'UltraShape refinement was cancelled before completion.',
    progress: 100,
    backend,
  };
}

function normalizeRuntimeError(error) {
  if (isProcessError(error)) {
    return error;
  }

  if (isAbortError(error)) {
    return createProcessError('CANCELLED', 'UltraShape refinement was cancelled.', undefined);
  }

  if (error instanceof Error) {
    return createProcessError('BACKEND_UNAVAILABLE', `BACKEND_UNAVAILABLE: ${error.message}`, 'backend');
  }

  return createProcessError(
    'BACKEND_UNAVAILABLE',
    'BACKEND_UNAVAILABLE: UltraShape backend failed unexpectedly.',
    'backend',
  );
}

function isAbortError(error) {
  if (!error || typeof error !== 'object') {
    return false;
  }

  return error.name === 'AbortError' || (typeof error.message === 'string' && /abort/i.test(error.message));
}

function isProcessError(error) {
  return Boolean(error instanceof Error && typeof error.code === 'string');
}

module.exports = {
  validatingProgressEvent,
  preflightProgressEvent,
  runningProgressEvent,
  packagingProgressEvent,
  completedProgressEvent,
  terminalErrorProgressEvent,
  cancelledProgressEvent,
  normalizeRuntimeError,
};
