import type {
  UltraShapePreflightResult,
  UltraShapeProcessError,
  UltraShapeProgressEvent,
  UltraShapeRefinerResult,
} from './types.js';
import { createProcessError } from './validate.js';

type ResolvedBackend = UltraShapeProgressEvent['backend'];

export function validatingProgressEvent(): UltraShapeProgressEvent {
  return {
    stage: 'validating',
    message: 'Validating UltraShape refiner inputs and parameters.',
    progress: 5,
  };
}

export function preflightProgressEvent(result: UltraShapePreflightResult): UltraShapeProgressEvent {
  return {
    stage: 'preflight',
    message:
      result.reason ?? `Preflight selected ${result.selectedBackend} execution for UltraShape refinement.`,
    progress: 20,
    backend: result.selectedBackend,
  };
}

export function runningProgressEvent(backend: ResolvedBackend): UltraShapeProgressEvent {
  return {
    stage: 'running',
    message: `Running UltraShape refinement via ${backend} backend.`,
    progress: 60,
    backend,
  };
}

export function packagingProgressEvent(backend: ResolvedBackend): UltraShapeProgressEvent {
  return {
    stage: 'packaging',
    message: 'Packaging refined mesh output.',
    progress: 90,
    backend,
  };
}

export function completedProgressEvent(result: UltraShapeRefinerResult): UltraShapeProgressEvent {
  return {
    stage: 'completed',
    message: `Refined mesh ready at ${result.refinedMesh.path}.`,
    progress: 100,
    backend: result.backendUsed,
  };
}

export function terminalErrorProgressEvent(
  error: unknown,
  backend?: ResolvedBackend,
): UltraShapeProgressEvent {
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

export function cancelledProgressEvent(backend?: ResolvedBackend): UltraShapeProgressEvent {
  return {
    stage: 'cancelled',
    message: 'UltraShape refinement was cancelled before completion.',
    progress: 100,
    backend,
  };
}

export function normalizeRuntimeError(error: unknown): UltraShapeProcessError {
  if (isProcessError(error)) {
    return error;
  }

  if (isAbortError(error)) {
    return createProcessError('CANCELLED', 'UltraShape refinement was cancelled.', undefined);
  }

  if (error instanceof Error) {
    return createProcessError(
      'LOCAL_RUNTIME_UNAVAILABLE',
      `LOCAL_RUNTIME_UNAVAILABLE: ${error.message}`,
      'backend',
    );
  }

  return createProcessError(
    'LOCAL_RUNTIME_UNAVAILABLE',
    'LOCAL_RUNTIME_UNAVAILABLE: The TypeScript compatibility layer failed before reaching the Python runtime boundary.',
    'backend',
  );
}

function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false;
  }

  const candidate = error as { name?: unknown; message?: unknown };
  return (
    candidate.name === 'AbortError' ||
    (typeof candidate.message === 'string' && /abort/i.test(candidate.message))
  );
}

function isProcessError(error: unknown): error is UltraShapeProcessError {
  if (!(error instanceof Error) || !('code' in error)) {
    return false;
  }

  const code = (error as { code?: unknown }).code;
  return typeof code === 'string';
}
