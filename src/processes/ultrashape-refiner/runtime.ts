import { copyFileSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

import type {
  UltraShapeExecutionClient,
  UltraShapeExecutionRequest,
  UltraShapeRuntimeAdapter,
} from '../../adapters/ultrashape/client.js';
import { UltraShapeLocalAdapter } from '../../adapters/ultrashape/local.js';
import { UltraShapeRemoteAdapter } from '../../adapters/ultrashape/remote.js';
import {
  cancelledProgressEvent,
  completedProgressEvent,
  normalizeRuntimeError,
  packagingProgressEvent,
  runningProgressEvent,
  terminalErrorProgressEvent,
} from './progress.js';
import type {
  UltraShapeNormalizedRequest,
  UltraShapePreflightResult,
  UltraShapeProgressEvent,
  UltraShapeRefinerResult,
} from './types.js';

export interface UltraShapeRuntimeOptions {
  remoteClient?: UltraShapeExecutionClient;
  localAdapter?: UltraShapeRuntimeAdapter;
  onProgress?: (event: UltraShapeProgressEvent) => void;
}

export async function runRefinerRuntime(
  request: UltraShapeNormalizedRequest,
  preflight: UltraShapePreflightResult,
  options: UltraShapeRuntimeOptions = {},
): Promise<UltraShapeRefinerResult> {
  const emit = options.onProgress ?? (() => undefined);
  const signal = request.abortSignal;

  try {
    throwIfAborted(signal);

    const adapter = selectAdapter(preflight.selectedBackend, options);

    emit(runningProgressEvent(preflight.selectedBackend));

    const artifact = await adapter.run({
      request,
      backend: preflight.selectedBackend,
      signal,
      onUpdate: (update) => {
        emit({
          stage: 'running',
          message: update.message,
          progress: update.progress,
          backend: preflight.selectedBackend,
        });
      },
    });

    throwIfAborted(signal);
    emit(packagingProgressEvent(preflight.selectedBackend));

    const refinedMeshPath = packageArtifact(
      artifact.path,
      request.outputDir,
      artifact.format ?? request.params.output_format,
    );

    const result: UltraShapeRefinerResult = {
      refinedMesh: {
        path: refinedMeshPath,
        kind: 'mesh',
      },
      backendUsed: preflight.selectedBackend,
      outputFormat: artifact.format ?? request.params.output_format,
      warnings: artifact.warnings,
    };

    emit(completedProgressEvent(result));
    return result;
  } catch (error) {
    const normalized = normalizeRuntimeError(error);

    if (normalized.code === 'CANCELLED') {
      emit(cancelledProgressEvent(preflight.selectedBackend));
    } else {
      emit(terminalErrorProgressEvent(normalized, preflight.selectedBackend));
    }

    throw normalized;
  }
}

function selectAdapter(
  backend: UltraShapeExecutionRequest['backend'],
  options: UltraShapeRuntimeOptions,
): UltraShapeRuntimeAdapter {
  if (backend === 'local') {
    return options.localAdapter ?? new UltraShapeLocalAdapter();
  }

  if (!options.remoteClient) {
    return new UltraShapeRemoteAdapter({
      async execute() {
        throw new Error('Remote/hybrid backend is not configured for this request.');
      },
    }, backend);
  }

  return new UltraShapeRemoteAdapter(options.remoteClient, backend);
}

function packageArtifact(sourcePath: string, outputDir: string, outputFormat: UltraShapeRefinerResult['outputFormat']): string {
  mkdirSync(outputDir, { recursive: true });

  const destinationPath = resolve(join(outputDir, `refined.${outputFormat}`));
  const resolvedSource = resolve(sourcePath);

  if (resolvedSource !== destinationPath) {
    copyFileSync(resolvedSource, destinationPath);
  }

  return destinationPath;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new DOMException('Aborted', 'AbortError');
  }
}
