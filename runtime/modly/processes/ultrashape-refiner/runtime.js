const { copyFileSync, mkdirSync } = require('node:fs');
const { join, resolve } = require('node:path');

const { UltraShapeLocalAdapter } = require('../../adapters/ultrashape/local.js');
const { UltraShapeRemoteAdapter } = require('../../adapters/ultrashape/remote.js');
const {
  cancelledProgressEvent,
  completedProgressEvent,
  normalizeRuntimeError,
  packagingProgressEvent,
  runningProgressEvent,
  terminalErrorProgressEvent,
} = require('./progress.js');

async function runRefinerRuntime(request, preflight, options = {}) {
  const emit = options.onProgress || (() => undefined);
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
      artifact.format || request.params.output_format,
    );

    const result = {
      refinedMesh: {
        path: refinedMeshPath,
        kind: 'mesh',
      },
      backendUsed: preflight.selectedBackend,
      outputFormat: artifact.format || request.params.output_format,
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

function selectAdapter(backend, options) {
  if (backend === 'local') {
    return options.localAdapter || new UltraShapeLocalAdapter();
  }

  if (!options.remoteClient) {
    return new UltraShapeRemoteAdapter(
      {
        async execute() {
          throw new Error('Remote/hybrid backend is not configured for this request.');
        },
      },
      backend,
    );
  }

  return new UltraShapeRemoteAdapter(options.remoteClient, backend);
}

function packageArtifact(sourcePath, outputDir, outputFormat) {
  mkdirSync(outputDir, { recursive: true });

  const destinationPath = resolve(join(outputDir, `refined.${outputFormat}`));
  const resolvedSource = resolve(sourcePath);

  if (resolvedSource !== destinationPath) {
    copyFileSync(resolvedSource, destinationPath);
  }

  return destinationPath;
}

function throwIfAborted(signal) {
  if (signal && signal.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new DOMException('Aborted', 'AbortError');
  }
}

module.exports = {
  runRefinerRuntime,
};
