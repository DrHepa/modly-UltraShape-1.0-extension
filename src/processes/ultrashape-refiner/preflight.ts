import type {
  UltraShapeBackendMode,
  UltraShapeOutputFormat,
  UltraShapePreflightResult,
  UltraShapeRuntimeCapabilities,
} from './types.js';
import { createProcessError } from './validate.js';

export interface UltraShapePreflightOptions {
  hostPlatform?: NodeJS.Platform;
  hostArch?: string;
  localAvailable?: boolean;
  requestedOutputFormat?: UltraShapeOutputFormat | 'obj' | 'fbx' | 'ply';
}

export function detectRuntimeCapabilities(
  options: UltraShapePreflightOptions = {},
): UltraShapeRuntimeCapabilities {
  const hostPlatform = options.hostPlatform ?? process.platform;
  const hostArch = options.hostArch ?? process.arch;
  const localSupported = options.localAvailable ?? true;
  const recommendedBackend = 'local';

  let reason: string | undefined;
  if (localSupported) {
    reason = 'UltraShape local runtime is the only active backend in this MVP.';
  } else {
    reason = 'LOCAL_RUNTIME_UNAVAILABLE: UltraShape local runtime is not available for this host/request.';
  }

  return {
    hostPlatform,
    hostArch,
    localSupported,
    recommendedBackend,
    reason,
  };
}

export function preflightRefinerExecution(
  requestedBackend: UltraShapeBackendMode | 'remote' | 'hybrid',
  options: UltraShapePreflightOptions = {},
): UltraShapePreflightResult {
  const capabilities = detectRuntimeCapabilities(options);
  ensureMvpOutputFormat(options.requestedOutputFormat);
  const selection = selectBackend(requestedBackend, capabilities);

  return {
    ...capabilities,
    selectedBackend: selection.selectedBackend,
    fallbackApplied: selection.fallbackApplied,
    reason: selection.reason ?? capabilities.reason,
  };
}

function selectBackend(
  requestedBackend: UltraShapeBackendMode | 'remote' | 'hybrid',
  capabilities: UltraShapeRuntimeCapabilities,
): Pick<UltraShapePreflightResult, 'selectedBackend' | 'fallbackApplied' | 'reason'> {
  switch (requestedBackend) {
    case 'local':
      if (capabilities.localSupported) {
        return {
          selectedBackend: 'local',
          fallbackApplied: false,
          reason: capabilities.reason,
        };
      }

      throw createProcessError(
        'LOCAL_RUNTIME_UNAVAILABLE',
        'LOCAL_RUNTIME_UNAVAILABLE: Requested local backend is unavailable.',
        'backend',
      );

    case 'remote':
    case 'hybrid':
      throw createProcessError(
        'LOCAL_RUNTIME_UNAVAILABLE',
        'LOCAL_RUNTIME_UNAVAILABLE: UltraShape local execution does not support deferred remote or hybrid backends in this MVP.',
        'backend',
      );

    case 'auto':
    default:
      if (capabilities.localSupported) {
        return {
          selectedBackend: 'local',
          fallbackApplied: false,
          reason: capabilities.reason,
        };
      }

      throw createProcessError(
        'LOCAL_RUNTIME_UNAVAILABLE',
        'LOCAL_RUNTIME_UNAVAILABLE: UltraShape local runtime is unavailable for this host/request.',
        'backend',
      );
  }
}

function ensureMvpOutputFormat(requestedOutputFormat?: UltraShapePreflightOptions['requestedOutputFormat']): void {
  if (!requestedOutputFormat || requestedOutputFormat === 'glb') {
    return;
  }

  throw createProcessError(
    'LOCAL_RUNTIME_UNAVAILABLE',
    'LOCAL_RUNTIME_UNAVAILABLE: UltraShape local execution is glb-only in this MVP.',
    'output_format',
  );
}
