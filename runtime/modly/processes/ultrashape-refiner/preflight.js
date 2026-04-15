const { createProcessError } = require('./validate.js');

function detectRuntimeCapabilities(options = {}) {
  const hostPlatform = options.hostPlatform ?? process.platform;
  const hostArch = options.hostArch ?? process.arch;
  const localSupported = options.localAvailable ?? !isLinuxArm64(hostPlatform, hostArch);
  const remoteSupported = options.remoteAvailable ?? true;
  const recommendedBackend =
    isLinuxArm64(hostPlatform, hostArch) && remoteSupported
      ? 'remote'
      : localSupported
        ? 'local'
        : remoteSupported
          ? 'remote'
          : 'local';

  let reason;
  if (isLinuxArm64(hostPlatform, hostArch) && remoteSupported) {
    reason = 'Linux ARM64 prefers remote/hybrid execution because local UltraShape support is not a reliable first target.';
  } else if (!localSupported && remoteSupported) {
    reason = 'Local backend is unavailable; remote execution is eligible.';
  } else if (!localSupported && !remoteSupported) {
    reason = 'Neither local nor remote backend is available.';
  }

  return {
    hostPlatform,
    hostArch,
    localSupported,
    remoteSupported,
    recommendedBackend,
    reason,
  };
}

function preflightRefinerExecution(requestedBackend, options = {}) {
  const capabilities = detectRuntimeCapabilities(options);
  const selection = selectBackend(requestedBackend, capabilities);

  return {
    ...capabilities,
    selectedBackend: selection.selectedBackend,
    fallbackApplied: selection.fallbackApplied,
    reason: selection.reason ?? capabilities.reason,
  };
}

function selectBackend(requestedBackend, capabilities) {
  switch (requestedBackend) {
    case 'remote':
      ensureRemote(capabilities);
      return {
        selectedBackend: 'remote',
        fallbackApplied: false,
        reason: capabilities.reason,
      };

    case 'hybrid':
      ensureRemote(capabilities);
      return {
        selectedBackend: 'hybrid',
        fallbackApplied: false,
        reason: capabilities.reason,
      };

    case 'local':
      if (capabilities.localSupported) {
        return {
          selectedBackend: 'local',
          fallbackApplied: false,
          reason: capabilities.reason,
        };
      }

      if (capabilities.remoteSupported) {
        return {
          selectedBackend: capabilities.recommendedBackend,
          fallbackApplied: true,
          reason: 'Requested local backend is unavailable; falling back to an eligible remote/hybrid path.',
        };
      }

      throw createProcessError(
        'BACKEND_UNAVAILABLE',
        'Requested local backend is unavailable and no remote fallback is configured.',
        'backend',
      );

    case 'auto':
    default:
      if (capabilities.recommendedBackend === 'local' && capabilities.localSupported) {
        return {
          selectedBackend: 'local',
          fallbackApplied: false,
          reason: capabilities.reason,
        };
      }

      if (capabilities.remoteSupported) {
        return {
          selectedBackend: capabilities.recommendedBackend,
          fallbackApplied: capabilities.recommendedBackend !== 'local',
          reason: capabilities.reason,
        };
      }

      if (capabilities.localSupported) {
        return {
          selectedBackend: 'local',
          fallbackApplied: false,
          reason: capabilities.reason,
        };
      }

      throw createProcessError(
        'BACKEND_UNAVAILABLE',
        'No eligible UltraShape backend is available for this host.',
        'backend',
      );
  }
}

function ensureRemote(capabilities) {
  if (!capabilities.remoteSupported) {
    throw createProcessError(
      'BACKEND_UNAVAILABLE',
      'Remote/hybrid backend is unavailable for this request.',
      'backend',
    );
  }
}

function isLinuxArm64(platform, arch) {
  return platform === 'linux' && arch === 'arm64';
}

module.exports = {
  detectRuntimeCapabilities,
  preflightRefinerExecution,
};
