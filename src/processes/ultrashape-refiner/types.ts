export type UltraShapeAssetKind = 'image' | 'mesh';
export type UltraShapeBackendMode = 'auto' | 'local';
export type UltraShapeOutputFormat = 'glb';
export type UltraShapeProgressStage =
  | 'validating'
  | 'preflight'
  | 'running'
  | 'packaging'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface UltraShapeAssetRef {
  path: string;
  kind: UltraShapeAssetKind;
  mediaType?: string;
}

export type UltraShapeAssetInput = string | UltraShapeAssetRef;

export interface UltraShapeRefinerParams {
  checkpoint: string | null;
  backend: UltraShapeBackendMode;
  steps: number;
  guidance_scale: number;
  seed: number | null;
  preserve_scale: boolean;
  output_format: UltraShapeOutputFormat;
}

export interface UltraShapeFallbackBundle {
  reference_image: string;
  coarse_mesh: string;
  output_dir: string;
  checkpoint?: string | null;
  params?: Partial<UltraShapeRefinerParams>;
}

/** Temporary compatibility-only seam for legacy callers. */
export type UltraShapeCompatibilityFallbackBundle = UltraShapeFallbackBundle;

export interface UltraShapeNativeRequest {
  reference_image: UltraShapeAssetInput;
  coarse_mesh: UltraShapeAssetInput;
  output_dir: string;
  checkpoint?: string | null;
  params?: Partial<UltraShapeRefinerParams>;
  correlation_id?: string;
  abortSignal?: AbortSignal;
}

export type UltraShapeRequestInput = UltraShapeNativeRequest | UltraShapeCompatibilityFallbackBundle;

export interface UltraShapeNormalizedRequest {
  correlationId: string;
  referenceImage: UltraShapeAssetRef;
  coarseMesh: UltraShapeAssetRef;
  outputDir: string;
  params: UltraShapeRefinerParams;
  abortSignal?: AbortSignal;
  requestedBackend?: UltraShapeBackendMode;
}

export interface UltraShapeRefinerResult {
  refinedMesh: UltraShapeAssetRef;
  backendUsed: Exclude<UltraShapeBackendMode, 'auto'>;
  outputFormat: UltraShapeOutputFormat;
  warnings?: string[];
}

export interface UltraShapeProgressEvent {
  stage: UltraShapeProgressStage;
  message: string;
  progress?: number;
  backend?: Exclude<UltraShapeBackendMode, 'auto'>;
}

export interface UltraShapeRuntimeCapabilities {
  hostPlatform?: NodeJS.Platform;
  hostArch?: string;
  localSupported: boolean;
  recommendedBackend: 'local';
  reason?: string;
}

export interface UltraShapePreflightResult extends UltraShapeRuntimeCapabilities {
  selectedBackend: 'local';
  fallbackApplied: boolean;
}

export type UltraShapeErrorCode =
  | 'MISSING_INPUT'
  | 'UNREADABLE_ASSET'
  | 'UNSUPPORTED_ASSET_TYPE'
  | 'INVALID_PARAMS'
  | 'DEPENDENCY_MISSING'
  | 'WEIGHTS_MISSING'
  | 'LOCAL_RUNTIME_UNAVAILABLE'
  | 'CANCELLED';

export interface UltraShapeProcessError extends Error {
  code: UltraShapeErrorCode;
  field?: 'reference_image' | 'coarse_mesh' | 'output_dir' | keyof UltraShapeRefinerParams;
  recoverable?: boolean;
}
