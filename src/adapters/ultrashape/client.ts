import type {
  UltraShapeNormalizedRequest,
  UltraShapeOutputFormat,
} from '../../processes/ultrashape-refiner/types.js';

export type UltraShapeAdapterBackend = 'local' | 'remote' | 'hybrid';

export interface UltraShapeAdapterUpdate {
  stage: 'queued' | 'running';
  message: string;
  progress?: number;
}

export interface UltraShapeExecutionArtifact {
  path: string;
  format?: UltraShapeOutputFormat;
  warnings?: string[];
}

export interface UltraShapeExecutionRequest {
  request: UltraShapeNormalizedRequest;
  backend: UltraShapeAdapterBackend;
  /** `compatibility-fallback` only exists for temporary legacy payload support. */
  contract?: 'named-inputs' | 'compatibility-fallback';
  signal?: AbortSignal;
  onUpdate?: (update: UltraShapeAdapterUpdate) => void;
}

export interface UltraShapeExecutionClient {
  execute(request: UltraShapeExecutionRequest): Promise<UltraShapeExecutionArtifact>;
}

export interface UltraShapeRuntimeAdapter {
  readonly backend: UltraShapeAdapterBackend;
  run(request: UltraShapeExecutionRequest): Promise<UltraShapeExecutionArtifact>;
}
