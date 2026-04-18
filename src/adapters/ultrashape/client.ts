import type {
  UltraShapeNormalizedRequest,
  UltraShapeOutputFormat,
} from '../../processes/ultrashape-refiner/types.js';

export type UltraShapeAdapterBackend = 'local';

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
  signal?: AbortSignal;
  onUpdate?: (update: UltraShapeAdapterUpdate) => void;
}

export interface UltraShapeRuntimeAdapter {
  readonly backend: UltraShapeAdapterBackend;
  run(request: UltraShapeExecutionRequest): Promise<UltraShapeExecutionArtifact>;
}
