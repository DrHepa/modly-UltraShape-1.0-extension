import type {
  UltraShapeExecutionRequest,
  UltraShapeRuntimeAdapter,
} from './client.js';
import { createProcessError } from '../../processes/ultrashape-refiner/validate.js';

export class UltraShapeLocalAdapter implements UltraShapeRuntimeAdapter {
  readonly backend = 'local' as const;

  async run(_request: UltraShapeExecutionRequest): Promise<never> {
    throw createProcessError(
      'BACKEND_UNAVAILABLE',
      'Local UltraShape execution is intentionally unsupported in this extension until a reliable runtime path exists.',
      'backend',
    );
  }
}
