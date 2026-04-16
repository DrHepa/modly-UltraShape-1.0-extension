import type {
  UltraShapeExecutionRequest,
  UltraShapeRuntimeAdapter,
} from './client.js';
import { createProcessError } from '../../processes/ultrashape-refiner/validate.js';

export class UltraShapeLocalAdapter implements UltraShapeRuntimeAdapter {
  readonly backend = 'local' as const;

  async run(_request: UltraShapeExecutionRequest): Promise<never> {
    throw createProcessError(
      'LOCAL_RUNTIME_UNAVAILABLE',
      'LOCAL_RUNTIME_UNAVAILABLE: The TypeScript compatibility adapter does not execute UltraShape locally; use the repo-root processor.py runtime boundary.',
      'backend',
    );
  }
}
