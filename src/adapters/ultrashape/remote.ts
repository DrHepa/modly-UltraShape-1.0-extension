import type {
  UltraShapeExecutionClient,
  UltraShapeExecutionRequest,
  UltraShapeRuntimeAdapter,
} from './client.js';
import { createProcessError } from '../../processes/ultrashape-refiner/validate.js';

export class UltraShapeRemoteAdapter implements UltraShapeRuntimeAdapter {
  readonly backend: 'remote' | 'hybrid';

  constructor(
    private readonly client: UltraShapeExecutionClient,
    backend: 'remote' | 'hybrid' = 'remote',
  ) {
    this.backend = backend;
  }

  async run(request: UltraShapeExecutionRequest) {
    if (request.backend !== 'remote' && request.backend !== 'hybrid') {
      throw createProcessError(
        'BACKEND_UNAVAILABLE',
        `Remote adapter cannot run backend ${request.backend}.`,
        'backend',
      );
    }

    return this.client.execute({
      ...request,
      backend: request.backend,
    });
  }
}
