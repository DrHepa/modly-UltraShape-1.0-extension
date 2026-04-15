const { createProcessError } = require('../../processes/ultrashape-refiner/validate.js');

class UltraShapeRemoteAdapter {
  constructor(client, backend = 'remote') {
    this.client = client;
    this.backend = backend;
  }

  async run(request) {
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

module.exports = {
  UltraShapeRemoteAdapter,
};
