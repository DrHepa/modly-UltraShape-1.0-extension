const { createProcessError } = require('../../processes/ultrashape-refiner/validate.js');

class UltraShapeLocalAdapter {
  constructor() {
    this.backend = 'local';
  }

  async run() {
    throw createProcessError(
      'BACKEND_UNAVAILABLE',
      'Local UltraShape execution is intentionally unsupported in this extension until a reliable runtime path exists.',
      'backend',
    );
  }
}

module.exports = {
  UltraShapeLocalAdapter,
};
