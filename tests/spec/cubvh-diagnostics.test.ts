import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const runtimeVendorPath = path.join(repoRoot, 'runtime/vendor');

function runDiagnosticsSnippet(source: string) {
  return spawnSync('python3', ['-c', source], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONPATH: [runtimeVendorPath, process.env.PYTHONPATH].filter(Boolean).join(':'),
    },
  });
}

describe('cubvh diagnostics classifier', () => {
  it('records CUDA unavailable reason when torch reports CUDA unavailable normally', () => {
    const result = runDiagnosticsSnippet(
      [
        'import json, types',
        'from ultrashape_runtime.cubvh_diagnostics import capture_metadata',
        'cubvh = types.SimpleNamespace(__file__="/tmp/cubvh.so", sparse_marching_cubes=lambda *args, **kwargs: None)',
        'cuda = types.SimpleNamespace(is_available=lambda: False)',
        'torch = types.SimpleNamespace(__version__="2.7.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)',
        'print(json.dumps(capture_metadata(cubvh, torch)))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      cuda_available: false,
      cuda_unavailable_reason: 'torch.cuda.is_available() returned false',
      torch_version: '2.7.0+cu128',
      torch_cuda_version: '12.8',
      cubvh_module_path: '/tmp/cubvh.so',
      cuda_runtime_hint: '12.8',
      cuda_driver_hint_unavailable_reason: expect.stringContaining('not exposed'),
    });
  });

  it('records available CUDA GPU metadata, arch list, build env, and driver/runtime hints', () => {
    const result = runDiagnosticsSnippet(
      [
        'import json, os, types',
        'from ultrashape_runtime.cubvh_diagnostics import capture_metadata',
        'os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1;12.1+PTX"',
        'os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"',
        'os.environ["CUDA_VERSION"] = "12.8"',
        'cubvh = types.SimpleNamespace(__file__="/tmp/cubvh_cuda.so", sparse_marching_cubes=lambda *args, **kwargs: None, sparse_marching_cubes_cpu=lambda *args, **kwargs: None)',
        'cuda = types.SimpleNamespace(',
        '    is_available=lambda: True,',
        '    current_device=lambda: 0,',
        '    get_device_name=lambda index: "NVIDIA GB10",',
        '    get_device_capability=lambda index: (12, 1),',
        '    get_arch_list=lambda: ["sm_121", "compute_121"],',
        '    driver_version=lambda: 58095,',
        ')',
        'torch = types.SimpleNamespace(__version__="2.7.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)',
        'print(json.dumps(capture_metadata(cubvh, torch)))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      cuda_available: true,
      device_name: 'NVIDIA GB10',
      device_capability: [12, 1],
      torch_arch_list: ['sm_121', 'compute_121'],
      env_torch_cuda_arch_list: '12.1;12.1+PTX',
      selected_cuda_home: '/usr/local/cuda-12.8',
      default_nvcc_cuda_version: '12.8',
      torch_cuda_version: '12.8',
      cuda_runtime_hint: '12.8',
      cuda_driver_hint: 58095,
      cubvh_callables: {
        sparse_marching_cubes: true,
        sparse_marching_cubes_cpu: true,
      },
    });
  });

  it('classifies import failures, missing CUDA callables, and tiny CUDA OOMs', () => {
    const result = runDiagnosticsSnippet(
      [
        'import json',
        'from ultrashape_runtime.cubvh_diagnostics import classify_error, rebuild_guidance',
        'tiny_input = {"cell_count": 316, "coords_shape": (316, 3), "corners_shape": (316, 8)}',
        'metadata = {"device_capability": (12, 1), "torch_cuda_version": "12.8", "env_torch_cuda_arch_list": "8.9", "selected_cuda_home": "/usr/local/cuda-12.8", "cubvh_module_path": "/tmp/cubvh.so"}',
        'tiny = classify_error("CUDA out of memory while launching sparse_marching_cubes", path="cuda", input_info=tiny_input, metadata=metadata)',
        'print(json.dumps({',
        '  "import_failure": classify_error("No module named cubvh", path="import"),',
        '  "missing_callable": classify_error("missing callable sparse_marching_cubes", path="cuda", input_info=tiny_input, metadata=metadata),',
        '  "tiny_oom": tiny,',
        '  "tiny_guidance": rebuild_guidance(tiny, metadata),',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      import_failure: 'import_failure',
      missing_callable: 'missing_callable',
      tiny_oom: 'cuda_oom_tiny_input_suspected_kernel_or_arch_mismatch',
      tiny_guidance: {
        recommended: true,
        reason: 'force_source_compile_or_match_torch_cuda_arch',
        fields: {
          device_capability: [12, 1],
          torch_cuda_version: '12.8',
          env_torch_cuda_arch_list: '8.9',
        },
      },
    });
  });

  it('separates CPU fallback unavailable/failure from suspected arch/toolkit mismatch', () => {
    const result = runDiagnosticsSnippet(
      [
        'import json',
        'from ultrashape_runtime.cubvh_diagnostics import classify_error, rebuild_guidance',
        'tiny_input = {"cell_count": 1, "coords_shape": (1, 3), "corners_shape": (1, 8)}',
        'metadata = {"device_capability": (12, 1), "torch_cuda_version": "12.8", "env_torch_cuda_arch_list": "8.9", "selected_cuda_home": "/usr/local/cuda-12.8", "cubvh_module_path": "/tmp/cubvh.so"}',
        'arch = classify_error("CUDA error: no kernel image is available for execution on the device", path="cuda", input_info=tiny_input, metadata=metadata)',
        'print(json.dumps({',
        '  "cpu_unavailable": classify_error("missing callable sparse_marching_cubes_cpu", path="cpu", input_info=tiny_input),',
        '  "cpu_failure": classify_error("CPU sparse marching cubes failed", path="cpu", input_info=tiny_input),',
        '  "arch_mismatch": arch,',
        '  "arch_guidance": rebuild_guidance(arch, metadata),',
        '}))',
      ].join('\n'),
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      cpu_unavailable: 'cpu_fallback_unavailable',
      cpu_failure: 'cpu_fallback_failure',
      arch_mismatch: 'suspected_arch_toolkit_mismatch',
      arch_guidance: {
        recommended: true,
        reason: 'force_source_compile_or_match_torch_cuda_arch',
      },
    });
  });
});
