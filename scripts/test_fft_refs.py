#!/usr/bin/env python3
"""Test FFT reference compilation and correctness against PyTorch."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cuda_hercules.eval import load_task_def, _build_call_args
from cuda_hercules.compiler import compile_cuda_module

root = os.path.join(os.path.dirname(__file__), '..')
os.chdir(root)

tasks = [
    'fft_c2c_1d_1024_fp32',
    'fft_c2c_1d_4096_fp32',
    'fft_c2c_1d_16384_fp32',
    'fft_c2c_1d_65536_fp32',
    'fft_c2c_1d_262144_fp32',
    'fft_c2c_1d_1024_fp64',
    'fft_c2c_1d_4096_fp64',
    'fft_c2c_1d_16384_fp64',
    'fft_c2c_1d_65536_fp64',
    'fft_c2c_2d_512x512_fp32',
    'fft_c2c_2d_1024x1024_fp32',
    'fft_c2c_3d_64x64x64_fp32',
    'fft_c2c_3d_128x128x128_fp32',
    'fft_r2c_1d_1024_fp32',
    'fft_r2c_1d_4096_fp32',
    'fft_r2c_1d_16384_fp32',
    'fft_r2c_1d_65536_fp32',
]

for name in tasks:
    task_dir = f'tasks/class2/general/{name}'
    task_def = load_task_def(task_dir)
    build_dir = os.path.join(root, '.cache/ref_builds', name)
    os.makedirs(build_dir, exist_ok=True)

    try:
        ref_fn = compile_cuda_module(
            cuda_source=os.path.join(task_dir, 'reference.cu'),
            function_signature=task_def['FUNCTION_SIGNATURE'],
            extra_include_dirs=[],
            extra_ldflags=['-lcufft'],
            build_dir=build_dir,
        )
    except Exception as e:
        print(f'  {name}: COMPILE FAILED - {e}')
        continue

    torch.manual_seed(42); torch.cuda.manual_seed(42)
    inputs = task_def['get_inputs']()

    ref_outputs = task_def['get_outputs'](inputs)
    ref_args = _build_call_args(task_def, inputs, ref_outputs)
    ref_fn(*ref_args)
    torch.cuda.synchronize()

    py_outputs = task_def['reference_fn'](inputs)

    ref_data = ref_outputs[0][1].float()
    py_data = py_outputs[0][1].float()
    diff = (ref_data - py_data).abs()
    maxd = diff.max().item()
    ok = torch.allclose(ref_data, py_data, atol=1e-2, rtol=1e-2)
    status = "OK" if ok else "MISMATCH"
    print(f'  {name}: {status} (max_diff={maxd:.6g})')
