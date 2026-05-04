#!/usr/bin/env python3
"""Test SageAttention reference compilation and correctness against PyTorch."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cuda_hercules.eval import load_task_def, _build_call_args
from cuda_hercules.compiler import compile_cuda_module

root = os.path.join(os.path.dirname(__file__), '..')
os.chdir(root)

tasks = [
    'sageattn_qk_int8_pv_fp16_hdim64',
    'sageattn_qk_int8_pv_fp16_hdim64_causal',
    'sageattn_qk_int8_pv_fp16_hdim128',
    'sageattn_qk_int8_pv_fp16_hdim128_causal',
]

for name in tasks:
    task_dir = f'tasks/class2/general/{name}'
    task_def = load_task_def(task_dir)
    build_dir = os.path.join(root, '.cache/ref_builds', name)
    os.makedirs(build_dir, exist_ok=True)

    extra_includes = [os.path.join(root, d) for d in task_def.get('REFERENCE_EXTRA_INCLUDES', [])]
    extra_flags = task_def.get('REFERENCE_EXTRA_CUDA_FLAGS', [])
    extra_ldflags = task_def.get('REFERENCE_EXTRA_LDFLAGS', [])

    try:
        ref_fn = compile_cuda_module(
            cuda_source=os.path.join(task_dir, 'reference.cu'),
            function_signature=task_def['FUNCTION_SIGNATURE'],
            extra_include_dirs=extra_includes,
            extra_cuda_cflags=extra_flags,
            extra_ldflags=extra_ldflags if extra_ldflags else None,
            build_dir=build_dir,
        )
    except Exception as e:
        print(f'  {name}: COMPILE FAILED - {e}')
        continue

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
    ok = torch.allclose(ref_data, py_data, atol=5e-2, rtol=5e-2)
    status = "OK" if ok else "MISMATCH"
    print(f'  {name}: {status} (max_diff={maxd:.6g})')
