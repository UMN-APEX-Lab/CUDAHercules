#!/usr/bin/env python3
"""Test that reference code compiles and runs correctly."""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from cuda_hercules.eval import load_task_def, _build_call_args, check_correctness
from cuda_hercules.compiler import compile_cuda_module
from cuda_hercules.timing import time_cuda_function
from cuda_hercules.utils import get_project_root

def test_reference(task_dir):
    task_name = os.path.basename(task_dir)
    print(f"\n{'='*60}")
    print(f"Testing: {task_name}")
    print(f"{'='*60}")

    root_dir = get_project_root()
    task_def = load_task_def(task_dir)
    sig = task_def["FUNCTION_SIGNATURE"]

    # 1. Test PyTorch reference_fn
    print("\n[1] Testing PyTorch reference_fn...")
    torch.manual_seed(42)
    inputs = task_def["get_inputs"]()
    print(f"    Inputs: {[(n, t.shape, t.dtype) for n, t in inputs]}")

    outputs = task_def["get_outputs"](inputs)
    print(f"    Outputs: {[(n, t.shape, t.dtype) for n, t in outputs]}")

    ref_outputs = task_def["reference_fn"](inputs)
    print(f"    reference_fn OK")
    for name, t in ref_outputs:
        print(f"      {name}: mean={t.float().mean():.6f}, std={t.float().std():.6f}")

    # 2. Compile CUDA reference if available
    ref_cu_path = os.path.join(task_dir, "reference.cu")
    if not task_def["HAS_CUDA_REFERENCE"] or not os.path.exists(ref_cu_path):
        print(f"\n[2] No CUDA reference — skipping compilation test")
        return True

    print(f"\n[2] Compiling CUDA reference...")
    extra_includes = []
    for inc in task_def["REFERENCE_EXTRA_INCLUDES"]:
        if os.path.isabs(inc):
            inc_path = inc
        else:
            # Try relative to task dir first, then project root
            inc_path = os.path.join(task_dir, inc)
            if not os.path.isdir(inc_path):
                inc_path = os.path.join(root_dir, inc)
        if os.path.isdir(inc_path):
            extra_includes.append(inc_path)
        else:
            print(f"    WARNING: include dir not found: {inc}")
    print(f"    Include dirs: {extra_includes}")
    print(f"    Extra CUDA flags: {task_def.get('REFERENCE_EXTRA_CUDA_FLAGS', [])}")

    build_dir = os.path.join(root_dir, ".cache", "ref_builds", task_name)
    os.makedirs(build_dir, exist_ok=True)
    ref_fn = compile_cuda_module(
        cuda_source=ref_cu_path,
        function_signature=sig,
        extra_include_dirs=extra_includes,
        extra_cuda_cflags=task_def.get("REFERENCE_EXTRA_CUDA_FLAGS", []),
        build_dir=build_dir,
        verbose=True,
    )
    print(f"    Compilation OK")

    # 3. Run CUDA reference
    print(f"\n[3] Running CUDA reference...")
    torch.manual_seed(42)
    inputs = task_def["get_inputs"]()
    cuda_outputs = task_def["get_outputs"](inputs)
    args = _build_call_args(task_def, inputs, cuda_outputs)
    ref_fn(*args)
    torch.cuda.synchronize()
    print(f"    Execution OK")
    for name, t in cuda_outputs:
        print(f"      {name}: mean={t.float().mean():.6f}, std={t.float().std():.6f}")

    # 4. Check correctness vs PyTorch
    print(f"\n[4] Checking CUDA reference vs PyTorch...")
    ref_outputs = task_def["reference_fn"](inputs)
    correct, per_output = check_correctness(ref_outputs, cuda_outputs, task_def["TOLERANCES"])
    for name, ok in per_output.items():
        print(f"      {name}: {'PASS' if ok else 'FAIL'}")
    print(f"    Overall: {'PASS' if correct else 'FAIL'}")

    # 5. Performance
    print(f"\n[5] Timing CUDA reference (10 trials)...")
    stats = time_cuda_function(ref_fn, args, num_trials=10, verbose=False)
    print(f"    {stats.mean_ms:.4f} ms (std={stats.std_ms:.4f}, min={stats.min_ms:.4f}, max={stats.max_ms:.4f})")

    return correct


if __name__ == "__main__":
    root = get_project_root()
    tasks_to_test = sys.argv[1:] if len(sys.argv) > 1 else []

    if not tasks_to_test:
        # Test all tasks that have CUDA references (recursive scan)
        tasks_dir = os.path.join(root, "tasks")
        for dirpath, dirs, files in os.walk(tasks_dir):
            if "reference.cu" in files and "def.py" in files:
                tasks_to_test.append(dirpath)
        tasks_to_test.sort()

    results = {}
    for task_dir in tasks_to_test:
        task_dir = os.path.abspath(task_dir)
        try:
            ok = test_reference(task_dir)
            results[os.path.basename(task_dir)] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            results[os.path.basename(task_dir)] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")
