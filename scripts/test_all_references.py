#!/usr/bin/env python3
"""Test all references one by one, writing results to CSV incrementally."""
import csv
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.eval import load_task_def, _build_call_args, check_correctness
from cuda_hercules.compiler import compile_cuda_module
from cuda_hercules.utils import get_project_root

import torch


def test_one_reference(task_dir):
    """Test a single reference. Returns (status, error_msg)."""
    root_dir = get_project_root()
    task_name = os.path.basename(task_dir)
    task_def = load_task_def(task_dir)
    sig = task_def["FUNCTION_SIGNATURE"]

    ref_cu_path = os.path.join(task_dir, "reference.cu")
    if not task_def["HAS_CUDA_REFERENCE"] or not os.path.exists(ref_cu_path):
        return "SKIP", "no reference.cu"

    # Compile with persistent cache
    extra_includes = []
    for inc in task_def["REFERENCE_EXTRA_INCLUDES"]:
        if os.path.isabs(inc):
            inc_path = inc
        else:
            inc_path = os.path.join(task_dir, inc)
            if not os.path.isdir(inc_path):
                inc_path = os.path.join(root_dir, inc)
        if os.path.isdir(inc_path):
            extra_includes.append(inc_path)

    build_dir = os.path.join(root_dir, ".cache", "ref_builds", task_name)
    os.makedirs(build_dir, exist_ok=True)

    ref_fn = compile_cuda_module(
        cuda_source=ref_cu_path,
        function_signature=sig,
        extra_include_dirs=extra_includes,
        extra_cuda_cflags=task_def.get("REFERENCE_EXTRA_CUDA_FLAGS", []),
        build_dir=build_dir,
        verbose=False,
    )

    # Run + correctness check vs PyTorch reference_fn
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    inputs = task_def["get_inputs"]()
    cuda_outputs = task_def["get_outputs"](inputs)
    args = _build_call_args(task_def, inputs, cuda_outputs)
    ref_fn(*args)
    torch.cuda.synchronize()

    ref_outputs = task_def["reference_fn"](inputs)
    correct, per_output = check_correctness(ref_outputs, cuda_outputs, task_def["TOLERANCES"])

    if correct:
        return "PASS", ""
    else:
        fails = [k for k, v in per_output.items() if not v]
        return "FAIL", f"mismatch on: {fails}"


def main():
    root = get_project_root()
    tasks_dir = os.path.join(root, "tasks")
    csv_path = os.path.join(root, "results", "reference_test_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Load already-tested tasks to support resuming
    tested = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tested[row["task"]] = row["status"]
        print(f"Resuming: {len(tested)} tasks already tested")

    # Discover all tasks (dirs containing def.py) recursively under tasks/
    all_task_paths = []
    for root, dirs, files in os.walk(tasks_dir):
        if "def.py" in files:
            task_name = os.path.basename(root)
            all_task_paths.append((task_name, root))
    all_task_paths.sort()
    total = len(all_task_paths)
    pass_count = sum(1 for v in tested.values() if v == "PASS")
    fail_count = sum(1 for v in tested.values() if v == "FAIL")

    # Open CSV in append mode if resuming, else write header
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csvfile = open(csv_path, "a", newline="")
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["task", "status", "error"])
        csvfile.flush()

    for i, (task_name, task_path) in enumerate(all_task_paths, 1):
        # Skip already tested
        if task_name in tested:
            print(f"[{i}/{total}] {task_name} ... {tested[task_name]} (cached)")
            continue

        print(f"[{i}/{total}] {task_name} ... ", end="", flush=True)
        try:
            status, error = test_one_reference(task_path)
        except Exception as e:
            status = "ERROR"
            error = str(e).replace("\n", " ")[:200]
            traceback.print_exc()

        print(status)
        writer.writerow([task_name, status, error])
        csvfile.flush()  # Flush after each row to prevent data loss

        if status == "PASS":
            pass_count += 1
        elif status in ("FAIL", "ERROR"):
            fail_count += 1

    csvfile.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY: {pass_count} PASS, {fail_count} FAIL/ERROR, {total} TOTAL")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
