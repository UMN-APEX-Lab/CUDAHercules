#!/usr/bin/env python3
"""
Evaluate LLM-generated solution for the MGG multi-GPU GCN task.

Usage:
    python eval_solution.py <path/to/neighbor_utils.cuh>
    python eval_solution.py <path/to/solution_dir>   # dir containing src/include/neighbor_utils.cuh
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_REL_PATH = "src/include/neighbor_utils.cuh"

IGNORE_PATTERNS = shutil.ignore_patterns(
    "build", "cmake-build-*", "__pycache__", "*.o", "*.so", "*.a",
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MGG GCN solution")
    parser.add_argument("solution", help="Path to neighbor_utils.cuh or directory")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--skip-ref", action="store_true",
                        help="Skip reference run")
    args = parser.parse_args()

    sol_path = os.path.abspath(args.solution)

    tmp_dir = tempfile.mkdtemp(prefix="kh_mgg_eval_")
    task_copy = os.path.join(tmp_dir, "task")

    try:
        # 1. Copy task
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        env = os.environ.copy()

        # ── Phase 1: Reference run ──
        ref_time_ms = -1
        if not args.skip_ref:
            print("=== Phase 1: Building and running REFERENCE ===", flush=True)
            run_py = os.path.join(task_copy, "run.py")
            proc = subprocess.run(
                [sys.executable, run_py],
                capture_output=True, text=True,
                cwd=task_copy, env=env, timeout=args.timeout)
            output = proc.stdout + proc.stderr
            m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
            ref_time_ms = float(m.group(1)) if m else -1
            print(f"Reference Kernel time: {ref_time_ms:.4f} ms", flush=True)

        # ── Phase 2: Replace solution ──
        print("\n=== Phase 2: Building and running SOLUTION ===", flush=True)

        # Find and replace solution file
        if os.path.isdir(sol_path):
            src = os.path.join(sol_path, SOLUTION_REL_PATH)
            if not os.path.isfile(src):
                src = os.path.join(sol_path, "neighbor_utils.cuh")
        else:
            src = sol_path

        if not os.path.isfile(src):
            print(json.dumps({"compiled": False, "correct": False,
                              "kernel_time_ms": -1, "ref_time_ms": ref_time_ms,
                              "output": f"Solution file not found: {src}"}))
            sys.exit(1)

        dst = os.path.join(task_copy, SOLUTION_REL_PATH)
        shutil.copy2(src, dst)

        # Remove build dir to force rebuild
        build_dir = os.path.join(task_copy, "src", "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)

        # Run solution
        run_py = os.path.join(task_copy, "run.py")
        try:
            proc = subprocess.run(
                [sys.executable, run_py],
                capture_output=True, text=True,
                cwd=task_copy, env=env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(json.dumps({"compiled": True, "correct": False,
                              "kernel_time_ms": -1, "ref_time_ms": ref_time_ms,
                              "output": f"Timeout ({args.timeout}s)"}))
            sys.exit(1)

        output = proc.stdout + proc.stderr
        correct = "Passed" in output and proc.returncode == 0

        # Check compilation
        compiled = True
        if "Build failed" in output or "CMake failed" in output:
            compiled = False

        m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
        kernel_time_ms = float(m.group(1)) if m else -1

        speedup = ref_time_ms / kernel_time_ms if ref_time_ms > 0 and kernel_time_ms > 0 else -1

        result = {
            "compiled": compiled,
            "correct": correct,
            "kernel_time_ms": kernel_time_ms,
            "ref_time_ms": ref_time_ms,
            "speedup": round(speedup, 4) if speedup > 0 else -1,
            "output": output[-3000:],
        }

        print(json.dumps(result, indent=2))
        sys.exit(0 if correct else 1)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"Temp directory: {tmp_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
