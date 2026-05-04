#!/usr/bin/env python3
"""
Evaluate an LLM-generated solution for the ExaChem CCSD(T) kernel task.

Usage:
    python eval_solution.py <path/to/solution_dir_or_file>

If a directory is given, it should contain solution.cu (and optionally
ccsd_t_g2s_device_functions.cu and tensor_core_helper.cuh).
If a single file is given, it is used as solution.cu.

Creates a temp copy of the task, replaces solution files, runs the benchmark,
and prints a JSON result.
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
SOLUTION_FILES = [
    "solution.cu",
    "ccsd_t_g2s_device_functions.cu",
    "tensor_core_helper.cuh",
]

IGNORE_PATTERNS = shutil.ignore_patterns(
    "build", "cmake-build-*", "__pycache__",
    "*.o", "*.so", "*.a", "ref_benchmark", "sol_benchmark",
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ExaChem CCSD(T) solution")
    parser.add_argument("solution", help="Path to solution directory or solution.cu file")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Execution timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Do not delete temp directory after evaluation")
    args = parser.parse_args()

    solution_path = os.path.abspath(args.solution)

    # Determine solution files
    sol_files = {}
    if os.path.isdir(solution_path):
        for f in SOLUTION_FILES:
            p = os.path.join(solution_path, f)
            if os.path.isfile(p):
                sol_files[f] = p
    elif os.path.isfile(solution_path):
        sol_files["solution.cu"] = solution_path
    else:
        print(json.dumps({"compiled": False, "correct": False,
                          "kernel_time_ms": -1,
                          "output": f"Solution not found: {solution_path}"}))
        sys.exit(1)

    if "solution.cu" not in sol_files:
        print(json.dumps({"compiled": False, "correct": False,
                          "kernel_time_ms": -1,
                          "output": "solution.cu not found in provided path"}))
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="kh_exachem_eval_")
    task_copy = os.path.join(tmp_dir, "task")

    try:
        # 1. Copy task directory
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        # 2. Replace solution files
        for fname, src_path in sol_files.items():
            dst = os.path.join(task_copy, fname)
            shutil.copy2(src_path, dst)

        # 3. Run via run.py
        run_py = os.path.join(task_copy, "run.py")
        env = os.environ.copy()

        try:
            proc = subprocess.run(
                [sys.executable, run_py],
                capture_output=True, text=True,
                cwd=task_copy,
                env=env, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print(json.dumps({
                "compiled": True, "correct": False,
                "kernel_time_ms": -1,
                "output": f"Execution timed out ({args.timeout}s)",
            }))
            sys.exit(1)

        output = proc.stdout + proc.stderr

        # 4. Parse results
        compiled = True
        if "Build failed" in output and proc.returncode != 0:
            compiled = False

        correct = "Passed" in output and proc.returncode == 0

        m = re.search(r"Kernel time:\s*([\d.]+)\s*ms", output)
        kernel_time_ms = float(m.group(1)) if m else -1
        m = re.search(r"Ref time:\s*([\d.]+)\s*ms", output)
        ref_time_ms = float(m.group(1)) if m else -1
        m = re.search(r"Speedup:\s*([\d.]+)\s*x", output)
        speedup = float(m.group(1)) if m else -1

        m = re.search(r"Energy_T:\s*([-+\d.eE]+)", output)
        energy_t = float(m.group(1)) if m else None
        m = re.search(r"Energy_T5:\s*([-+\d.eE]+)", output)
        energy_t5 = float(m.group(1)) if m else None

        result = {
            "compiled": compiled,
            "correct": correct,
            "kernel_time_ms": kernel_time_ms,
            "ref_time_ms": ref_time_ms,
            "speedup": speedup,
            "energy_t": energy_t,
            "energy_t5": energy_t5,
            "output": output[-3000:],
        }

        print(json.dumps(result, indent=2))
        sys.exit(0 if correct else 1)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"Temp directory kept at: {tmp_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
