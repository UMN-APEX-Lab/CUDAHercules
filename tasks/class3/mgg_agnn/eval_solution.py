#!/usr/bin/env python3
"""
Evaluate an LLM-generated solution for the MGG AGNN training task.

Usage:
    python eval_solution.py <path/to/solution.cu>

Creates a temp copy of the task, replaces solution.cu, runs the benchmark
(which builds via PyTorch cpp_extension), and prints a JSON result.
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
SOLUTION_FILES = ["solution.cu", "wrapper.cpp"]
GRAPH_DATA_SRC = os.path.join(TASK_DIR, "..", "tcgnn_gcn", "data", "amazon0505.npz")
GRAPH_DATA_DST = os.path.join("..", "tcgnn_gcn", "data", "amazon0505.npz")

IGNORE_PATTERNS = shutil.ignore_patterns(
    "build", "cmake-build-release", "cmake-build-*", "__pycache__",
    "*.o", "*.so", "*.a",
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MGG AGNN solution")
    parser.add_argument("solution", help="Path to solution.cu or directory containing solution files")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Execution timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Do not delete temp directory after evaluation")
    args = parser.parse_args()

    solution_path = os.path.abspath(args.solution)
    sol_files = {}
    if os.path.isdir(solution_path):
        for fname in SOLUTION_FILES:
            candidate = os.path.join(solution_path, fname)
            if os.path.isfile(candidate):
                sol_files[fname] = candidate
    elif os.path.isfile(solution_path):
        sol_files["solution.cu"] = solution_path
    else:
        print(json.dumps({"compiled": False, "correct": False,
                          "kernel_time_ms": -1,
                          "output": f"Solution path not found: {solution_path}"}))
        sys.exit(1)

    if "solution.cu" not in sol_files:
        print(json.dumps({"compiled": False, "correct": False,
                          "kernel_time_ms": -1,
                          "output": "solution.cu not found in provided path"}))
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="kh_mgg_agnn_eval_")
    task_copy = os.path.join(tmp_dir, "task")

    try:
        # 1. Copy task directory
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        # 2. Replace solution file(s)
        for fname, src_path in sol_files.items():
            dst = os.path.join(task_copy, fname)
            shutil.copy2(src_path, dst)

        # 3. Recreate the sibling graph dataset path expected by run.py.
        if not os.path.isfile(GRAPH_DATA_SRC):
            print(json.dumps({
                "compiled": False, "correct": False,
                "kernel_time_ms": -1,
                "output": f"Graph data not found: {GRAPH_DATA_SRC}",
            }))
            sys.exit(1)
        graph_dst = os.path.join(task_copy, GRAPH_DATA_DST)
        os.makedirs(os.path.dirname(graph_dst), exist_ok=True)
        if not os.path.exists(graph_dst):
            os.symlink(GRAPH_DATA_SRC, graph_dst)

        # 4. Run via run.py (it handles building via torch cpp_extension)
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

        # 5. Parse results
        # Check if compilation failed (look for common build error indicators)
        compiled = True
        if ("error:" in output.lower() and "Build" in output
                and proc.returncode != 0):
            compiled = False

        correct = "Passed" in output and proc.returncode == 0

        # Performance
        m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
        kernel_time_ms = float(m.group(1)) if m else -1
        m = re.search(r"Ref time:\s*([0-9.]+)\s*ms", output)
        ref_time_ms = float(m.group(1)) if m else -1
        m = re.search(r"Speedup:\s*([0-9.]+)x", output)
        speedup = float(m.group(1)) if m else -1

        # Loss metrics
        m = re.search(r"Ref loss:\s*([0-9.]+)\s*->\s*([0-9.]+)", output)
        ref_loss_first = float(m.group(1)) if m else -1
        ref_loss_last = float(m.group(2)) if m else -1
        m = re.search(r"Solution loss:\s*([0-9.]+)\s*->\s*([0-9.]+)", output)
        sol_loss_first = float(m.group(1)) if m else -1
        sol_loss_last = float(m.group(2)) if m else -1
        m = re.search(r"Loss ratio:\s*([0-9.]+)", output)
        loss_ratio = float(m.group(1)) if m else -1

        # Loss checkpoints
        def parse_checkpoints(prefix):
            m = re.search(prefix + r".*?25%=([0-9.]+).*?50%=([0-9.]+).*?75%=([0-9.]+).*?100%=([0-9.]+)", output)
            if m:
                return {"25%": float(m.group(1)), "50%": float(m.group(2)),
                        "75%": float(m.group(3)), "100%": float(m.group(4))}
            return {}

        result = {
            "compiled": compiled,
            "correct": correct,
            "kernel_time_ms": kernel_time_ms,
            "ref_time_ms": ref_time_ms,
            "speedup": speedup,
            "ref_loss": {"first": ref_loss_first, "last": ref_loss_last,
                         "checkpoints": parse_checkpoints("Ref loss checkpoints:")},
            "solution_loss": {"first": sol_loss_first, "last": sol_loss_last,
                              "checkpoints": parse_checkpoints("Solution loss checkpoints:")},
            "loss_ratio": loss_ratio,
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
