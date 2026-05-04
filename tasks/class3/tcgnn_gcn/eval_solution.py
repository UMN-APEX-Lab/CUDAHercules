#!/usr/bin/env python3
"""
Evaluate an LLM-generated solution for the TC-GNN GCN training task.

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

IGNORE_PATTERNS = shutil.ignore_patterns(
    "build", "cmake-build-release", "cmake-build-*", "__pycache__",
    "*.o", "*.so", "*.a",
)


def _extract_run_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith("RUN_SUMMARY_JSON "):
            payload = line[len("RUN_SUMMARY_JSON "):].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TC-GNN GCN solution")
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

    tmp_dir = tempfile.mkdtemp(prefix="kh_tcgnn_eval_")
    task_copy = os.path.join(tmp_dir, "task")

    try:
        # 1. Copy task directory
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        # 2. Replace solution file(s)
        for fname, src_path in sol_files.items():
            dst = os.path.join(task_copy, fname)
            shutil.copy2(src_path, dst)

        # 3. Run via run.py (it handles building via torch cpp_extension)
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
        # Check if compilation failed (look for common build error indicators)
        compiled = True
        if ("error:" in output.lower() and "Build" in output
                and proc.returncode != 0):
            compiled = False

        summary = _extract_run_summary(output)
        aggregate = summary.get("aggregate", {}) if isinstance(summary, dict) else {}
        correct = bool(summary.get("correct", False)) and proc.returncode == 0 if summary else (
            "Passed" in output and proc.returncode == 0
        )

        # Performance
        if aggregate:
            kernel_time_ms = float(aggregate.get("kernel_time_ms", -1))
            ref_time_ms = float(aggregate.get("ref_time_ms", -1))
            speedup = float(aggregate.get("speedup", -1))
            ref_loss_payload = aggregate.get("ref_loss", {})
            sol_loss_payload = aggregate.get("solution_loss", {})
            ref_loss_first = float(ref_loss_payload.get("first", -1))
            ref_loss_last = float(ref_loss_payload.get("last", -1))
            sol_loss_first = float(sol_loss_payload.get("first", -1))
            sol_loss_last = float(sol_loss_payload.get("last", -1))
            loss_ratio = float(aggregate.get("loss_ratio", -1))
            ref_ckpts = ref_loss_payload.get("checkpoints", {})
            sol_ckpts = sol_loss_payload.get("checkpoints", {})
        else:
            m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
            kernel_time_ms = float(m.group(1)) if m else -1
            m = re.search(r"Ref time:\s*([0-9.]+)\s*ms", output)
            ref_time_ms = float(m.group(1)) if m else -1
            m = re.search(r"Speedup:\s*([0-9.]+)x", output)
            speedup = float(m.group(1)) if m else -1

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

            ref_ckpts = parse_checkpoints("Ref loss checkpoints:")
            sol_ckpts = parse_checkpoints("Solution loss checkpoints:")

        result = {
            "compiled": compiled,
            "correct": correct,
            "kernel_time_ms": kernel_time_ms,
            "ref_time_ms": ref_time_ms,
            "speedup": speedup,
            "ref_loss": {"first": ref_loss_first, "last": ref_loss_last,
                         "checkpoints": ref_ckpts},
            "solution_loss": {"first": sol_loss_first, "last": sol_loss_last,
                              "checkpoints": sol_ckpts},
            "loss_ratio": loss_ratio,
            "graphs": summary.get("graphs", []) if summary else [],
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
