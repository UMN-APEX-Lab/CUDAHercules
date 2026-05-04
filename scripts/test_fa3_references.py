#!/usr/bin/env python3
"""Test all FA3 references, each in a subprocess to survive CUDA crashes."""
import csv
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FA3_DIR = os.path.join(PROJECT_ROOT, "tasks", "class2", "hopper")
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "fa3_test_results.csv")

# Single-task test script run as subprocess
TEST_SCRIPT = '''
import os, sys, traceback
sys.path.insert(0, os.path.join("{root}", "src"))
import torch
from cuda_hercules.eval import load_task_def, _build_call_args, check_correctness
from cuda_hercules.compiler import compile_cuda_module

task_path = "{task_path}"
task_name = "{task_name}"
root = "{root}"

try:
    task_def = load_task_def(task_path)
    sig = task_def["FUNCTION_SIGNATURE"]
    ref_cu = os.path.join(task_path, "reference.cu")
    extra_includes = []
    for inc in task_def["REFERENCE_EXTRA_INCLUDES"]:
        inc_path = os.path.join(task_path, inc)
        if not os.path.isdir(inc_path):
            inc_path = os.path.join(root, inc)
        if os.path.isdir(inc_path):
            extra_includes.append(inc_path)
    build_dir = os.path.join(root, ".cache", "ref_builds", task_name)
    os.makedirs(build_dir, exist_ok=True)
    ref_fn = compile_cuda_module(
        cuda_source=ref_cu, function_signature=sig,
        extra_include_dirs=extra_includes,
        extra_cuda_cflags=task_def.get("REFERENCE_EXTRA_CUDA_FLAGS", []),
        build_dir=build_dir, verbose=False)
    print("COMPILE:OK")
except Exception as e:
    print(f"COMPILE:FAIL:{e}")
    sys.exit(0)

try:
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    inputs = task_def["get_inputs"]()
    outputs = task_def["get_outputs"](inputs)
    args = _build_call_args(task_def, inputs, outputs)
    ref_fn(*args)
    torch.cuda.synchronize()
    print("RUN:OK")
except Exception as e:
    print(f"RUN:FAIL:{e}")
    sys.exit(0)

try:
    ref_outputs = task_def["reference_fn"](inputs)
    ok, per = check_correctness(ref_outputs, outputs, task_def["TOLERANCES"])
    print(f"CORRECT:{'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"CORRECT:ERROR:{e}")
'''


def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    tasks = sorted(d for d in os.listdir(FA3_DIR) if os.path.isdir(os.path.join(FA3_DIR, d)))
    print(f"Testing {len(tasks)} FA3 tasks...")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "compile", "run", "correct", "error"])
        f.flush()

        for i, task_name in enumerate(tasks, 1):
            task_path = os.path.join(FA3_DIR, task_name)
            print(f"[{i}/{len(tasks)}] {task_name} ... ", end="", flush=True)

            script = TEST_SCRIPT.replace("{root}", PROJECT_ROOT).replace("{task_path}", task_path).replace("{task_name}", task_name)
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=300
                )
                output = result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                output = "COMPILE:FAIL:timeout"
            except Exception as e:
                output = f"COMPILE:FAIL:{e}"

            compiled = "COMPILE:OK" in output
            ran = "RUN:OK" in output
            correct = ""
            error = ""

            if "CORRECT:PASS" in output:
                correct = "PASS"
            elif "CORRECT:FAIL" in output:
                correct = "FAIL"

            if "COMPILE:FAIL:" in output:
                error = output.split("COMPILE:FAIL:")[-1].strip()[:200]
            elif "RUN:FAIL:" in output:
                error = output.split("RUN:FAIL:")[-1].strip()[:200]
            elif "CORRECT:ERROR:" in output:
                error = output.split("CORRECT:ERROR:")[-1].strip()[:200]
            elif not compiled and result.returncode != 0:
                # Process crashed (e.g., CHECK_CUDA exit)
                lines = output.strip().split("\n")
                error = lines[-1][:200] if lines else f"exit code {result.returncode}"

            status = correct if correct else ("ran" if ran else ("compiled" if compiled else "COMPILE_FAIL"))
            print(f"{status}" + (f"  [{error[:80]}]" if error else ""))
            w.writerow([task_name, compiled, ran, correct, error])
            f.flush()

    print(f"\nResults saved to: {CSV_PATH}")

    # Summary
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    comp = sum(1 for r in rows if r["compile"] == "True")
    run = sum(1 for r in rows if r["run"] == "True")
    correct = sum(1 for r in rows if r["correct"] == "PASS")
    print(f"Compiled: {comp}/{len(rows)}, Ran: {run}/{len(rows)}, Correct: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
