#!/usr/bin/env python3
"""
Re-evaluate self-refine results with subprocess isolation.

For each task, tries all r0..rN solutions in order, stops at first correct one.
Each eval runs in a subprocess to avoid CUDA context poisoning.

Usage:
    python scripts/reeval_selfrefine.py results/opus46_class2sub_selfrefine10
    python scripts/reeval_selfrefine.py results/opus46_class2sub_selfrefine10 --task-list tasks/class2/general/subset_54.txt
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_hercules.runner import discover_tasks, parse_filters, get_gpu_sm
from cuda_hercules.utils import get_project_root


def eval_one_solution(task_id, solution_path, timeout=180):
    """Evaluate a single solution in a subprocess. Returns (compiled, correct, speedup, error, per_size)."""
    script = f"""
import sys, json
sys.path.insert(0, 'src')
from cuda_hercules.runner import discover_tasks, run_task, parse_filters
from cuda_hercules.utils import get_project_root

root = get_project_root()
tasks = {{t.task_id: t for t in discover_tasks(root)}}
config = tasks.get('{task_id}')
if not config:
    print(json.dumps({{"error": "task not found"}}))
    sys.exit(0)

tr = run_task(config, '{solution_path}', measure_perf=True, verbose=False)
print(json.dumps({{
    "compiled": tr.compiled,
    "correct": tr.correct,
    "speedup": tr.speedup,
    "error": tr.error_msg[:300] if tr.error_msg else "",
    "per_size": tr.per_size,
}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
            cwd=get_project_root(),
        )
        if proc.returncode != 0:
            stderr = proc.stderr[-300:] if proc.stderr else ""
            return False, False, 0.0, f"subprocess error: {stderr}", []

        # Parse last line as JSON (skip any warnings)
        for line in reversed(proc.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                result = json.loads(line)
                return (result["compiled"], result["correct"], result["speedup"],
                        result.get("error", ""), result.get("per_size", []))
        return False, False, 0.0, f"no JSON output: {proc.stdout[-200:]}", []
    except subprocess.TimeoutExpired:
        return False, False, 0.0, "timeout", []
    except Exception as e:
        return False, False, 0.0, str(e)[:200], []


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate self-refine results with isolation")
    parser.add_argument("run_dir", help="Self-refine results directory")
    parser.add_argument("--task-list", default="", help="Optional task subset file")
    parser.add_argument("--timeout", type=int, default=180, help="Per-solution timeout (sec)")
    parser.add_argument("--skip-passed", action="store_true", help="Skip tasks that already passed")
    parser.add_argument("--all-rounds", action="store_true", help="Evaluate ALL rounds (don't stop at first correct)")
    args = parser.parse_args()

    root = get_project_root()

    # Load task subset filter
    allowed = None
    if args.task_list:
        allowed = set()
        with open(args.task_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed.add(line)

    # Find all task dirs in run_dir
    task_dirs = sorted(glob.glob(os.path.join(args.run_dir, "class*")))

    total = compiled = correct = 0
    results_summary = []

    for task_dir in task_dirs:
        result_file = os.path.join(task_dir, "result.json")
        if not os.path.exists(result_file):
            continue

        old_result = json.load(open(result_file))
        task_id = old_result["task_id"]
        task_name = task_id.split("/")[-1]

        # Filter by task list
        if allowed and task_name not in allowed and task_id not in allowed:
            continue

        # Skip already passed if requested
        if args.skip_passed and old_result.get("passed_at_round", -1) >= 0:
            total += 1
            compiled += 1
            correct += 1
            print(f"[SKIP] {task_name}: already passed at r{old_result['passed_at_round']}")
            continue

        total += 1

        # Find all solution files (r0_solution.cu, r0_solution.h, ...)
        sol_files = sorted(glob.glob(os.path.join(task_dir, "r*_solution.*")))
        sol_files = [f for f in sol_files if f.endswith('.cu') or f.endswith('.h')]
        if not sol_files:
            print(f"[SKIP] {task_name}: no solution files")
            continue

        print(f"[{total}] {task_name} ({len(sol_files)} rounds)", end=" ", flush=True)

        best_compiled = -1
        best_passed = -1
        best_speedup = 0.0
        round_results = []

        for sol_file in sol_files:
            round_idx = int(os.path.basename(sol_file).split("_")[0][1:])  # r0 -> 0

            is_compiled, is_correct, speedup, error, per_size = eval_one_solution(
                task_id, sol_file, timeout=args.timeout
            )

            round_results.append({
                "round": round_idx,
                "compiled": is_compiled,
                "correct": is_correct,
                "speedup": speedup,
                "error": error,
                "per_size": per_size,
            })

            if is_compiled and best_compiled < 0:
                best_compiled = round_idx
            if is_correct:
                if best_passed < 0:
                    best_passed = round_idx
                if speedup > best_speedup:
                    best_speedup = speedup
                print(f"r{round_idx}:PASS({speedup:.2f}x)", end=" ", flush=True)
                if not args.all_rounds:
                    break  # Stop at first correct
            else:
                tag = "COMP" if is_compiled else "FAIL"
                print(f"r{round_idx}:{tag}", end=" ", flush=True)

        status = f"best=PASS@r{best_passed} {best_speedup:.2f}x" if best_passed >= 0 else "FAIL"
        print(f" -> {status}")

        if best_compiled >= 0:
            compiled += 1
        if best_passed >= 0:
            correct += 1

        # Update result.json
        old_result["compiled_at_round"] = best_compiled
        old_result["passed_at_round"] = best_passed
        old_result["best_speedup"] = best_speedup
        old_result["rounds"] = round_results

        with open(result_file, "w") as f:
            json.dump(old_result, f, indent=2, default=str)

    print(f"\nDone: {total} tasks, {compiled} compiled, {correct} correct ({correct/total:.1%})")


if __name__ == "__main__":
    main()
