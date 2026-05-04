#!/usr/bin/env python3
"""Re-evaluate existing pass@N runs with subprocess isolation."""

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.result import TaskResult, TaskStatus
from cuda_hercules.runner import discover_tasks, get_gpu_sm, parse_filters
from cuda_hercules.score import compute_scores, format_score
from cuda_hercules.static_checker import validate_cuda_solution
from cuda_hercules.task_schema import TaskConfig
from cuda_hercules.utils import get_project_root


def load_allowed(task_list_path: str) -> set[str]:
    allowed = set()
    with open(task_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                allowed.add(line)
    return allowed


def find_solution_path(config: TaskConfig, sample_dir: str) -> str | None:
    solution_files = [f for f in (config.runner.solution_files or []) if not f.endswith("/")]
    is_multi_file = len(solution_files) > 1

    if is_multi_file:
        return sample_dir

    sol_name = config.runner.solution_file or (solution_files[0] if solution_files else "")
    if sol_name:
        candidate = os.path.join(sample_dir, sol_name)
        if os.path.exists(candidate):
            return candidate

    sample_files = sorted(
        f for f in os.listdir(sample_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(sample_dir, f))
    )
    if not sample_files:
        return None
    return os.path.join(sample_dir, sample_files[0])


def collect_code_blobs(config: TaskConfig, sample_dir: str) -> tuple[dict[str, str], bool]:
    solution_files = [f for f in (config.runner.solution_files or []) if not f.endswith("/")]
    is_multi_file = len(solution_files) > 1

    if is_multi_file:
        blobs = {}
        for rel_path in solution_files:
            path = os.path.join(sample_dir, rel_path)
            if os.path.isfile(path):
                with open(path) as f:
                    blobs[rel_path] = f.read()
        return blobs, True

    sol_path = find_solution_path(config, sample_dir)
    if not sol_path or not os.path.isfile(sol_path):
        return {}, False

    with open(sol_path) as f:
        return {os.path.basename(sol_path): f.read()}, False


def static_check_sample(config: TaskConfig, sample_dir: str) -> tuple[bool, str]:
    sample_codes, _ = collect_code_blobs(config, sample_dir)
    if not sample_codes:
        return False, "missing solution files"

    non_empty = 0
    for filename, code in sample_codes.items():
        if not code or len(code.strip()) <= 10:
            continue
        non_empty += 1
        check = validate_cuda_solution(
            code,
            blocked_patterns=config.anti_cheat.blocked_patterns or [],
            required_patterns=config.anti_cheat.required_patterns or [],
        )
        if not check.valid:
            return False, f"static check failed: {filename}"

    if non_empty == 0:
        return False, "empty"
    return True, ""


def evaluate_sample_in_subprocess(task_id: str, solution_path: str, timeout: int) -> dict:
    root = get_project_root()
    script = f"""
import json
import os
import sys
sys.path.insert(0, os.path.join({root!r}, 'src'))
from cuda_hercules.runner import discover_tasks, run_task

tasks = {{t.task_id: t for t in discover_tasks({root!r})}}
config = tasks[{task_id!r}]
tr = run_task(config, {solution_path!r}, measure_perf=True, verbose=False)
print(json.dumps({{
    "compiled": tr.compiled,
    "correct": tr.correct,
    "speedup": tr.speedup,
    "latency_ms": tr.latency_mean_ms,
    "error": tr.error_msg or "",
}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": "timeout",
        }

    output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if proc.returncode != 0 or not output_lines:
        error = (proc.stderr or proc.stdout)[-400:]
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"subprocess error: {error}",
        }

    try:
        return json.loads(output_lines[-1])
    except json.JSONDecodeError:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"invalid subprocess json: {output_lines[-1][:200]}",
        }


def raw_to_task_result(raw: dict) -> TaskResult:
    return TaskResult(
        task_id=raw["task_id"],
        compiled=raw["num_compiled"] > 0,
        correct=raw["pass_at_n"],
        speedup=raw["best_speedup"],
        latency_mean_ms=raw["best_latency_ms"],
        domain=raw["domain"],
        level=raw["task_class"],
        status=TaskStatus.PASS if raw["pass_at_n"] else TaskStatus.FAIL,
        error_msg=raw.get("error", ""),
    )


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate pass@N runs with subprocess isolation")
    parser.add_argument("--run-dir", required=True, help="Existing pass@N results directory")
    parser.add_argument("--filter", action="append", default=[], help="Task filters")
    parser.add_argument("--task-list", default="", help="Optional task subset file")
    parser.add_argument("--timeout", type=int, default=900, help="Per-sample subprocess timeout in seconds")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    root = get_project_root()

    filters = parse_filters(args.filter) if args.filter else {}
    tasks = discover_tasks(root, filters)
    gpu_sm = get_gpu_sm()
    tasks = [t for t in tasks if t.hardware.min_sm <= gpu_sm]
    config_map = {t.task_id: t for t in tasks}

    allowed = load_allowed(args.task_list) if args.task_list else None

    task_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "sample_*"))
    )

    print(f"Re-evaluating pass@N run: {run_dir}")
    print(f"  GPU: SM {gpu_sm}")
    print(f"  Tasks found: {len(task_dirs)}")
    print()

    all_raw = []
    task_results = []

    for idx, task_out in enumerate(task_dirs, start=1):
        dir_name = os.path.basename(task_out)
        config = None
        for task_id, cfg in config_map.items():
            if task_id.replace("/", "_") == dir_name:
                config = cfg
                break
        if config is None:
            if not args.quiet:
                print(f"[{idx}] {dir_name}: no matching task config, skipping")
            continue

        if allowed and config.name not in allowed and config.task_id not in allowed and config.task_id.split("/")[-1] not in allowed:
            continue

        sample_dirs = sorted(
            d for d in glob.glob(os.path.join(task_out, "sample_*"))
            if os.path.isdir(d)
        )
        if not sample_dirs:
            continue

        raw = {
            "task_id": config.task_id,
            "task_class": config.task_class,
            "domain": config.domain,
            "num_samples": len(sample_dirs),
            "num_compiled": 0,
            "num_correct": 0,
            "pass_at_n": False,
            "best_speedup": 0.0,
            "best_latency_ms": 0.0,
            "error": "",
            "sample_details": [],
        }

        if not args.quiet:
            print(f"[{idx}] {config.task_id}", flush=True)

        for sample_index, sample_dir in enumerate(sample_dirs):
            sample_codes, is_multi_file = collect_code_blobs(config, sample_dir)
            total_lines = sum(len((code or "").splitlines()) for code in sample_codes.values())

            ok, static_error = static_check_sample(config, sample_dir)
            if not ok:
                raw["sample_details"].append({
                    "sample": sample_index,
                    "compiled": False,
                    "correct": False,
                    "error": static_error,
                })
                if not args.quiet:
                    print(f"  s{sample_index}: {total_lines}L FAIL ({static_error})")
                continue

            solution_path = sample_dir if is_multi_file else find_solution_path(config, sample_dir)
            if not solution_path:
                raw["sample_details"].append({
                    "sample": sample_index,
                    "compiled": False,
                    "correct": False,
                    "error": "missing solution files",
                })
                continue

            sample_result = evaluate_sample_in_subprocess(config.task_id, solution_path, args.timeout)
            raw["sample_details"].append({"sample": sample_index, **sample_result})

            if sample_result["compiled"]:
                raw["num_compiled"] += 1
            if sample_result["correct"]:
                raw["num_correct"] += 1
                raw["pass_at_n"] = True
                if sample_result["speedup"] > raw["best_speedup"]:
                    raw["best_speedup"] = sample_result["speedup"]
                    raw["best_latency_ms"] = sample_result["latency_ms"]

            if not args.quiet:
                status = "PASS" if sample_result["correct"] else ("COMPILED" if sample_result["compiled"] else "FAIL")
                extra = f" {sample_result['speedup']:.2f}x" if sample_result["speedup"] > 0 else ""
                print(f"  s{sample_index}: {total_lines}L {status}{extra}")

        if not args.quiet:
            summary = f"PASS@{len(sample_dirs)}" if raw["pass_at_n"] else f"FAIL ({raw['num_compiled']}/{len(sample_dirs)} compiled)"
            print(f"  → {summary}")

        with open(os.path.join(task_out, "result_reeval.json"), "w") as f:
            json.dump(raw, f, indent=2, default=str)

        all_raw.append(raw)
        task_results.append(raw_to_task_result(raw))

    score = compute_scores(task_results)
    report = format_score(score)
    total = len(all_raw)
    total_samples = sum(r["num_samples"] for r in all_raw)
    total_compiled = sum(r["num_compiled"] for r in all_raw)
    total_correct = sum(r["num_correct"] for r in all_raw)
    pass_at_n = sum(1 for r in all_raw if r["pass_at_n"])

    print(f"\n{report}")
    if total > 0 and total_samples > 0:
        print("\nRe-evaluated Pass@N Statistics:")
        print(f"  Total samples:  {total_samples}")
        print(f"  Compiled:       {total_compiled}/{total_samples} ({total_compiled/total_samples:.1%})")
        print(f"  Correct:        {total_correct}/{total_samples} ({total_correct/total_samples:.1%})")
        print(f"  Pass@N:         {pass_at_n}/{total} ({pass_at_n/total:.1%})")

    with open(os.path.join(run_dir, "results_reeval.json"), "w") as f:
        json.dump(all_raw, f, indent=2, default=str)
    with open(os.path.join(run_dir, "score_reeval.json"), "w") as f:
        json.dump(asdict(score), f, indent=2, default=str)
    with open(os.path.join(run_dir, "report_reeval.txt"), "w") as f:
        f.write(report)
        if total > 0:
            f.write(f"\n\nPass@N: {pass_at_n}/{total} ({pass_at_n/total:.1%})")

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
