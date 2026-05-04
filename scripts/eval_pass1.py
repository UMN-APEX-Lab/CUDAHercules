#!/usr/bin/env python3
"""
CUDA-Hercules Pass@N Evaluation.

Generates N solutions per task, evaluates each, reports pass@N
(task passes if ANY of the N samples is correct).

Usage:
    # pass@1
    python scripts/eval_pass1.py --model gpt-4o --api-key sk-... --num-samples 1

    # pass@10
    python scripts/eval_pass1.py --model Qwen/Qwen3.5-35B-A3B \
        --api-base http://localhost:8000/v1 --num-samples 10

    # pass@1 on Class 2 only
    python scripts/eval_pass1.py --model gpt-4o --filter backend=class2_defpy
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_hercules.runner import discover_tasks, run_task, parse_filters, get_gpu_sm
from cuda_hercules.prompt_builder import build_prompt
from cuda_hercules.llm_api import query_server
from cuda_hercules.utils import extract_cuda_code
from cuda_hercules.score import compute_scores, format_score
from cuda_hercules.static_checker import validate_cuda_solution
from cuda_hercules.task_schema import TaskConfig
from cuda_hercules.result import TaskResult, TaskStatus
from cuda_hercules.utils import get_project_root


def evaluate_sample_in_subprocess(
    task_id: str,
    solution_path: str,
    timeout_sec: int,
) -> dict:
    """Evaluate one sample in a fresh process to avoid CUDA context poisoning."""
    root = get_project_root()
    solution_path = os.path.abspath(solution_path)
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
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"timeout after {timeout_sec}s",
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


def eval_task_pass_n(
    config: TaskConfig,
    num_samples: int,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    run_dir: str,
    verbose: bool = True,
    backend: str = "openai",
    vertex_region: str = "global",
    vertex_project: str = "neu-research",
    is_reasoning_model: bool = False,
    reasoning_effort: str = "",
    isolate_eval: bool = True,
    sample_timeout_sec: int = 1800,
) -> dict:
    """Generate N samples for a task, evaluate each, return best result."""
    root = get_project_root()
    task_dir = os.path.join(root, config.runner.workdir) if config.runner.workdir else ""

    task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
    os.makedirs(task_out, exist_ok=True)

    result = {
        "task_id": config.task_id,
        "task_class": config.task_class,
        "domain": config.domain,
        "model": model,
        "num_samples": num_samples,
        "num_compiled": 0,
        "num_correct": 0,
        "pass_at_n": False,
        "best_speedup": 0.0,
        "best_latency_ms": 0.0,
        "generation_time_s": 0.0,
        "error": "",
    }

    # 1. Build prompt
    try:
        yaml_dir = getattr(config, '_yaml_dir', task_dir)
        prompt = build_prompt(task_dir, config, description_dir=yaml_dir)
    except Exception as e:
        result["error"] = f"Prompt build failed: {e}"
        if verbose:
            print(f"  [SKIP] Prompt error: {e}")
        return result

    with open(os.path.join(task_out, "prompt.txt"), "w") as f:
        f.write(prompt)

    # Determine if this is a multi-file task
    # Filter out directory entries (e.g., "custom_cuda_backend/")
    solution_files = [f for f in (config.runner.solution_files or []) if not f.endswith("/")]
    is_multi_file = len(solution_files) > 1

    # 2. Generate N samples
    t0 = time.time()

    if is_multi_file:
        # Multi-file task: generate each file separately per sample
        if verbose:
            print(f"  Multi-file task ({len(solution_files)} files × {num_samples} samples)...", flush=True)

        all_sample_codes = []  # list of dicts: {filename: code}
        for s in range(num_samples):
            sample_codes = {}
            for sol_file in solution_files:
                file_prompt = prompt + f"\n\nYou are writing the file: `{sol_file}`\nReturn ONLY the complete content of this file."
                try:
                    resp = query_server(
                        prompt=file_prompt, model=model, temperature=temperature,
                        max_tokens=max_tokens, api_base=api_base, api_key=api_key,
                        backend=backend, vertex_region=vertex_region, vertex_project=vertex_project,
                        is_reasoning_model=is_reasoning_model, reasoning_effort=reasoning_effort,
                    )
                    code = extract_cuda_code(resp) if resp else ""
                    sample_codes[sol_file] = code
                except Exception as e:
                    sample_codes[sol_file] = ""
            all_sample_codes.append(sample_codes)
    else:
        # Single-file task: generate N completions at once
        if verbose:
            print(f"  Generating {num_samples} samples...", end=" ", flush=True)
        try:
            responses = query_server(
                prompt=prompt, model=model, temperature=temperature,
                max_tokens=max_tokens, api_base=api_base, api_key=api_key,
                num_completions=num_samples,
                backend=backend, vertex_region=vertex_region, vertex_project=vertex_project,
                is_reasoning_model=is_reasoning_model, reasoning_effort=reasoning_effort,
            )
            if isinstance(responses, str):
                responses = [responses]
        except Exception as e:
            result["error"] = f"Generation failed: {e}"
            result["generation_time_s"] = time.time() - t0
            if verbose:
                print(f"ERROR: {e}")
            return result

        codes = [extract_cuda_code(r) if r else "" for r in responses]
        # Convert to same format as multi-file
        sol_name = config.runner.solution_file or (solution_files[0] if solution_files else "solution.cu")
        all_sample_codes = [{sol_name: code} for code in codes]

    gen_time = time.time() - t0
    result["generation_time_s"] = gen_time
    if verbose:
        print(f"({gen_time:.0f}s)", flush=True)

    # 3. Evaluate each sample
    best_result = None
    sample_details = []

    for s, sample_codes in enumerate(all_sample_codes):
        # Check for empty
        total_lines = sum(len((c or "").splitlines()) for c in sample_codes.values())
        non_empty = sum(1 for c in sample_codes.values() if c and len(c.strip()) > 10)
        if non_empty == 0:
            sample_details.append({"sample": s, "compiled": False, "correct": False, "error": "empty"})
            if verbose:
                print(f"    s{s}: empty")
            continue

        # Save all files for this sample
        sample_dir = os.path.join(task_out, f"sample_{s}")
        os.makedirs(sample_dir, exist_ok=True)
        for filename, code in sample_codes.items():
            filepath = os.path.join(sample_dir, filename)
            # Skip directory-type entries (e.g., "custom_cuda_backend/")
            if filename.endswith("/") or os.path.isdir(filepath):
                os.makedirs(filepath, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(code or "")

        if verbose:
            print(f"    s{s}: {total_lines}L ({len(sample_codes)} files)", end=" → ", flush=True)

        # Static check on all files
        blocked = False
        for filename, code in sample_codes.items():
            if not code:
                continue
            check = validate_cuda_solution(
                code,
                blocked_patterns=config.anti_cheat.blocked_patterns or [],
                required_patterns=config.anti_cheat.required_patterns or [],
            )
            if not check.valid:
                sample_details.append({"sample": s, "compiled": False, "correct": False, "error": f"static check failed: {filename}"})
                if verbose:
                    print(f"BLOCKED ({filename})")
                blocked = True
                break
        if blocked:
            continue

        # Evaluate: pass directory for multi-file, single file for single-file
        sol_path = sample_dir if is_multi_file else os.path.join(sample_dir, list(sample_codes.keys())[0])

        try:
            if isolate_eval:
                sample_result = evaluate_sample_in_subprocess(
                    config.task_id, sol_path, sample_timeout_sec
                )
                compiled = sample_result["compiled"]
                correct = sample_result["correct"]
                speedup = sample_result["speedup"]
                latency_ms = sample_result["latency_ms"]
                error = sample_result.get("error", "")
            else:
                tr = run_task(config, sol_path, measure_perf=True, verbose=False)
                compiled = tr.compiled
                correct = tr.correct
                speedup = tr.speedup
                latency_ms = tr.latency_mean_ms
                error = tr.error_msg or ""

            sample_details.append({
                "sample": s,
                "compiled": compiled,
                "correct": correct,
                "speedup": speedup,
                "latency_ms": latency_ms,
                "error": error,
            })

            if compiled:
                result["num_compiled"] += 1
            if correct:
                result["num_correct"] += 1
                result["pass_at_n"] = True
                if best_result is None or speedup > result["best_speedup"]:
                    result["best_speedup"] = speedup
                    result["best_latency_ms"] = latency_ms

            status = "PASS" if correct else ("COMPILED" if compiled else "FAIL")
            extra = f" {speedup:.2f}x" if speedup > 0 else ""
            if verbose:
                print(f"    s{s}: {total_lines}L {status}{extra}")

        except Exception as e:
            sample_details.append({"sample": s, "compiled": False, "correct": False, "error": str(e)[:200]})
            if verbose:
                print(f"    s{s}: {total_lines}L ERROR")

    # Summary for this task
    status = f"PASS@{num_samples}" if result["pass_at_n"] else \
             f"FAIL ({result['num_compiled']}/{num_samples} compiled)"
    if verbose:
        print(f"  → {status}")

    result["sample_details"] = sample_details

    with open(os.path.join(task_out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="CUDA-Hercules Pass@N Evaluation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-base", default="", help="API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--backend", default="openai", choices=["openai", "vertex"],
                        help="LLM backend: openai (default) or vertex (Vertex AI)")
    parser.add_argument("--vertex-region", default="global", help="Vertex AI region")
    parser.add_argument("--vertex-project", default="neu-research", help="Vertex AI project ID")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning/thinking mode (skips temperature)")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                        help="Reasoning effort level (for o1/o3/GPT-5.4, default: low)")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@N)")
    parser.add_argument(
        "--isolate-eval",
        dest="isolate_eval",
        action="store_true",
        help="Evaluate each sample in a fresh subprocess to avoid CUDA context poisoning.",
    )
    parser.add_argument(
        "--no-isolate-eval",
        dest="isolate_eval",
        action="store_false",
        help="Evaluate samples in-process (legacy behavior).",
    )
    parser.set_defaults(isolate_eval=True)
    parser.add_argument(
        "--sample-timeout-sec",
        type=int,
        default=1800,
        help="Per-sample timeout when isolation is enabled.",
    )
    parser.add_argument("--filter", action="append", default=[], help="Task filters")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--task-list", default="", help="File with task names to include (one per line)")
    parser.add_argument("--run-name", default="", help="Run name")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    N = args.num_samples
    root = get_project_root()

    filters = parse_filters(args.filter) if args.filter else {}
    tasks = discover_tasks(root, filters)

    if not tasks:
        print("No tasks found matching filters.")
        sys.exit(1)

    gpu_sm = get_gpu_sm()
    tasks = [t for t in tasks if t.hardware.min_sm <= gpu_sm]

    # Filter by task list file (subset selection)
    if args.task_list:
        with open(args.task_list) as f:
            allowed = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed.add(line)
        tasks = [t for t in tasks if t.name in allowed or t.task_id in allowed
                 or t.task_id.split("/")[-1] in allowed]
        if not tasks:
            print(f"No tasks matched task-list file: {args.task_list}")
            sys.exit(1)

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    run_name = args.run_name or f"{args.model.replace('/', '_')}_pass{N}_{int(time.time())}"
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"CUDA-Hercules Pass@{N} Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  API:      {args.api_base or 'default'}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Samples:  {N} per task")
    print(f"  GPU:      SM {gpu_sm}")
    print(f"  Output:   {run_dir}")
    print()

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "model": args.model, "api_base": args.api_base,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "num_samples": N, "filters": args.filter,
            "num_tasks": len(tasks), "gpu_sm": gpu_sm,
        }, f, indent=2)

    # Resume: reload completed tasks from per-task result.json files
    all_raw = []
    task_results = []
    completed_ids = set()
    for config in tasks:
        task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
        task_result_file = os.path.join(task_out, "result.json")
        if os.path.isfile(task_result_file):
            with open(task_result_file) as f:
                raw = json.load(f)
            all_raw.append(raw)
            tr = TaskResult(
                task_id=raw["task_id"],
                compiled=raw["num_compiled"] > 0,
                correct=raw["pass_at_n"],
                speedup=raw["best_speedup"],
                latency_mean_ms=raw["best_latency_ms"],
                domain=raw["domain"],
                level=raw["task_class"],
                status=TaskStatus.PASS if raw["pass_at_n"] else TaskStatus.FAIL,
                error_msg=raw["error"],
            )
            task_results.append(tr)
            completed_ids.add(config.task_id)
    if completed_ids:
        print(f"  Resume: {len(completed_ids)} tasks already done, skipping\n")

    consecutive_api_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for i, config in enumerate(tasks):
        if config.task_id in completed_ids:
            continue

        if not args.quiet:
            print(f"[{i+1}/{len(tasks)}] {config.task_id}", flush=True)

        raw = eval_task_pass_n(
            config=config,
            num_samples=N,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            run_dir=run_dir,
            verbose=not args.quiet,
            backend=args.backend,
            vertex_region=args.vertex_region,
            vertex_project=args.vertex_project,
            is_reasoning_model=args.reasoning,
            reasoning_effort=args.reasoning_effort,
            isolate_eval=args.isolate_eval,
            sample_timeout_sec=args.sample_timeout_sec,
        )
        all_raw.append(raw)

        # Stop if API keeps failing
        if raw.get("error", "").startswith("Generation failed"):
            consecutive_api_failures += 1
            if consecutive_api_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n[ABORT] {MAX_CONSECUTIVE_FAILURES} consecutive API failures. Stopping.")
                break
        else:
            consecutive_api_failures = 0

        # Convert to TaskResult for scoring (use best sample)
        tr = TaskResult(
            task_id=raw["task_id"],
            compiled=raw["num_compiled"] > 0,
            correct=raw["pass_at_n"],
            speedup=raw["best_speedup"],
            latency_mean_ms=raw["best_latency_ms"],
            domain=raw["domain"],
            level=raw["task_class"],
            status=TaskStatus.PASS if raw["pass_at_n"] else TaskStatus.FAIL,
            error_msg=raw["error"],
        )
        task_results.append(tr)

    # Score
    score = compute_scores(task_results)
    report = format_score(score)
    print(f"\n{report}")

    # Extra pass@N stats
    total = len(all_raw)
    total_samples = sum(r["num_samples"] for r in all_raw)
    total_compiled = sum(r["num_compiled"] for r in all_raw)
    total_correct = sum(r["num_correct"] for r in all_raw)
    pass_at_n = sum(1 for r in all_raw if r["pass_at_n"])

    print(f"\nPass@{N} Statistics:")
    print(f"  Total samples:  {total_samples}")
    print(f"  Compiled:       {total_compiled}/{total_samples} ({total_compiled/total_samples:.1%})")
    print(f"  Correct:        {total_correct}/{total_samples} ({total_correct/total_samples:.1%})")
    print(f"  Pass@{N}:        {pass_at_n}/{total} ({pass_at_n/total:.1%})")

    # Save
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(all_raw, f, indent=2, default=str)

    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(asdict(score), f, indent=2, default=str)

    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report)
        f.write(f"\n\nPass@{N}: {pass_at_n}/{total} ({pass_at_n/total:.1%})")

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
