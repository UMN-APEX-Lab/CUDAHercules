#!/usr/bin/env python3
"""
CUDA-Hercules Self-Refine Evaluation.

For each task:
  1. Generate initial solution (pass 0)
  2. Compile & test
  3. If failed, feed error back to LLM as feedback and ask it to fix
  4. Repeat up to --max-refine rounds
  5. Report: which round first passed, total rounds used

Usage:
    python scripts/eval_self_refine.py \
        --model gpt-4o \
        --api-key sk-... \
        --filter backend=class2_defpy \
        --max-refine 3 \
        --max-tasks 10 \
        --run-name gpt4o_refine3
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict

ROUND_TIMEOUT_SEC = 1800  # 30 min wall-clock budget per refine round


def _round_timeout_trigger(pid: int, flag: dict) -> None:
    """Fires after ROUND_TIMEOUT_SEC; SIGINT reliably interrupts blocking
    C calls (torch compile, ninja subprocess wait, CUDA sync, httpx)."""
    flag["timed_out"] = True
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_hercules.runner import discover_tasks, run_task, parse_filters, get_gpu_sm
from cuda_hercules.prompt_builder import build_prompt
from cuda_hercules.llm_api import query_server
from cuda_hercules.utils import extract_cuda_code, get_project_root
from cuda_hercules.score import compute_scores, format_score
from cuda_hercules.static_checker import validate_cuda_solution
from cuda_hercules.task_schema import TaskConfig
from cuda_hercules.result import TaskResult, TaskStatus


REFINE_SYSTEM_PROMPT = """You are a CUDA kernel programming expert. \
You write high-performance, correct CUDA code. \
When given compiler errors or test failures, you carefully analyze the problem and fix your code."""

REFINE_FIX_TEMPLATE = """Your previous CUDA solution failed. Here is the feedback:

## Error
{error}

{correctness_diff}

## Instructions
Refer to the original task description earlier in this conversation.
Carefully analyze the error above. Fix the issue in your code and return the complete corrected .cu file in a ```cuda code block.
Do NOT explain the changes — just return the fixed code."""

REFINE_OPTIMIZE_TEMPLATE = """Your CUDA solution is correct but needs performance optimization.

## Current Performance
- Overall speedup vs reference: {speedup:.2f}x
{per_size_detail}
{perf_detail}

## Instructions
Refer to the original task description earlier in this conversation.
Optimize your code for better performance. Consider:
- Shared memory tiling and bank conflict avoidance
- Memory coalescing and vectorized loads/stores (float4)
- Warp-level primitives (__shfl_down_sync, warp reduction)
- Register blocking and loop unrolling
- Occupancy optimization (block size tuning)
- Reducing branch divergence

Return the complete optimized .cu file in a ```cuda code block.
Do NOT explain — just return the optimized code."""


def eval_task_self_refine(
    config: TaskConfig,
    max_refine: int,
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
) -> dict:
    """Generate, test, refine up to max_refine rounds."""
    root = get_project_root()
    task_dir = os.path.join(root, config.runner.workdir) if config.runner.workdir else ""

    task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
    os.makedirs(task_out, exist_ok=True)

    result = {
        "task_id": config.task_id,
        "task_class": config.task_class,
        "domain": config.domain,
        "model": model,
        "final_status": TaskStatus.FAIL.value,
        "max_refine": max_refine,
        "passed_at_round": -1,  # -1 = never passed
        "total_rounds": 0,
        "compiled_at_round": -1,
        "best_speedup": 0.0,
        "rounds": [],
    }

    # Build initial prompt
    try:
        yaml_dir = getattr(config, '_yaml_dir', task_dir)
        task_prompt = build_prompt(task_dir, config, description_dir=yaml_dir)
    except Exception as e:
        result["error"] = f"Prompt build failed: {e}"
        if verbose:
            print(f"  [SKIP] {e}")
        return result

    with open(os.path.join(task_out, "prompt.txt"), "w") as f:
        f.write(task_prompt)

    current_code = ""
    previous_code = ""
    latest_feedback = ""

    def _build_messages() -> list[dict]:
        """Build a compact context window for the next LLM call.

        Keep only:
          1. original task description
          2. previous round's code
          3. latest refinement feedback / optimization request
        """
        messages = [
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        if previous_code and latest_feedback:
            messages.append({"role": "assistant", "content": f"```cuda\n{previous_code}\n```"})
            messages.append({"role": "user", "content": latest_feedback})
        return messages

    for round_idx in range(max_refine + 1):  # round 0 = initial, 1..max_refine = refinements
        round_label = f"r{round_idx}"
        if verbose:
            tag = "GEN" if round_idx == 0 else f"FIX-{round_idx}"
            print(f"    [{tag}]", end=" ", flush=True)

        # Per-round wall-clock timeout — fires SIGINT to interrupt any hang
        # (vLLM, torch compile, CUDA sync, etc.). Raises KeyboardInterrupt in
        # the main thread, which bubbles past the inner `except Exception`
        # blocks since KeyboardInterrupt is BaseException, not Exception.
        _timeout_flag = {"timed_out": False}
        _round_timer = threading.Timer(
            ROUND_TIMEOUT_SEC,
            _round_timeout_trigger,
            args=(os.getpid(), _timeout_flag),
        )
        _round_timer.daemon = True
        _round_timer.start()

        try:
            # Call LLM
            t0 = time.time()
            try:
                response = query_server(
                    prompt=_build_messages(),
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_base=api_base,
                    api_key=api_key,
                    backend=backend,
                    vertex_region=vertex_region,
                    vertex_project=vertex_project,
                    is_reasoning_model=is_reasoning_model,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                round_info = {"round": round_idx, "error": f"LLM call failed: {e}"}
                result["rounds"].append(round_info)
                if verbose:
                    print(f"LLM ERROR")
                break
            gen_time = time.time() - t0

            # Extract code
            current_code = extract_cuda_code(response) if response else ""
            if not current_code or len(current_code.strip()) < 10:
                round_info = {"round": round_idx, "error": "empty response", "gen_time_s": gen_time}
                result["rounds"].append(round_info)
                if verbose:
                    print(f"empty ({gen_time:.0f}s)")
                # Can't refine empty code
                break

            # Save
            sol_name = config.runner.solution_file or "solution.cu"
            sol_path = os.path.join(task_out, f"{round_label}_{sol_name}")
            with open(sol_path, "w") as f:
                f.write(current_code)

            lines = len(current_code.splitlines())
            if verbose:
                print(f"{lines}L ({gen_time:.0f}s)", end=" → ", flush=True)

            # Static check
            check = validate_cuda_solution(
                current_code,
                blocked_patterns=config.anti_cheat.blocked_patterns or [],
                required_patterns=config.anti_cheat.required_patterns or [],
            )
            if not check.valid:
                error_msg = "; ".join(check.errors)
                round_info = {
                    "round": round_idx, "compiled": False, "correct": False,
                    "error": f"Static check: {error_msg}", "gen_time_s": gen_time,
                }
                result["rounds"].append(round_info)
                if verbose:
                    print(f"BLOCKED")

                # Feed back static check error for refinement
                previous_code = current_code
                latest_feedback = REFINE_FIX_TEMPLATE.format(
                    error=f"Static check failed: {error_msg}",
                    correctness_diff="",
                )
                result["total_rounds"] = round_idx + 1
                continue

            # Evaluate
            per_size = []
            try:
                tr = run_task(config, sol_path, measure_perf=True, verbose=False)
                compiled = tr.compiled
                correct = tr.correct
                speedup = tr.speedup
                status = tr.status.value if isinstance(tr.status, TaskStatus) else str(tr.status)
                per_size = getattr(tr, 'per_size', []) or []
                error_msg = tr.error_msg or ""
            except Exception as e:
                compiled = False
                correct = False
                speedup = 0.0
                status = TaskStatus.ERROR.value
                error_msg = str(e)[:500]

            round_info = {
                "round": round_idx,
                "status": status,
                "compiled": compiled,
                "correct": correct,
                "speedup": speedup,
                "per_size": per_size,
                "error": error_msg,
                "gen_time_s": gen_time,
                "lines": lines,
            }
            result["rounds"].append(round_info)
            result["total_rounds"] = round_idx + 1

            if compiled and result["compiled_at_round"] < 0:
                result["compiled_at_round"] = round_idx

            if correct:
                result["final_status"] = TaskStatus.PASS.value
                if result["passed_at_round"] < 0:
                    result["passed_at_round"] = round_idx
                if speedup > result["best_speedup"]:
                    result["best_speedup"] = speedup

            # Update final_report.json — tracks best-so-far across all rounds
            report_path = os.path.join(task_out, "final_report.json")
            report = {}
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)

            # Initialize report on first round
            if "task_id" not in report:
                report = {
                    "task_id": config.task_id,
                    "model": model,
                    "final_status": TaskStatus.FAIL.value,
                    "best_round": -1,
                    "best_speedup": 0.0,
                    "best_per_size": [],
                    "best_solution_file": "",
                    "first_compiled_round": -1,
                    "first_correct_round": -1,
                    "history": [],
                }

            # Record this round
            report["history"].append({
                "round": round_idx,
                "status": status,
                "compiled": compiled,
                "correct": correct,
                "speedup": speedup,
                "per_size": round_info["per_size"],
            })

            if compiled and report["first_compiled_round"] < 0:
                report["first_compiled_round"] = round_idx
            if correct and report["first_correct_round"] < 0:
                report["first_correct_round"] = round_idx

            # Update best if this round is better
            if correct and speedup > report["best_speedup"]:
                report["best_round"] = round_idx
                report["best_speedup"] = speedup
                report["best_per_size"] = round_info["per_size"]
                report["best_solution_file"] = f"{round_label}_{config.runner.solution_file or 'solution.cu'}"

            if status == TaskStatus.SKIP_ARCH.value:
                result["final_status"] = TaskStatus.SKIP_ARCH.value
                report["final_status"] = TaskStatus.SKIP_ARCH.value
            elif result["final_status"] != TaskStatus.SKIP_ARCH.value and correct:
                report["final_status"] = TaskStatus.PASS.value
            else:
                report["final_status"] = result["final_status"]

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            if status == TaskStatus.SKIP_ARCH.value:
                if verbose:
                    print("SKIP_ARCH")
                break

            if correct:
                status_str = f"PASS {speedup:.2f}x" if speedup > 0 else "PASS"
                if verbose:
                    print(status_str)

                # Continue refining for performance
                if round_idx < max_refine:
                    perf_detail = ""
                    if tr.ref_latency_mean_ms > 0:
                        perf_detail = f"- Reference time: {tr.ref_latency_mean_ms:.4f} ms"
                    if speedup >= 1.0:
                        perf_detail += "\n- Your code is faster than the reference, but can you push it further?"
                    else:
                        perf_detail += "\n- Your code is SLOWER than the reference. Significant optimization needed."

                    # Per-size speedup breakdown (Class 1)
                    per_size_detail = ""
                    if tr.per_size:
                        lines = []
                        for e in tr.per_size:
                            lines.append(f"  - {e['name']}: {e['speedup']:.2f}x "
                                         f"(kernel={e['kernel_min_ms']:.4f}ms, ref={e['ref_min_ms']:.4f}ms)")
                        per_size_detail = "- Per-size breakdown:\n" + "\n".join(lines)

                    previous_code = current_code
                    latest_feedback = REFINE_OPTIMIZE_TEMPLATE.format(
                        speedup=speedup,
                        per_size_detail=per_size_detail,
                        perf_detail=perf_detail,
                    )
            else:
                status_str = "COMPILED" if compiled else "FAIL"
                if verbose:
                    print(status_str)

                # Prepare error feedback for next round
                if round_idx < max_refine:
                    feedback_error = error_msg if error_msg else "Compilation or correctness check failed."

                    # Build correctness diff detail
                    correctness_diff = ""
                    if compiled and not correct:
                        feedback_error = "Code compiled successfully but produced incorrect results."
                        detail = tr.correctness_detail
                        if detail:
                            diff_lines = []
                            for name, info in detail.items():
                                if isinstance(info, dict) and not info.get("correct", True):
                                    if "max_diff" in info:
                                        diff_lines.append(
                                            f"- Output '{name}': max_diff={info['max_diff']:.6g}, "
                                            f"mean_diff={info['mean_diff']:.6g}, "
                                            f"mismatched={info.get('num_mismatched','?')}/{info.get('total_elements','?')}\n"
                                            f"  At worst element: expected={info.get('ref_at_max','?')}, got={info.get('sol_at_max','?')}")
                                    elif "error" in info:
                                        diff_lines.append(f"- Output '{name}': {info['error']}")
                            if diff_lines:
                                correctness_diff = "## Correctness Details\n" + "\n".join(diff_lines)

                    previous_code = current_code
                    latest_feedback = REFINE_FIX_TEMPLATE.format(
                        error=feedback_error,
                        correctness_diff=correctness_diff,
                    )
        except KeyboardInterrupt:
            # Only treat as round-timeout if our timer fired; otherwise re-raise
            # so a real Ctrl-C still exits the whole eval.
            if not _timeout_flag["timed_out"]:
                raise
            round_info = {
                "round": round_idx,
                "compiled": False,
                "correct": False,
                "speedup": 0.0,
                "per_size": [],
                "error": f"Round timed out after {ROUND_TIMEOUT_SEC}s",
            }
            result["rounds"].append(round_info)
            result["total_rounds"] = round_idx + 1
            if verbose:
                print(f"TIMEOUT", flush=True)
            continue
        finally:
            _round_timer.cancel()

    # Save result
    with open(os.path.join(task_out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="CUDA-Hercules Self-Refine Evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--backend", default="openai", choices=["openai", "vertex"],
                        help="LLM backend: openai (default) or vertex (Vertex AI)")
    parser.add_argument("--vertex-region", default="global", help="Vertex AI region")
    parser.add_argument("--vertex-project", default="neu-research", help="Vertex AI project ID")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable reasoning/thinking mode (skips temperature for o1/o3/GPT-5/Claude thinking)")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                        help="Reasoning effort level (for o1/o3/GPT-5, default: low)")
    parser.add_argument("--max-refine", type=int, default=3, help="Max refinement rounds (0=no refine)")
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--task-list", default="", help="File with task names (one per line) to restrict evaluation")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output", default="results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = get_project_root()
    filters = parse_filters(args.filter) if args.filter else {}
    tasks = discover_tasks(root, filters)

    if not tasks:
        print("No tasks found.")
        sys.exit(1)

    gpu_sm = get_gpu_sm()
    tasks = [t for t in tasks if t.hardware.min_sm <= gpu_sm]

    if args.task_list:
        with open(args.task_list) as f:
            allowed = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed.add(line)
        tasks = [t for t in tasks if t.task_id.split("/")[-1] in allowed]

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    run_name = args.run_name or f"{args.model.replace('/', '_')}_refine{args.max_refine}_{int(time.time())}"
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)

    R = args.max_refine
    print(f"CUDA-Hercules Self-Refine Evaluation")
    print(f"  Model:      {args.model}")
    print(f"  Max refine: {R} rounds (total attempts: {R+1})")
    print(f"  Tasks:      {len(tasks)}")
    print(f"  GPU:        SM {gpu_sm}")
    print(f"  Output:     {run_dir}")
    print()

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "model": args.model, "api_base": args.api_base,
            "backend": args.backend, "max_refine": R,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "filters": args.filter, "num_tasks": len(tasks), "gpu_sm": gpu_sm,
        }, f, indent=2)

    # Resume: reload completed tasks
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
            raw_status = raw.get("final_status", TaskStatus.PASS.value if raw["passed_at_round"] >= 0 else TaskStatus.FAIL.value)
            tr = TaskResult(
                task_id=raw["task_id"],
                compiled=raw["compiled_at_round"] >= 0,
                correct=raw["passed_at_round"] >= 0,
                speedup=raw["best_speedup"],
                domain=raw.get("domain", ""),
                level=raw.get("task_class", 0),
                status=TaskStatus(raw_status),
            )
            task_results.append(tr)
            completed_ids.add(config.task_id)
    if completed_ids:
        print(f"  Resume: {len(completed_ids)} tasks already done, skipping\n")

    for i, config in enumerate(tasks):
        if config.task_id in completed_ids:
            continue

        if not args.quiet:
            print(f"[{i+1}/{len(tasks)}] {config.task_id}", flush=True)

        raw = eval_task_self_refine(
            config=config, max_refine=R,
            model=args.model, api_base=args.api_base, api_key=args.api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            run_dir=run_dir, verbose=not args.quiet,
            backend=args.backend, vertex_region=args.vertex_region,
            vertex_project=args.vertex_project,
            is_reasoning_model=args.reasoning,
            reasoning_effort=args.reasoning_effort,
        )
        all_raw.append(raw)

        tr = TaskResult(
            task_id=raw["task_id"],
            compiled=raw["compiled_at_round"] >= 0,
            correct=raw["passed_at_round"] >= 0,
            speedup=raw["best_speedup"],
            domain=raw["domain"],
            level=raw["task_class"],
            status=TaskStatus(raw.get("final_status", TaskStatus.PASS.value if raw["passed_at_round"] >= 0 else TaskStatus.FAIL.value)),
        )
        task_results.append(tr)

    # Score
    score = compute_scores(task_results)
    print(f"\n{format_score(score)}")

    # Self-refine specific stats
    total = len(all_raw)
    pass_by_round = {}
    compile_by_round = {}
    for r in all_raw:
        pr = r["passed_at_round"]
        cr = r["compiled_at_round"]
        if pr >= 0:
            pass_by_round[pr] = pass_by_round.get(pr, 0) + 1
        if cr >= 0:
            compile_by_round[cr] = compile_by_round.get(cr, 0) + 1

    print(f"\nSelf-Refine Statistics (max {R} refinements):")
    cumulative_pass = 0
    cumulative_compile = 0
    for r in range(R + 1):
        p = pass_by_round.get(r, 0)
        c = compile_by_round.get(r, 0)
        cumulative_pass += p
        cumulative_compile += c
        label = "initial" if r == 0 else f"refine-{r}"
        print(f"  {label:>10}: +{p} pass (cum: {cumulative_pass}/{total} = {cumulative_pass/total:.1%}), "
              f"+{c} compile (cum: {cumulative_compile}/{total} = {cumulative_compile/total:.1%})")

    # Speedup progression for tasks that passed
    print(f"\nPerformance Progression (tasks that passed):")
    for raw in all_raw:
        if raw["passed_at_round"] < 0:
            continue
        speedups = [rd.get("speedup", 0) for rd in raw["rounds"] if rd.get("correct")]
        if speedups:
            progression = " → ".join(f"{s:.2f}x" for s in speedups)
            best = max(speedups)
            first = speedups[0]
            improved = f" (+{best/first:.1f}x)" if len(speedups) > 1 and first > 0 else ""
            print(f"  {raw['task_id']:<45} {progression}{improved}")

    # Save
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(all_raw, f, indent=2, default=str)
    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(asdict(score), f, indent=2, default=str)
    with open(os.path.join(run_dir, "refine_stats.json"), "w") as f:
        json.dump({"pass_by_round": pass_by_round, "compile_by_round": compile_by_round,
                    "total": total, "max_refine": R}, f, indent=2)

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
