"""Unified CLI entry point for CUDA-Hercules.

Usage:
    cuda-hercules list      [--filter ...]
    cuda-hercules prompt    --task <dir>
    cuda-hercules generate  --task <dir> --model <model> [--output <dir>]
    cuda-hercules run       --task <dir> --solution <file>
    cuda-hercules eval      --task <dir> --model <model> [--output <dir>]
    cuda-hercules run-all   --solutions-dir <dir> [--filter ...] [--output <dir>]
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from .utils import get_project_root


# ── list ──────────────────────────────────────────────────────────────

def cmd_list(args):
    """List all discovered tasks with GPU compatibility info."""
    from .runner import discover_tasks, parse_filters, get_gpu_sm

    root = get_project_root()
    filters = parse_filters(args.filter) if args.filter else {}
    tasks = discover_tasks(root, filters)

    gpu_sm = get_gpu_sm()
    print(f"GPU: SM {gpu_sm}")
    print(f"{'TASK ID':<55} {'CLS':>3} {'SM':>4} {'BACKEND':<16} {'DOMAIN':<12} {'RUN?'}")
    print("-" * 100)

    for config in tasks:
        can_run = "YES" if gpu_sm >= config.hardware.min_sm else "SKIP"
        print(
            f"{config.task_id:<55} {config.task_class:>3} {config.hardware.min_sm:>4} "
            f"{config.runner.backend:<16} {config.domain:<12} {can_run}"
        )

    print(f"\nTotal: {len(tasks)} tasks")


# ── prompt ────────────────────────────────────────────────────────────

def cmd_prompt(args):
    """Print the LLM prompt for a task (no LLM call)."""
    from .prompt_builder import build_prompt_from_yaml

    yaml_path = _resolve_yaml(args.task)
    prompt = build_prompt_from_yaml(yaml_path)
    print(prompt)


# ── generate ──────────────────────────────────────────────────────────

def cmd_generate(args):
    """Call LLM to generate a solution for a task."""
    from .task_schema import load_task_config
    from .prompt_builder import build_prompt
    from .llm_api import generate_kernel

    yaml_path = _resolve_yaml(args.task)
    config = load_task_config(yaml_path)
    yaml_dir = os.path.dirname(yaml_path)
    task_dir = os.path.join(get_project_root(), config.runner.workdir) if config.runner.workdir else yaml_dir

    print(f"Task:  {config.task_id}")
    print(f"Model: {args.model}")

    # Build prompt
    prompt = build_prompt(task_dir, config, description_dir=yaml_dir)
    if args.show_prompt:
        print(f"\n{'='*60}\nPROMPT\n{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")

    # Generate
    print("Generating solution...", flush=True)
    cuda_code = generate_kernel(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=getattr(args, 'api_base', '') or '',
    )

    if not cuda_code:
        print("ERROR: LLM returned empty code")
        sys.exit(1)

    # Save
    out_dir = args.output or os.path.join("results", config.task_id)
    os.makedirs(out_dir, exist_ok=True)

    sol_name = config.runner.solution_file or "solution.cu"
    sol_path = os.path.join(out_dir, sol_name)
    with open(sol_path, "w") as f:
        f.write(cuda_code)

    # Also save prompt and metadata
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"task_id": config.task_id, "model": args.model,
                    "temperature": args.temperature}, f, indent=2)

    print(f"Solution saved: {sol_path}")
    print(f"Lines: {len(cuda_code.splitlines())}")
    return sol_path


# ── run ───────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run evaluation on a pre-written solution."""
    from .runner import run_task
    from .task_schema import load_task_config

    yaml_path = _resolve_yaml(args.task)
    config = load_task_config(yaml_path)
    result = run_task(
        config, args.solution,
        measure_perf=not args.no_perf,
        verbose=not args.quiet,
    )

    _print_result(result)

    if args.output:
        _save_result(result, config, args.output)

    sys.exit(0 if result.correct else 1)


# ── eval (generate + run) ────────────────────────────────────────────

def cmd_eval(args):
    """Generate a solution with LLM and immediately evaluate it."""
    from .task_schema import load_task_config
    from .prompt_builder import build_prompt
    from .llm_api import generate_kernel
    from .runner import run_task

    yaml_path = _resolve_yaml(args.task)
    config = load_task_config(yaml_path)
    yaml_dir = os.path.dirname(yaml_path)
    task_dir = os.path.join(get_project_root(), config.runner.workdir) if config.runner.workdir else yaml_dir

    print(f"Task:  {config.task_id}")
    print(f"Model: {args.model}")

    # Generate
    prompt = build_prompt(task_dir, config, description_dir=yaml_dir)
    print("Generating solution...", flush=True)
    cuda_code = generate_kernel(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=getattr(args, 'api_base', '') or '',
    )

    if not cuda_code:
        print("ERROR: LLM returned empty code")
        sys.exit(1)

    # Save solution
    out_dir = args.output or os.path.join("results", config.task_id)
    os.makedirs(out_dir, exist_ok=True)
    sol_name = config.runner.solution_file or "solution.cu"
    sol_path = os.path.join(out_dir, sol_name)
    with open(sol_path, "w") as f:
        f.write(cuda_code)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    print(f"Solution saved: {sol_path} ({len(cuda_code.splitlines())} lines)")

    # Evaluate
    print("\nEvaluating...", flush=True)
    result = run_task(
        config, sol_path,
        measure_perf=not args.no_perf,
        verbose=not args.quiet,
    )

    _print_result(result)
    _save_result(result, config, out_dir,
                 extra={"model": args.model, "temperature": args.temperature})

    sys.exit(0 if result.correct else 1)


# ── agent (LangChain) ────────────────────────────────────────────────

def cmd_agent(args):
    """Run LangChain agent with tool calling for iterative CUDA optimization."""
    from .agent import run_agent

    task_dir = args.task
    if not os.path.isdir(task_dir):
        print(f"ERROR: {task_dir} is not a directory")
        sys.exit(1)

    result = run_agent(
        task_dir=task_dir,
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        output_dir=args.output or "",
        verbose=True,
    )

    print(f"\nResult: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Solution: {result['solution_path']}")
    sys.exit(0 if result["passed"] else 1)


# ── run-all ───────────────────────────────────────────────────────────

def cmd_run_all(args):
    """Run all tasks with solutions from a directory."""
    from .runner import discover_tasks, run_task, compute_summary, parse_filters

    root = get_project_root()
    filters = parse_filters(args.filter) if args.filter else {}
    tasks = discover_tasks(root, filters)

    if not tasks:
        print("No tasks found matching filters.")
        sys.exit(1)

    print(f"Found {len(tasks)} tasks")
    results = []

    for config in tasks:
        sol_path = _find_solution(config, args.solutions_dir)
        if not sol_path:
            if not args.quiet:
                print(f"[{config.task_id}] No solution found, skipping")
            continue

        result = run_task(
            config, sol_path,
            measure_perf=not args.no_perf,
            verbose=not args.quiet,
        )
        results.append(result)

    from .score import compute_scores, format_score
    score = compute_scores(results)
    print(format_score(score))
    summary = compute_summary(results)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {args.output}")


# ── Helpers ───────────────────────────────────────────────────────────

def _resolve_yaml(task_arg: str) -> str:
    """Resolve a task argument to a task.yaml path."""
    if os.path.isfile(task_arg) and task_arg.endswith(".yaml"):
        return task_arg
    if os.path.isdir(task_arg):
        yaml_path = os.path.join(task_arg, "task.yaml")
        if os.path.isfile(yaml_path):
            return yaml_path
    raise FileNotFoundError(f"Cannot find task.yaml for: {task_arg}")


def _print_result(result):
    """Print a single task result."""
    print(f"\n{'=' * 60}")
    print(f"Task:     {result.task_id}")
    print(f"Status:   {result.status.value}")
    print(f"Compiled: {'YES' if result.compiled else 'NO'}")
    print(f"Correct:  {'YES' if result.correct else 'NO'}")
    if result.speedup > 0:
        print(f"Speedup:  {result.speedup:.2f}x")
    if result.latency_mean_ms > 0:
        print(f"Latency:  {result.latency_mean_ms:.4f} ms")
    if result.error_msg:
        print(f"Error:    {result.error_msg[:300]}")


def _save_result(result, config, out_dir, extra=None):
    """Save result JSON to output directory."""
    os.makedirs(out_dir, exist_ok=True)
    data = asdict(result)
    if extra:
        data.update(extra)
    out_path = os.path.join(out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {out_path}")


def _find_solution(config, solutions_dir):
    """Find solution file for a task in the solutions directory."""
    if not solutions_dir:
        return None

    sol_file = config.runner.solution_file or "solution.cu"
    candidates = [
        os.path.join(solutions_dir, config.task_id, sol_file),
        os.path.join(solutions_dir, config.task_id, "solution.cu"),
        os.path.join(solutions_dir, config.task_id, "solution.h"),
        os.path.join(solutions_dir, config.name, sol_file),
    ]
    # For Class 3: solution_dir contains all solution files
    if config.runner.solution_files:
        sol_dir = os.path.join(solutions_dir, config.task_id)
        if os.path.isdir(sol_dir):
            return sol_dir

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cuda-hercules",
        description="CUDA-Hercules: Benchmark for LLM CUDA Optimization",
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_p = subparsers.add_parser("list", help="List available tasks")
    list_p.add_argument("--filter", action="append", default=[],
                        help="Filter (e.g. task_class=3, domain=ml, min_sm<=90, backend=class3_app)")

    # prompt
    prompt_p = subparsers.add_parser("prompt", help="Show the LLM prompt for a task")
    prompt_p.add_argument("--task", required=True, help="Task directory or task.yaml")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate solution with LLM")
    gen_p.add_argument("--task", required=True, help="Task directory or task.yaml")
    gen_p.add_argument("--model", default="gpt-4o", help="LLM model name")
    gen_p.add_argument("--api-base", default="", help="API base URL for local servers (e.g., http://localhost:8000/v1)")
    gen_p.add_argument("--temperature", type=float, default=0.6)
    gen_p.add_argument("--max-tokens", type=int, default=16384)
    gen_p.add_argument("--output", help="Output directory for solution")
    gen_p.add_argument("--show-prompt", action="store_true", help="Print the prompt")

    # run
    run_p = subparsers.add_parser("run", help="Evaluate a pre-written solution")
    run_p.add_argument("--task", required=True, help="Task directory or task.yaml")
    run_p.add_argument("--solution", required=True, help="Path to solution file")
    run_p.add_argument("--output", help="Output directory for results")
    run_p.add_argument("--no-perf", action="store_true", help="Skip performance measurement")
    run_p.add_argument("--quiet", action="store_true")

    # eval (generate + run)
    eval_p = subparsers.add_parser("eval", help="Generate with LLM and evaluate")
    eval_p.add_argument("--task", required=True, help="Task directory or task.yaml")
    eval_p.add_argument("--model", default="gpt-4o", help="LLM model")
    eval_p.add_argument("--api-base", default="", help="API base URL for local servers")
    eval_p.add_argument("--temperature", type=float, default=0.6)
    eval_p.add_argument("--max-tokens", type=int, default=16384)
    eval_p.add_argument("--output", help="Output directory")
    eval_p.add_argument("--no-perf", action="store_true")
    eval_p.add_argument("--quiet", action="store_true")

    # agent (LangChain tool-calling workflow)
    agent_p = subparsers.add_parser("agent", help="Run LangChain agent with tool calling (iterative)")
    agent_p.add_argument("--task", required=True, help="Task directory or task.yaml")
    agent_p.add_argument("--model", default="openai/Qwen/Qwen3.5-35B-A3B", help="LLM model")
    agent_p.add_argument("--api-base", default="http://localhost:8000/v1", help="API base URL")
    agent_p.add_argument("--temperature", type=float, default=0.6)
    agent_p.add_argument("--max-tokens", type=int, default=16384)
    agent_p.add_argument("--max-iterations", type=int, default=5, help="Max tool-calling iterations")
    agent_p.add_argument("--output", help="Output directory")

    # run-all
    all_p = subparsers.add_parser("run-all", help="Evaluate all tasks from solutions dir")
    all_p.add_argument("--solutions-dir", required=True, help="Directory with solutions")
    all_p.add_argument("--filter", action="append", default=[],
                       help="Filter (e.g. task_class=3, backend=class1_make)")
    all_p.add_argument("--output", default="results/", help="Output directory")
    all_p.add_argument("--no-perf", action="store_true")
    all_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "prompt": cmd_prompt,
        "generate": cmd_generate,
        "run": cmd_run,
        "eval": cmd_eval,
        "agent": cmd_agent,
        "run-all": cmd_run_all,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
