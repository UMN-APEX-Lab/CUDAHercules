"""
LangGraph ReAct Agent for CUDA-Hercules.

Iteratively generates, compiles, tests, and fixes CUDA solutions using tool calling.

Tools:
  - read_task_description: Read the full task prompt
  - list_task_files: List files in task directory
  - read_file: Read a specific file (reference code, template, etc.)
  - write_solution: Write CUDA code to the solution file
  - compile_and_test: Compile + run tests, return results/errors

Usage:
    cuda-hercules agent --task tasks/class2/general/layernorm_fwd_1024 \\
        --model openai/Qwen/Qwen3.5-35B-A3B \\
        --api-base http://localhost:8000/v1
"""

import os
import re
import shutil
import subprocess
import json
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .task_schema import load_task_config, TaskConfig
from .prompt_builder import build_prompt
from .utils import get_project_root


# ═══════════════════════════════════════════════════════════════════════
# Shared state (set before agent runs)
# ═══════════════════════════════════════════════════════════════════════

_state = {}


# ═══════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════

@tool
def read_task_description() -> str:
    """Read the full task description including function signature and requirements.
    Call this first to understand what you need to implement."""
    task_dir = _state["task_dir"]
    config = _state["config"]
    return build_prompt(task_dir, config, description_dir=_state.get("yaml_dir"))


@tool
def list_task_files() -> str:
    """List all files in the task directory. Useful to find reference code, templates, or headers."""
    task_dir = _state["task_dir"]
    files = []
    for root, dirs, filenames in os.walk(task_dir):
        dirs[:] = [d for d in dirs if d not in ('build', 'cmake-build-release', '__pycache__', 'data', 'ref_data')]
        for f in filenames:
            rel = os.path.relpath(os.path.join(root, f), task_dir)
            size = os.path.getsize(os.path.join(root, f))
            files.append(f"{rel} ({size} bytes)")
    return "\n".join(sorted(files))


@tool
def read_file(filename: str) -> str:
    """Read a file from the task directory. Use to examine reference code, solution templates, harness code, or headers.
    Examples: 'description.txt', 'solution.h', 'solution.cu', 'def.py', 'Makefile', 'harness.impl'"""
    task_dir = _state["task_dir"]
    path = os.path.join(task_dir, filename)
    if not os.path.isfile(path):
        return f"Error: file '{filename}' not found in task directory"
    with open(path) as f:
        content = f.read()
    if len(content) > 15000:
        content = content[:15000] + "\n\n... [truncated, file too long]"
    return content


@tool
def write_solution(code: str) -> str:
    """Write your CUDA solution code. Pass the complete .cu or .h file content.
    After writing, call compile_and_test() to verify it works."""
    sol_path = _state["solution_path"]
    with open(sol_path, "w") as f:
        f.write(code)
    lines = len(code.strip().splitlines())
    _state["write_count"] = _state.get("write_count", 0) + 1
    return f"Solution written ({lines} lines). Now call compile_and_test() to check it."


@tool
def compile_and_test() -> str:
    """Compile the current solution and run correctness tests.
    Returns: compilation output, test results, errors.
    If it fails, analyze the error and fix your code with write_solution(), then test again."""
    config = _state["config"]
    sol_path = _state["solution_path"]
    task_dir = _state["task_dir"]
    root = get_project_root()
    workdir = os.path.join(root, config.runner.workdir) if config.runner.workdir else task_dir
    backend = config.runner.backend

    try:
        if backend == "class1_make":
            return _run_class1(config, sol_path, workdir)
        elif backend in ("class3_app", "class4_challenge"):
            return _run_class3(config, workdir)
        elif backend == "class2_defpy":
            return _run_class2(config, sol_path, task_dir)
        else:
            return f"Unknown backend: {backend}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: execution exceeded time limit"
    except Exception as e:
        return f"ERROR: {str(e)[:1000]}"


@tool
def profile_nsys() -> str:
    """Run Nsight Systems profiling on the compiled solution.
    Returns a summary of GPU kernel execution times, memory operations, and API calls.
    Call this AFTER compile_and_test() passes to identify performance bottlenecks."""
    config = _state["config"]
    root = get_project_root()
    workdir = os.path.join(root, config.runner.workdir) if config.runner.workdir else _state["task_dir"]
    env = os.environ.copy()
    env.update(config.runner.env)

    run_cmd = config.execute.cmd or "./test"
    nsys_cmd = f"nsys profile --stats=true --force-overwrite=true -o /tmp/cuda_hercules_nsys {run_cmd}"

    try:
        proc = subprocess.run(nsys_cmd, shell=True, cwd=workdir,
                              capture_output=True, text=True, timeout=120, env=env)
        output = proc.stdout + proc.stderr
        # Extract key stats
        lines = output.split('\n')
        relevant = [l for l in lines if any(k in l.lower() for k in
                    ['kernel', 'memcpy', 'memset', 'time(%)', 'avg', 'total', 'cuda api'])]
        if relevant:
            return "Nsight Systems Profile Summary:\n" + "\n".join(relevant[:50])
        return f"Nsight Systems output:\n{output[-3000:]}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: nsys profiling exceeded time limit"
    except Exception as e:
        return f"nsys error: {str(e)[:500]}"


@tool
def profile_ncu(kernel_name: str = "") -> str:
    """Run Nsight Compute profiling on GPU kernels.
    Provides detailed metrics: occupancy, memory throughput, compute utilization, warp stalls.
    Args:
        kernel_name: Optional regex to filter specific kernel. Empty = profile all kernels.
    Call this AFTER compile_and_test() passes to get detailed kernel performance data."""
    config = _state["config"]
    root = get_project_root()
    workdir = os.path.join(root, config.runner.workdir) if config.runner.workdir else _state["task_dir"]
    env = os.environ.copy()
    env.update(config.runner.env)

    run_cmd = config.execute.cmd or "./test"
    ncu_cmd = f"ncu --set full --csv"
    if kernel_name:
        ncu_cmd += f' --kernel-name "{kernel_name}"'
    ncu_cmd += f" {run_cmd}"

    try:
        proc = subprocess.run(ncu_cmd, shell=True, cwd=workdir,
                              capture_output=True, text=True, timeout=180, env=env)
        output = proc.stdout + proc.stderr
        # Extract key metrics
        lines = output.split('\n')
        relevant = [l for l in lines if any(k in l for k in
                    ['Achieved Occupancy', 'Memory Throughput', 'Compute (SM)',
                     'SM [%]', 'Memory [%]', 'Duration', 'Registers',
                     'Shared Memory', 'Warp Cycles'])]
        if relevant:
            return "Nsight Compute Metrics:\n" + "\n".join(relevant[:30])
        # Fallback: show raw output
        return f"Nsight Compute output:\n{output[-3000:]}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: ncu profiling exceeded time limit"
    except Exception as e:
        return f"ncu error: {str(e)[:500]}"


# ── Backend runners ───────────────────────────────────────────────────

def _run_class1(config, sol_path, workdir):
    dest = os.path.join(workdir, config.runner.solution_file or "solution.h")
    if os.path.abspath(sol_path) != os.path.abspath(dest):
        shutil.copy2(sol_path, dest)

    # Build
    build_cmd = config.build.cmd or "make test"
    proc = subprocess.run(build_cmd, shell=True, cwd=workdir,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return f"COMPILATION FAILED (exit {proc.returncode}):\n{proc.stderr[-2000:]}"

    # Run
    env = os.environ.copy()
    env.update(config.runner.env)
    run_cmd = config.execute.cmd or "./test"
    proc = subprocess.run(run_cmd, shell=True, cwd=workdir,
                          capture_output=True, text=True,
                          timeout=config.runner.timeout_sec, env=env)
    output = proc.stdout + proc.stderr
    passed = proc.returncode == 0
    if config.execute.success.stdout_regex:
        passed = passed and bool(re.search(config.execute.success.stdout_regex, output))

    if passed:
        _state["passed"] = True
    return f"{'PASSED' if passed else 'FAILED'} (exit {proc.returncode})\n\n{output[-3000:]}"


def _run_class2(config, sol_path, task_dir):
    try:
        from .eval import eval_kernel
        result = eval_kernel(task_dir, sol_path)
        parts = []
        if not result.compiled:
            return f"COMPILATION FAILED:\n{result.errors[-1] if result.errors else 'Unknown error'}"
        parts.append(f"Compiled: YES")
        if result.correct:
            _state["passed"] = True
            parts.append(f"PASSED - Correct!")
            parts.append(f"Speedup: {result.speedup:.2f}x")
            parts.append(f"Solution: {result.solution_time_ms:.4f} ms, Reference: {result.reference_time_ms:.4f} ms")
        else:
            parts.append("FAILED - Incorrect results")
            for err in result.errors[:3]:
                parts.append(f"Error: {err}")
        return "\n".join(parts)
    except Exception as e:
        return f"EVALUATION ERROR:\n{str(e)[:1500]}"


def _run_class3(config, workdir):
    env = os.environ.copy()
    env.update(config.runner.env)
    cmd = config.execute.cmd or "python run.py"
    proc = subprocess.run(cmd, shell=True, cwd=workdir,
                          capture_output=True, text=True,
                          timeout=config.runner.timeout_sec, env=env)
    output = re.sub(r'\x1b\[[0-9;]*m', '', proc.stdout + proc.stderr)
    passed = proc.returncode == 0
    if config.execute.success.stdout_regex:
        passed = passed and bool(re.search(config.execute.success.stdout_regex, output))
    if passed:
        _state["passed"] = True
    return f"{'PASSED' if passed else 'FAILED'} (exit {proc.returncode})\n\n{output[-3000:]}"


# ═══════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert CUDA kernel programmer. Your job is to write high-performance GPU code that passes correctness tests, then optimize it for maximum performance.

Workflow:
1. Call read_task_description() to understand the task
2. Optionally call list_task_files() and read_file() to examine reference code or templates
3. Call write_solution() with your complete CUDA implementation
4. Call compile_and_test() to check compilation and correctness
5. If there are errors, analyze them carefully, fix your code, then write_solution() and compile_and_test() again
6. Once correctness passes, use profile_nsys() to see overall GPU utilization and kernel times
7. Use profile_ncu() to get detailed kernel metrics (occupancy, memory throughput, warp stalls)
8. Based on profiling data, optimize your code and iterate

Tips:
- Always include all necessary headers (#include <cuda_runtime.h>, etc.)
- Use extern "C" linkage for the entry function
- Use shared memory, warp-level primitives, and coalesced memory access for performance
- Read error messages carefully - they tell you exactly what's wrong
- After correctness passes, profile before optimizing - measure first, then improve"""


def run_agent(
    task_dir: str,
    model: str = "openai/Qwen/Qwen3.5-35B-A3B",
    api_base: str = "http://localhost:8000/v1",
    temperature: float = 0.6,
    max_tokens: int = 16384,
    max_iterations: int = 5,
    output_dir: str = "",
    verbose: bool = True,
) -> dict:
    """Run the ReAct agent on a task."""

    # Load task config
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No task.yaml in {task_dir}")

    config = load_task_config(yaml_path)
    root = get_project_root()
    abs_task_dir = os.path.join(root, config.runner.workdir) if config.runner.workdir else os.path.abspath(task_dir)

    # Setup output
    if not output_dir:
        output_dir = os.path.join("results", config.task_id)
    os.makedirs(output_dir, exist_ok=True)

    sol_name = config.runner.solution_file or "solution.cu"
    sol_path = os.path.join(output_dir, sol_name)

    # Set shared state
    _state.clear()
    _state["task_dir"] = abs_task_dir
    _state["yaml_dir"] = os.path.dirname(os.path.abspath(yaml_path))
    _state["config"] = config
    _state["solution_path"] = sol_path
    _state["passed"] = False
    _state["write_count"] = 0

    # Create LLM
    model_name = model.replace("openai/", "") if model.startswith("openai/") else model
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_base:
        llm_kwargs["openai_api_base"] = api_base
        llm_kwargs["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "dummy")

    llm = ChatOpenAI(**llm_kwargs)

    # Create ReAct agent with LangGraph
    tools = [read_task_description, list_task_files, read_file, write_solution,
             compile_and_test, profile_nsys, profile_ncu]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    if verbose:
        print(f"Task:       {config.task_id}")
        print(f"Model:      {model}")
        print(f"Backend:    {config.runner.backend}")
        print(f"Max iters:  {max_iterations}")
        print(f"Output:     {output_dir}")
        print()

    # Run agent
    initial_message = (
        f"Solve CUDA task: {config.task_id}\n"
        f"Start by calling read_task_description() to understand the task."
    )

    config_dict = {"recursion_limit": max_iterations * 4}  # each iteration ≈ 2-4 steps

    step_count = 0
    try:
        for event in agent.stream(
            {"messages": [("user", initial_message)]},
            config=config_dict,
            stream_mode="updates",
        ):
            # Print progress
            for node_name, node_output in event.items():
                if not verbose:
                    continue

                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_preview = str(tc.get('args', ''))[:80]
                            print(f"  → {tc['name']}({args_preview})")
                    elif hasattr(msg, 'content') and msg.content:
                        content = str(msg.content)
                        if node_name == "tools":
                            # Tool result - show first line
                            first_line = content.split('\n')[0][:120]
                            print(f"    ← {first_line}")
                        elif len(content) > 200:
                            print(f"  LLM: {content[:200]}...")

                step_count += 1

            # Early exit if passed
            if _state.get("passed"):
                break

    except Exception as e:
        if verbose:
            print(f"\nAgent error: {str(e)[:500]}")

    passed = _state.get("passed", False)
    write_count = _state.get("write_count", 0)

    # Save log
    with open(os.path.join(output_dir, "agent_log.json"), "w") as f:
        json.dump({
            "task_id": config.task_id,
            "model": model,
            "passed": passed,
            "write_count": write_count,
            "steps": step_count,
        }, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
        print(f"Solution writes: {write_count}")
        print(f"Steps: {step_count}")
        if os.path.isfile(sol_path):
            print(f"Solution: {sol_path}")

    return {
        "passed": passed,
        "write_count": write_count,
        "steps": step_count,
        "solution_path": sol_path,
        "output_dir": output_dir,
    }
