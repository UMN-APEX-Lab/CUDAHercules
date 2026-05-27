"""Unified task runner.

Loads task.yaml, checks GPU arch, dispatches to the appropriate backend,
and returns normalized TaskResult.
"""

import math
import os

import torch

from .backend_base import resolve_backend
from .result import TaskResult, TaskStatus
from .task_schema import TaskConfig, load_task_config

UNSUPPORTED_ARCH_PREFIX = "UNSUPPORTED_ARCH:"


def get_gpu_sm() -> int:
    """Get current GPU SM capability (e.g. 80, 90, 100, 120)."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def discover_tasks(
    project_root: str,
    filters: dict | None = None,
) -> list[TaskConfig]:
    """Find all tasks with task.yaml under tasks/, optionally filtered."""
    tasks_dir = os.path.join(project_root, "tasks")
    configs = []

    for dirpath, dirs, files in os.walk(tasks_dir):
        if "task.yaml" not in files:
            continue
        yaml_path = os.path.join(dirpath, "task.yaml")
        try:
            config = load_task_config(yaml_path)
            config._yaml_dir = dirpath  # for description.txt lookup in arch variants
            configs.append(config)
        except Exception as e:
            print(f"Warning: failed to load {yaml_path}: {e}")

    if filters:
        configs = _apply_filters(configs, filters)

    configs.sort(key=lambda c: c.task_id)
    return configs


def _apply_filters(configs: list[TaskConfig], filters: dict) -> list[TaskConfig]:
    result = configs

    if "level" in filters:
        level = int(filters["level"])
        result = [c for c in result if c.task_class == level]

    if "task_class" in filters:
        tc = int(filters["task_class"])
        result = [c for c in result if c.task_class == tc]

    if "domain" in filters:
        result = [c for c in result if c.domain == filters["domain"]]

    if "backend" in filters:
        result = [c for c in result if c.runner.backend == filters["backend"]]

    if "min_sm_lte" in filters:
        sm = int(filters["min_sm_lte"])
        result = [c for c in result if c.hardware.min_sm <= sm]

    if "tag" in filters:
        result = [c for c in result if filters["tag"] in c.tags]

    if "arch" in filters:
        # Filter by architecture subdirectory: general, hopper, blackwell
        # Matches middle segment (class1/hopper/task) or trailing segment (class3/llmc/hopper)
        arch = filters["arch"].lower()
        result = [c for c in result if f"/{arch}/" in c.task_id
                  or c.task_id.startswith(f"{arch}/")
                  or c.task_id.endswith(f"/{arch}")]

    return result


def parse_filters(filter_args: list[str]) -> dict:
    """Parse --filter args like 'level=2', 'min_sm<=90'."""
    filters = {}
    for f in filter_args:
        if "<=" in f:
            key, val = f.split("<=", 1)
            filters[f"{key.strip()}_lte"] = val.strip()
        elif "=" in f:
            key, val = f.split("=", 1)
            filters[key.strip()] = val.strip()
    return filters


def _is_unsupported_arch_error(error_msg: str) -> bool:
    return bool(error_msg) and error_msg.startswith(UNSUPPORTED_ARCH_PREFIX)


def run_task(
    config: TaskConfig,
    solution_path: str,
    measure_perf: bool = True,
    verbose: bool = True,
) -> TaskResult:
    """Run a single task and return the unified result."""
    result = TaskResult(
        task_id=config.task_id,
        domain=config.domain,
        level=config.task_class,
        min_sm=config.hardware.min_sm,
    )

    # Architecture check
    gpu_sm = get_gpu_sm()
    if gpu_sm > 0 and gpu_sm < config.hardware.min_sm:
        result.status = TaskStatus.SKIP_ARCH
        result.error_msg = f"GPU SM {gpu_sm} < required SM {config.hardware.min_sm}"
        if verbose:
            print(f"[{config.task_id}] SKIP_ARCH: {result.error_msg}")
        return result

    backend = resolve_backend(config.runner.backend)

    # Attach solution_path to config for backends that need it
    config._solution_path = solution_path
    config._measure_perf = measure_perf

    try:
        # Prepare
        if verbose:
            print(f"[{config.task_id}] Preparing...")
        backend.prepare(config, solution_path)

        # Build
        if verbose:
            print(f"[{config.task_id}] Building...")
        compiled, compile_err = backend.build(config, solution_path)
        result.compiled = compiled
        if not compiled:
            result.error_msg = compile_err
            result.status = TaskStatus.SKIP_ARCH if _is_unsupported_arch_error(compile_err) else TaskStatus.FAIL
            if verbose:
                prefix = "SKIP_ARCH" if result.status == TaskStatus.SKIP_ARCH else "BUILD FAILED"
                print(f"[{config.task_id}] {prefix}: {compile_err[:200]}")
            return result

        # Correctness
        if verbose:
            print(f"[{config.task_id}] Checking correctness...")
        correct, detail, err = backend.run_correctness(config)
        result.correct = correct
        result.correctness_detail = detail
        if err:
            result.error_msg = err
            if _is_unsupported_arch_error(err):
                result.status = TaskStatus.SKIP_ARCH
                if verbose:
                    print(f"[{config.task_id}] SKIP_ARCH: {err[:200]}")
                return result

        # Performance
        if measure_perf and correct and config.performance.enabled:
            if verbose:
                print(f"[{config.task_id}] Measuring performance...")
            perf = backend.run_performance(config)
            result.speedup = perf.speedup
            result.latency_mean_ms = perf.latency_mean_ms
            result.latency_std_ms = perf.latency_std_ms
            result.latency_min_ms = perf.latency_min_ms
            result.latency_p90_ms = perf.latency_p90_ms
            result.ref_latency_mean_ms = perf.ref_latency_mean_ms
            result.per_size = perf.per_size

        result.status = TaskStatus.PASS if correct else TaskStatus.FAIL
        if verbose:
            status = "PASS" if correct else "FAIL"
            print(f"[{config.task_id}] {status}")
            if result.speedup > 0:
                if result.per_size:
                    parts = " | ".join(f"{e['name']}: {e['speedup']:.2f}x" for e in result.per_size)
                    print(f"  Speedup: {result.speedup:.2f}x (avg)  [{parts}]")
                else:
                    print(f"  Speedup: {result.speedup:.2f}x")

    except Exception as e:
        result.status = TaskStatus.ERROR
        result.error_msg = str(e)
        if verbose:
            print(f"[{config.task_id}] ERROR: {e}")
    finally:
        try:
            backend.cleanup(config)
        except Exception:
            pass

    return result


def compute_summary(results: list[TaskResult]) -> dict:
    """Compute aggregate summary from results using BenchmarkScore."""
    from .score import compute_scores, format_score
    from dataclasses import asdict

    score = compute_scores(results)
    return asdict(score)
