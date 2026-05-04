"""Class 1 Make-based backend.

Executes tasks via Makefile (make test / ./test).
Correctness from exit code + stdout regex.
Performance parsed from stdout timing lines.

Uses a temp copy of the task directory so the original is never modified.
Supports cached reference benchmarks for speedup calculation.
"""

import json
import os
import re
import shutil
import shlex
import subprocess
import tempfile

import torch

from .backend_base import Backend
from .task_schema import TaskConfig
from .result import PerfMetrics
from .utils import get_project_root

UNSUPPORTED_ARCH_PREFIX = "UNSUPPORTED_ARCH:"

# Module-level cache for reference benchmarks
_ref_benchmarks_c1 = None


def _load_ref_benchmarks_c1():
    """Load pre-computed reference benchmarks for the current GPU."""
    global _ref_benchmarks_c1
    if _ref_benchmarks_c1 is not None:
        return _ref_benchmarks_c1

    root = get_project_root()
    bench_dir = os.path.join(root, "reference_benchmarks")
    if not os.path.isdir(bench_dir):
        _ref_benchmarks_c1 = {}
        return _ref_benchmarks_c1

    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().split("\n")[0].strip().replace(" ", "_")
    except Exception:
        gpu_name = ""

    bench_file = os.path.join(bench_dir, f"{gpu_name}.json")
    if os.path.exists(bench_file):
        with open(bench_file) as f:
            data = json.load(f)
        _ref_benchmarks_c1 = data.get("benchmarks", {})
    else:
        _ref_benchmarks_c1 = {}

    return _ref_benchmarks_c1


def _is_class1_general_task(config: TaskConfig) -> bool:
    return config.task_id.startswith("class1/general/")


def _get_gpu_sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _current_arch_make_vars(target_sm: int) -> dict[str, str]:
    arch = f"-gencode arch=compute_{target_sm},code=sm_{target_sm}"
    arch_ptx = f"-gencode arch=compute_{target_sm},code=compute_{target_sm}"
    return {
        "ARCH": arch,
        "ARCH_PTX": arch_ptx,
    }


def _format_make_cmd(build_cmd: str, make_vars: dict[str, str]) -> list[str]:
    cmd = shlex.split(build_cmd)
    if not cmd:
        return cmd
    if os.path.basename(cmd[0]) != "make":
        return cmd
    for key, value in make_vars.items():
        cmd.append(f"{key}={value}")
    return cmd


def _looks_like_arch_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    return any(
        pattern in text for pattern in [
            "no kernel image is available for execution on the device",
            "invalid device function",
            "unsupported gpu architecture",
            "architecture is not supported",
            "is not defined for option 'gpu-architecture'",
        ]
    )


class Class1MakeBackend(Backend):
    def __init__(self):
        self._workdir = ""      # original task dir (read-only)
        self._tmp_workdir = ""  # temp copy where we build and run
        self._stdout = ""
        self._correct = False
        self._make_vars = {}

    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        self._workdir = os.path.join(get_project_root(), config.runner.workdir)

        # Create temp copy of the task directory
        self._tmp_workdir = tempfile.mkdtemp(prefix="cuda_hercules_c1_")
        shutil.copytree(self._workdir, os.path.join(self._tmp_workdir, "task"),
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("ref", "test", "*.o"))
        self._run_dir = os.path.join(self._tmp_workdir, "task")

        # Copy LLM solution into temp copy
        sol_file = config.runner.solution_file or "solution.h"
        dest = os.path.join(self._run_dir, sol_file)
        if os.path.isfile(solution_path):
            shutil.copy2(solution_path, dest)

    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        # Provide absolute paths for reference_sources so temp-dir builds work
        root = get_project_root()
        env = os.environ.copy()
        ref_sources = os.path.join(root, "reference_sources")
        env["CUTLASS_DIR"] = os.path.join(ref_sources, "cutlass")
        env["TK_DIR"] = os.path.join(ref_sources, "ThunderKittens")
        env["DEEPGEMM_DIR"] = os.path.join(ref_sources, "deep_gemm")
        build_cmd = config.build.cmd
        build_cmd_args = shlex.split(build_cmd)

        self._make_vars = {}
        target_sm = 0
        if _is_class1_general_task(config):
            target_sm = _get_gpu_sm()
            if target_sm > 0:
                self._make_vars = _current_arch_make_vars(target_sm)
                build_cmd_args = _format_make_cmd(build_cmd, self._make_vars)
                # Validate the task itself against the current GPU target arch
                # before compiling the LLM solution. This avoids treating a
                # task-level arch incompatibility as a model code failure.
                ref_cmd = ["make", "ref"] + [f"{k}={v}" for k, v in self._make_vars.items()]
                try:
                    ref_proc = subprocess.run(
                        ref_cmd,
                        cwd=self._run_dir,
                        capture_output=True,
                        text=True,
                        timeout=config.runner.timeout_sec,
                        env=env,
                    )
                except subprocess.TimeoutExpired:
                    return False, (
                        f"{UNSUPPORTED_ARCH_PREFIX} class1/general reference build timed out "
                        f"for target sm_{target_sm}"
                    )
                if ref_proc.returncode != 0:
                    err = ref_proc.stderr[-2000:] if ref_proc.stderr else "Unknown reference compilation error"
                    return False, (
                        f"{UNSUPPORTED_ARCH_PREFIX} class1/general reference build failed for target "
                        f"sm_{target_sm}: {err}"
                    )

        try:
            proc = subprocess.run(
                build_cmd_args,
                cwd=self._run_dir,
                capture_output=True,
                text=True,
                timeout=config.runner.timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"

        if proc.returncode != 0:
            err = proc.stderr[-2000:] if proc.stderr else "Unknown compilation error"
            return False, err
        return True, ""

    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        env = os.environ.copy()
        env.update(config.runner.env)

        # Kernel execution timeout: 10 min (overrides yaml timeout_sec, which also
        # controls compile step). Only the execute step gets the bump.
        exec_timeout = max(config.runner.timeout_sec, 600)
        try:
            proc = subprocess.run(
                shlex.split(config.execute.cmd),
                cwd=self._run_dir,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, {}, f"Execution timed out ({exec_timeout}s)"

        self._stdout = proc.stdout

        correct = proc.returncode == config.execute.success.exit_code
        if config.execute.success.stdout_regex:
            correct = correct and bool(
                re.search(config.execute.success.stdout_regex, proc.stdout, re.MULTILINE)
            )

        self._correct = correct
        error = proc.stderr if not correct else ""
        if not correct and _is_class1_general_task(config) and _looks_like_arch_error(error):
            target_sm = _get_gpu_sm()
            error = (
                f"{UNSUPPORTED_ARCH_PREFIX} class1/general runtime failed on target "
                f"sm_{target_sm}: {error}"
            )
        return correct, {}, error

    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        if not self._correct or not config.performance.enabled:
            return PerfMetrics()

        metrics = PerfMetrics()

        # Check for cached reference benchmarks
        cached = _load_ref_benchmarks_c1().get(config.task_id, {})
        cached_per_size = cached.get("per_size", [])

        # Parse per-size blocks: "=== name ===", then Ref/Kernel/Speedup lines
        size_pattern = re.compile(
            r"===\s*(.+?)\s*===.*?"
            r"Ref time:.*?min:\s*([\d.]+)\s*ms.*?"
            r"Kernel time:.*?min:\s*([\d.]+)\s*ms.*?"
            r"Speedup:\s*([\d.]+)x",
            re.DOTALL,
        )

        per_size = []
        for i, m in enumerate(size_pattern.finditer(self._stdout)):
            ref_min = float(m.group(2))
            # Override ref time with cached benchmark if available
            if i < len(cached_per_size):
                ref_min = cached_per_size[i]["ref_min_ms"]

            kernel_min = float(m.group(3))
            speedup = ref_min / kernel_min if kernel_min > 0 else 0.0
            entry = {
                "name": m.group(1).strip(),
                "ref_min_ms": ref_min,
                "kernel_min_ms": kernel_min,
                "speedup": speedup,
            }
            per_size.append(entry)

        metrics.per_size = per_size

        if per_size:
            # Final speedup = average across all sizes
            metrics.speedup = sum(e["speedup"] for e in per_size) / len(per_size)
            # Use last (largest) size for latency fields
            metrics.latency_min_ms = per_size[-1]["kernel_min_ms"]
            metrics.ref_latency_min_ms = per_size[-1]["ref_min_ms"]
        else:
            # Fallback: single-size parsing (legacy)
            parser = config.performance.parser
            kernel_regex = parser.get("kernel_time_ms_regex", "")
            if kernel_regex:
                m = re.search(kernel_regex, self._stdout)
                if m:
                    metrics.latency_mean_ms = float(m.group(1))
            min_match = re.search(
                r"Kernel time:.*?min:\s*([\d.]+)\s*ms", self._stdout
            )
            if min_match:
                metrics.latency_min_ms = float(min_match.group(1))
            ref_regex = parser.get("ref_time_ms_regex", "")
            if ref_regex:
                m = re.search(ref_regex, self._stdout)
                if m:
                    metrics.ref_latency_mean_ms = float(m.group(1))
            ref_min_match = re.search(
                r"Ref time:.*?min:\s*([\d.]+)\s*ms", self._stdout
            )
            if ref_min_match:
                metrics.ref_latency_min_ms = float(ref_min_match.group(1))
            if metrics.latency_min_ms > 0 and metrics.ref_latency_min_ms > 0:
                metrics.speedup = metrics.ref_latency_min_ms / metrics.latency_min_ms

        return metrics

    def cleanup(self, config: TaskConfig) -> None:
        if self._tmp_workdir and os.path.isdir(self._tmp_workdir):
            shutil.rmtree(self._tmp_workdir, ignore_errors=True)
        self._tmp_workdir = ""
