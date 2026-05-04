"""Class 2 def.py-based backend.

Uses the existing eval.py pipeline: compile via pybind11, compare tensors,
time with CUDA events.

Supports cached reference benchmarks: if reference_benchmarks/<gpu>.json exists,
skips reference timing and uses pre-measured values.
"""

import json
import os

from .backend_base import Backend
from .task_schema import TaskConfig
from .result import PerfMetrics
from .utils import get_project_root

# Module-level cache for reference benchmarks
_ref_benchmarks = None


def _load_ref_benchmarks():
    """Load pre-computed reference benchmarks for the current GPU."""
    global _ref_benchmarks
    if _ref_benchmarks is not None:
        return _ref_benchmarks

    import subprocess
    root = get_project_root()
    bench_dir = os.path.join(root, "reference_benchmarks")
    if not os.path.isdir(bench_dir):
        _ref_benchmarks = {}
        return _ref_benchmarks

    # Find benchmark file matching current GPU
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
        _ref_benchmarks = data.get("benchmarks", {})
    else:
        _ref_benchmarks = {}

    return _ref_benchmarks


class Class2DefpyBackend(Backend):
    def __init__(self):
        self._eval_result = None  # KernelExecResult from eval.py
        self._workdir = ""

    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        self._workdir = os.path.join(get_project_root(), config.runner.workdir)

    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        """Run the full eval pipeline and cache the result."""
        from .eval import eval_kernel

        # Look up cached reference timing
        benchmarks = _load_ref_benchmarks()
        cached_ref_time = 0.0
        if config.task_id in benchmarks:
            cached_ref_time = benchmarks[config.task_id].get("ref_mean_ms", 0.0)

        tol = config.correctness.tolerances
        self._eval_result = eval_kernel(
            task_dir=self._workdir,
            solution_cu=solution_path,
            measure_performance=config.performance.enabled,
            atol=tol.get("atol", 1e-2),
            rtol=tol.get("rtol", 1e-2),
            cached_ref_time_ms=cached_ref_time,
        )
        return self._eval_result.compiled, self._eval_result.compilation_error

    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        if self._eval_result is None:
            return False, {}, "Build was not called"
        r = self._eval_result
        return r.correctness, r.per_output_correctness, r.runtime_error

    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        if self._eval_result is None:
            return PerfMetrics()

        r = self._eval_result
        metrics = PerfMetrics()

        if r.solution_time:
            metrics.latency_mean_ms = r.solution_time.mean_ms
            metrics.latency_std_ms = r.solution_time.std_ms
            metrics.latency_min_ms = r.solution_time.min_ms
            times = sorted(r.solution_time.all_times_ms)
            if times:
                metrics.latency_p90_ms = times[int(0.9 * len(times))]

        if r.reference_time:
            metrics.ref_latency_mean_ms = r.reference_time.mean_ms
            metrics.ref_latency_min_ms = r.reference_time.min_ms

        metrics.speedup = r.speedup
        return metrics

    def cleanup(self, config: TaskConfig) -> None:
        pass
