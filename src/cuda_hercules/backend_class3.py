"""Class 3 Application-level backend.

Each Class 3 task has its own `eval_solution.py` script that handles:
  - Creating a temp copy of the task
  - Replacing solution files
  - Building and running
  - Reporting results as JSON

This backend simply delegates to `eval_solution.py <solution_path>`.
"""

import json
import os
import re
import signal
import subprocess
import sys

from .backend_base import Backend
from .task_schema import TaskConfig
from .result import PerfMetrics
from .utils import get_project_root


def _run_with_pgroup(cmd, *, cwd=None, env=None, timeout=None, shell=False, text=True):
    """subprocess.run equivalent that puts the child in its own session/pgroup
    and kills the entire group on timeout.

    Class 3 run.py/run.sh scripts launch compiled binaries (./build/train,
    benchmark_spmm, ...) as separate processes in the same process group.
    subprocess.run's built-in timeout only SIGKILLs the top-level child, so
    those grandchildren orphan and keep GPU memory locked. start_new_session
    moves the whole tree into one pgid; killpg then sweeps it cleanly.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return -9, stdout, stderr, True


class Class3AppBackend(Backend):
    def __init__(self):
        self._workdir = ""
        self._stdout = ""
        self._result = {}
        self._correct = False

    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        self._workdir = os.path.join(get_project_root(), config.runner.workdir)

    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        eval_script = os.path.join(self._workdir, "eval_solution.py")
        if not os.path.isfile(eval_script):
            return True, ""

        solution_path = getattr(config, "_solution_path", "") or solution_path
        if not solution_path:
            return False, "No solution path provided"

        env = os.environ.copy()
        env.update(config.runner.env)

        cmd = [sys.executable, eval_script, solution_path]
        rc, stdout, stderr, timed_out = _run_with_pgroup(
            cmd, cwd=self._workdir, env=env, timeout=config.runner.timeout_sec,
        )
        if timed_out:
            self._stdout = ""
            self._result = {
                "compiled": False,
                "correct": False,
                "error": f"eval_solution.py timed out ({config.runner.timeout_sec}s)",
            }
            self._correct = False
            return False, self._result["error"]

        self._stdout = (stdout or "") + (stderr or "")
        self._result = self._extract_last_json_object(self._stdout)
        if not isinstance(self._result, dict) or not self._result:
            error = (self._stdout or "")[-2000:] or "eval_solution.py produced no JSON result"
            self._result = {"compiled": False, "correct": False, "error": error}
            self._correct = False
            return False, error

        compiled = bool(self._result.get("compiled", False))
        self._correct = bool(self._result.get("correct", False))
        error = self._result.get("error", "")
        if not compiled and not error:
            error = self._result.get("output", self._stdout[-2000:])
        return compiled, error if not compiled else ""

    @staticmethod
    def _extract_last_json_object(text: str) -> dict:
        """Extract the last complete JSON object from mixed stdout/stderr text."""
        decoder = json.JSONDecoder()
        best = {}
        idx = 0
        n = len(text)

        while idx < n:
            brace = text.find("{", idx)
            if brace == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, brace)
            except json.JSONDecodeError:
                idx = brace + 1
                continue
            if isinstance(obj, dict):
                best = obj
            idx = end
        return best

    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        # Find eval_solution.py in the task directory
        eval_script = os.path.join(self._workdir, "eval_solution.py")
        if not os.path.isfile(eval_script):
            # Fallback: run run.py directly (no solution replacement)
            return self._run_fallback(config)
        if not self._result:
            compiled, error = self.build(config, getattr(config, "_solution_path", ""))
            if not compiled:
                return False, {}, error

        compiled = self._result.get("compiled", False)
        correct = self._result.get("correct", False)
        self._correct = correct

        detail = {}
        if "kernel_time_ms" in self._result:
            detail["kernel_time_ms"] = self._result["kernel_time_ms"]

        error = self._result.get("error", "")
        if not correct and not error:
            error = self._result.get("output", self._stdout[-2000:])

        return correct, detail, error

    def _run_fallback(self, config: TaskConfig) -> tuple[bool, dict, str]:
        """Fallback: run run.py directly (for tasks without eval_solution.py)."""
        env = os.environ.copy()
        env.update(config.runner.env)

        cmd = config.execute.cmd
        rc, stdout, stderr, timed_out = _run_with_pgroup(
            cmd, cwd=self._workdir, env=env, timeout=config.runner.timeout_sec, shell=True,
        )
        if timed_out:
            return False, {}, f"Execution timed out ({config.runner.timeout_sec}s)"

        self._stdout = (stdout or "") + (stderr or "")
        correct = rc == config.execute.success.exit_code
        if config.execute.success.stdout_regex:
            correct = correct and bool(
                re.search(config.execute.success.stdout_regex, self._stdout, re.MULTILINE))

        self._correct = correct
        return correct, {}, self._stdout[-2000:] if not correct else ""

    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        if not self._correct or not config.performance.enabled:
            return PerfMetrics()

        metrics = PerfMetrics()

        # Try from eval_solution.py result
        if "kernel_time_ms" in self._result:
            kernel_time_ms = float(self._result["kernel_time_ms"])
            metrics.latency_mean_ms = kernel_time_ms
            metrics.latency_min_ms = kernel_time_ms
            metrics.speedup = float(self._result.get("speedup", 0.0) or 0.0)

            ref_time_ms = self._result.get("ref_time_ms")
            if ref_time_ms is not None:
                ref_time_ms = float(ref_time_ms)
                metrics.ref_latency_mean_ms = ref_time_ms
                metrics.ref_latency_min_ms = ref_time_ms

            if metrics.speedup <= 0 and kernel_time_ms > 0 and metrics.ref_latency_mean_ms > 0:
                metrics.speedup = metrics.ref_latency_mean_ms / kernel_time_ms
            return metrics

        # Fallback: parse from stdout
        parser = config.performance.parser
        kernel_regex = parser.get("kernel_time_ms_regex", "")
        if kernel_regex:
            m = re.search(kernel_regex, self._stdout)
            if m:
                metrics.latency_mean_ms = float(m.group(1))
                metrics.latency_min_ms = metrics.latency_mean_ms

        ref_regex = parser.get("ref_time_ms_regex", "")
        if ref_regex:
            m = re.search(ref_regex, self._stdout)
            if m:
                metrics.ref_latency_mean_ms = float(m.group(1))
                metrics.ref_latency_min_ms = metrics.ref_latency_mean_ms

        speedup_regex = parser.get("speedup_regex", "")
        if speedup_regex:
            m = re.search(speedup_regex, self._stdout)
            if m:
                metrics.speedup = float(m.group(1))

        if metrics.speedup <= 0 and metrics.latency_mean_ms > 0 and metrics.ref_latency_mean_ms > 0:
            metrics.speedup = metrics.ref_latency_mean_ms / metrics.latency_mean_ms

        return metrics

    def cleanup(self, config: TaskConfig) -> None:
        pass
