"""Class 4 Frontier/Unsolved challenge backend.

Same temp-copy mechanism as Class 3.
Tasks with no known hand-written CUDA solution.
"""

import os
import re
import signal
import shutil
import subprocess
import tempfile

from .backend_base import Backend
from .task_schema import TaskConfig
from .result import PerfMetrics
from .utils import get_project_root


def _run_with_pgroup(cmd, *, cwd=None, env=None, timeout=None, shell=False, text=True):
    """Run a command in its own process group and kill the whole tree on timeout."""
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


class Class4ChallengeBackend(Backend):
    def __init__(self):
        self._workdir = ""
        self._tmp_workdir = ""
        self._stdout = ""
        self._correct = False

    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        self._workdir = os.path.join(get_project_root(), config.runner.workdir)

        # Create temp copy
        self._tmp_workdir = tempfile.mkdtemp(prefix="cuda_hercules_c4_")
        shutil.copytree(self._workdir, os.path.join(self._tmp_workdir, "task"),
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(
                            "build", ".torch_extensions", "__pycache__", "*.o", "*.so"
                        ))
        self._run_dir = os.path.join(self._tmp_workdir, "task")

        # Copy LLM solution into temp copy
        sol_file = config.runner.solution_file
        if sol_file and os.path.isfile(solution_path):
            dest = os.path.join(self._run_dir, sol_file)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(solution_path, dest)

    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        cmd = config.build.cmd
        if not cmd:
            return True, ""

        env = os.environ.copy()
        env.update(config.runner.env)
        env["CUDA_HERCULES_ROOT"] = get_project_root()

        rc, stdout, stderr, timed_out = _run_with_pgroup(
            cmd,
            cwd=self._run_dir,
            env=env,
            timeout=config.runner.timeout_sec,
            shell=True,
        )
        self._stdout = (stdout or "") + (stderr or "")
        if timed_out:
            return False, f"Build timed out ({config.runner.timeout_sec}s)"
        if rc != 0:
            return False, self._stdout[-2000:] or f"Build failed with exit code {rc}"
        return True, ""

    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        env = os.environ.copy()
        env.update(config.runner.env)
        env["CUDA_HERCULES_ROOT"] = get_project_root()

        cmd = config.execute.cmd
        try:
            rc, stdout, stderr, timed_out = _run_with_pgroup(
                cmd, shell=True,
                cwd=self._run_dir,
                timeout=config.runner.timeout_sec,
                env=env,
            )
            if timed_out:
                return False, {}, f"Execution timed out ({config.runner.timeout_sec}s)"
        except Exception as e:
            return False, {}, str(e)

        self._stdout = (stdout or "") + (stderr or "")

        correct = rc == config.execute.success.exit_code
        if config.execute.success.stdout_regex:
            correct = correct and bool(
                re.search(config.execute.success.stdout_regex, self._stdout, re.MULTILINE)
            )

        self._correct = correct
        detail = {}
        for m in re.finditer(r'(\w[\w\s]+?):\s+([\d.]+)\s+(?:ms|TFLOPS|x)', self._stdout):
            detail[m.group(1).strip()] = float(m.group(2))

        error = self._stdout[-2000:] if not correct else ""
        return correct, detail, error

    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        if not self._correct or not config.performance.enabled:
            return PerfMetrics()

        metrics = PerfMetrics()
        parser = config.performance.parser

        kernel_regex = parser.get("kernel_time_ms_regex", "")
        if kernel_regex:
            m = re.search(kernel_regex, self._stdout)
            if m:
                metrics.latency_mean_ms = float(m.group(1))
                metrics.latency_min_ms = metrics.latency_mean_ms

        baseline_regex = parser.get("baseline_time_ms_regex", "")
        if baseline_regex:
            m = re.search(baseline_regex, self._stdout)
            if m:
                metrics.ref_latency_mean_ms = float(m.group(1))
                metrics.ref_latency_min_ms = metrics.ref_latency_mean_ms
                if metrics.latency_min_ms > 0:
                    metrics.speedup = metrics.ref_latency_min_ms / metrics.latency_min_ms

        return metrics

    def cleanup(self, config: TaskConfig) -> None:
        if self._tmp_workdir and os.path.isdir(self._tmp_workdir):
            shutil.rmtree(self._tmp_workdir, ignore_errors=True)
        self._tmp_workdir = ""
