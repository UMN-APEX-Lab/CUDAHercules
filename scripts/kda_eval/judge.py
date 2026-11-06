#!/usr/bin/env python3
"""Judge for sandboxed agent sessions (runs OUTSIDE the sandbox, cf env).

Serves one task over a unix socket that is bind-mounted into the sandbox at
/run/kh/judge.sock. The agent never sees the task directory, harness,
reference or evaluator code; it only receives the same feedback that the
tool-augmented controller (scripts/eval_agent.py) gives a model:
static-check result, compile errors, correctness diffs, solution/reference
timings and speedup.

Budget enforcement lives here, not in the agent: a hard cap on evaluations
and a wall-clock deadline. Every submission is archived under
<run_dir>/submissions/ and logged to submissions.jsonl.

Protocol: one JSON object per line, request -> response.
  {"op": "status"}
  {"op": "submit", "mode": "full"|"check", "content": "...", "source_path": "..."}
"""

import argparse
import json
import os
import socketserver
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import eval_agent as ea  # noqa: E402
from cuda_hercules.result import TaskStatus  # noqa: E402
from cuda_hercules.runner import discover_tasks  # noqa: E402
from cuda_hercules.static_checker import LIBRARY_RULES_PROMPT, validate_cuda_solution  # noqa: E402
from cuda_hercules.utils import get_project_root  # noqa: E402

MAX_ERROR_CHARS = 8000


def load_task(task_id: str):
    tasks = {t.task_id: t for t in discover_tasks(get_project_root())}
    if task_id not in tasks:
        raise SystemExit(f"unknown task: {task_id}")
    return tasks[task_id]


def _sanitize(text: str, sol_path: str = "", source_path: str = "") -> str:
    """Strip host paths so feedback does not reveal the harness layout, and
    point diagnostics for the judge's copy back at the agent's own file."""
    if sol_path and source_path:
        text = text.replace(sol_path, source_path)
    for p in (get_project_root(), os.path.expanduser("~")):
        text = text.replace(p, "<host>")
    if len(text) > MAX_ERROR_CHARS:
        half = MAX_ERROR_CHARS // 2
        text = text[:half] + "\n... [truncated] ...\n" + text[-half:]
    return text


class Judge:
    def __init__(self, config, run_dir: str, max_evals: int, time_limit_s: float,
                 eval_timeout_s: int, target_speedup: float):
        self.config = config
        self.run_dir = run_dir
        self.max_evals = max_evals
        self.time_limit_s = time_limit_s
        self.eval_timeout_s = eval_timeout_s
        self.target_speedup = target_speedup
        self.solution_file = config.runner.solution_file or "solution.cu"
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.evals_used = 0
        self.has_pass = False
        self.best_index = -1
        self.best_speedup = 0.0
        self.sub_root = os.path.join(run_dir, "submissions")
        os.makedirs(self.sub_root, exist_ok=True)
        self.log_path = os.path.join(run_dir, "submissions.jsonl")
        self.state_path = os.path.join(run_dir, "judge_state.json")
        self._write_state()

    # -- budget -----------------------------------------------------------
    def elapsed(self) -> float:
        return time.time() - self.t0

    def budget(self) -> dict:
        promoted = self.best_index > 0 and self.best_speedup >= self.target_speedup
        return {
            "elapsed_s": round(self.elapsed(), 1),
            "limit_s": self.time_limit_s,
            "evals_used": self.evals_used,
            "evals_limit": self.max_evals,
            "has_pass": self.has_pass,
            "best_speedup": round(self.best_speedup, 4),
            "best_submission": self.best_index,
            "target_speedup": self.target_speedup,
            "promoted": promoted,
            "summary": self._summary_line(),
        }

    def _summary_line(self) -> str:
        best = (f"best timed PASS: submission #{self.best_index} at {self.best_speedup:.3f}x"
                if self.best_index > 0 else "no timed PASS yet")
        return (f"Evaluations {self.evals_used}/{self.max_evals}; "
                f"time {self.elapsed() / 60:.1f}/{self.time_limit_s / 60:.0f} min; {best}; "
                f"target {self.target_speedup:.2f}x vs reference.")

    def _write_state(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump(self.budget(), f, indent=2)

    # -- ops --------------------------------------------------------------
    def status(self) -> dict:
        return {"text": self._summary_line(), "budget": self.budget()}

    def submit(self, mode: str, content: str, source_path: str) -> dict:
        with self.lock:
            if self.elapsed() >= self.time_limit_s:
                return {"rejected": True, "status": "REJECTED",
                        "text": "REJECTED: wall-clock budget exhausted. " + self._summary_line()}
            if self.evals_used >= self.max_evals:
                return {"rejected": True, "status": "REJECTED",
                        "text": "REJECTED: evaluation budget exhausted. " + self._summary_line()}
            self.evals_used += 1
            idx = self.evals_used
            resp = self._evaluate(idx, mode, content, source_path)
            self._write_state()
            resp["text"] += "\n\n" + self._summary_line()
            return resp

    def _evaluate(self, idx: int, mode: str, content: str, source_path: str) -> dict:
        cfg = self.config
        sub_dir = os.path.join(self.sub_root, f"{idx:03d}")
        os.makedirs(sub_dir, exist_ok=True)
        sol_path = os.path.join(sub_dir, self.solution_file)
        with open(sol_path, "w") as f:
            f.write(content)
        rec = {"index": idx, "mode": mode, "source_path": source_path,
               "submitted_at_s": round(self.elapsed(), 1), "lines": len(content.splitlines())}
        t_eval = time.time()

        check = validate_cuda_solution(
            content,
            blocked_patterns=cfg.anti_cheat.blocked_patterns or [],
            required_patterns=cfg.anti_cheat.required_patterns or [],
        )
        if not check.valid:
            errors = "; ".join(check.errors)
            rec.update(status="BLOCKED", error=f"Static check failed: {errors}")
            text = (f"Submission #{idx} ({mode}): BLOCKED by static check\n{errors}\n\n"
                    f"Library rules:\n{LIBRARY_RULES_PROMPT}")
            self._log(rec, t_eval)
            return {"status": "BLOCKED", "text": text}

        tr = ea._run_task_isolated(cfg, sol_path, timeout_sec=self.eval_timeout_s,
                                   measure_perf=(mode == "full"))
        status = tr.status.value if tr.status else "ERROR"
        rec.update(status=status, compiled=tr.compiled, correct=tr.correct,
                   speedup=tr.speedup, latency_mean_ms=tr.latency_mean_ms,
                   ref_latency_mean_ms=tr.ref_latency_mean_ms, per_size=tr.per_size or [],
                   error=(tr.error_msg or "")[:2000])

        lines = [f"Submission #{idx} ({mode}): {status}"]
        if tr.status == TaskStatus.PASS:
            self.has_pass = True
            if mode == "full":
                lines.append(f"- Speedup vs reference: {tr.speedup:.3f}x")
                if tr.ref_latency_mean_ms > 0:
                    lines.append(f"- Solution time: {tr.latency_mean_ms:.4f} ms, "
                                 f"reference time: {tr.ref_latency_mean_ms:.4f} ms")
                for e in tr.per_size or []:
                    lines.append(f"  - {e['name']}: {e['speedup']:.2f}x "
                                 f"(kernel={e['kernel_min_ms']:.4f}ms, ref={e['ref_min_ms']:.4f}ms)")
                if tr.speedup > self.best_speedup or self.best_index < 0:
                    self.best_speedup = tr.speedup
                    self.best_index = idx
                    lines.append("- New best timed submission.")
            else:
                lines.append("- Correctness passed (no timing in --check mode).")
        else:
            if tr.compiled and not tr.correct:
                lines.append("Compiled successfully but produced incorrect results.")
            if tr.error_msg:
                lines.append(_sanitize(tr.error_msg, sol_path, source_path))
            diff = ea._format_correctness_diff(tr)
            if diff:
                lines.append(diff)
        self._log(rec, t_eval)
        return {"status": status, "text": "\n".join(lines)}

    def _log(self, rec: dict, t_eval: float) -> None:
        rec["eval_s"] = round(time.time() - t_eval, 1)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        judge: Judge = self.server.judge
        try:
            req = json.loads(self.rfile.readline().decode())
            op = req.get("op")
            if op == "status":
                resp = judge.status()
            elif op == "submit":
                mode = req.get("mode", "full")
                if mode not in ("full", "check"):
                    raise ValueError(f"bad mode {mode!r}")
                resp = judge.submit(mode, req.get("content", ""), req.get("source_path", ""))
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as e:  # never crash the server on a bad request
            resp = {"status": "ERROR", "text": f"judge error: {e}"}
        self.wfile.write((json.dumps(resp) + "\n").encode())


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-evals", type=int, required=True)
    ap.add_argument("--time-limit", type=float, required=True, help="seconds")
    ap.add_argument("--eval-timeout", type=int, default=900)
    ap.add_argument("--target-speedup", type=float, default=1.0)
    args = ap.parse_args()

    judge = Judge(load_task(args.task_id), args.run_dir, args.max_evals,
                  args.time_limit, args.eval_timeout, args.target_speedup)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = _Server(args.socket, _Handler)
    os.chmod(args.socket, 0o600)
    server.judge = judge
    print(f"judge ready: {args.task_id}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
