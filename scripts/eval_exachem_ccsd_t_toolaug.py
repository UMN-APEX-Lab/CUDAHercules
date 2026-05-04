#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/exachem_ccsd_t.

Customized loop (patterned after eval_tcgnn_gcn_toolaug.py):
  - Baseline measured once at setup: 1 warmup + 3 timed runs of `python run.py`,
    mean of the "Kernel time:" value.
  - Planning phase (round 0): model produces a plan; no actions parsed.
  - Rounds 1..N: model may write/read/profile/update_plan. If any write_file
    happened, harness runs eval_solution.py and reports results.

This evaluator INTENTIONALLY strips MMA / Tensor-Core hints from the task
description shown to the LLM, so the model has to discover that FP64 Tensor
Core MMA is the right optimization strategy on its own.

Usage:
    python scripts/eval_exachem_ccsd_t_toolaug.py \
        --model "Qwen/Qwen3.5-122B-A10B" \
        --api-base http://134.84.150.135:8001/v1 --api-key EMPTY \
        --run-name exachem_tool_qwen
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.llm_api import query_server
from cuda_hercules.utils import get_project_root

# ── Task-specific constants ──────────────────────────────────────────────────
TASK_ID = "class3/exachem_ccsd_t/blackwell"
TASK_DIR_REL = "tasks/class3/exachem_ccsd_t"

SOLUTION_FILES = [
    "solution.cu",
    "ccsd_t_g2s_device_functions.cu",
    "tensor_core_helper.cuh",
]

# Files the LLM may read freely (interface, types, test driver, build system).
# Reference kernel implementations are NOT here.
READABLE_INFRA_FILES = {
    "src/kernel_interface.cuh",
    "src/standalone_common.cuh",
    "src/main.cu",
    "src/Makefile",
}

# Hide upstream reference implementations (enforced via anti-cheat pattern scan
# on write_file and also at eval_solution.py stage).
FORBIDDEN_PATTERNS = [
    (re.compile(r"src/ccsd_t_all_fused_gpu\.cu", re.IGNORECASE),
     "src/ccsd_t_all_fused_gpu.cu — reference kernel path"),
    (re.compile(r"src/ccsd_t_g2s_device_functions\.cu", re.IGNORECASE),
     "src/ccsd_t_g2s_device_functions.cu — reference G2S path"),
    (re.compile(r"src/tensor_core_helper\.cuh", re.IGNORECASE),
     "src/tensor_core_helper.cuh — reference helper path"),
    (re.compile(r"CUDA_HERCULES_PLACEHOLDER_SOLUTION", re.IGNORECASE),
     "placeholder marker — solution must actually be implemented"),
    (re.compile(r"CUDA_HERCULES_PLACEHOLDER_G2S", re.IGNORECASE),
     "placeholder marker — G2S must actually be implemented"),
    (re.compile(r"CUDA_HERCULES_PLACEHOLDER_TENSOR_CORE_HELPER", re.IGNORECASE),
     "placeholder marker — helper must actually be implemented"),
]

BASELINE_WARMUP = 1
BASELINE_RUNS = 3
MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
REF_RUN_TIMEOUT_SEC = 600
EVAL_TIMEOUT_SEC = 900
PROFILE_TIMEOUT_SEC = 900
# Hidden anti-cheat: a real FP64 tensor-contraction kernel over 3 test cases
# (H2O / Ubiquitin / Water-53, ~1.2M grid blocks total) cannot finish in under
# this many milliseconds. A hardcoded-energy cheat returns in ~0 ms.
KERNEL_TIME_FLOOR_MS = 100.0

# ── Stripped task description (overrides description.txt) ────────────────────
# The original description.txt leaks the MMA / Tensor-Core strategy. This
# stripped version keeps the problem statement and interface, but removes
# all hints about which algorithms/instructions to use.

STRIPPED_DESCRIPTION = """\
# Task: ExaChem CCSD(T) Fully-Fused GPU Kernel Optimization

Optimize the fully-fused CCSD(T) perturbative triples kernel from ExaChem, a
production quantum chemistry application.

## Overview

Coupled-cluster with singles, doubles, and perturbative triples — CCSD(T) — is
the "gold standard" of computational quantum chemistry. The perturbative
triples correction is the most compute-intensive step, scaling as O(N^7).

The kernel you are optimizing fuses all tensor-contraction equations into a
single GPU kernel launch. For a given set of orbital tile indices
(h1, h2, h3, p4, p5, p6) it computes three residual-equation groups:

  - Singles (S1): 9 equation variants — direct t1 × v2 products
  - Doubles-1 (D1): 9 equations × noab occupied blocks — t2 × v2 over h7
  - Doubles-2 (D2): 9 equations × nvab virtual blocks — t2 × v2 over p7

The kernel accumulates a 6-dimensional output tensor T3 and then divides by
orbital-energy denominators to produce the final energy contributions.

## Data type

**FP64 (double precision)** everywhere — required for quantum-chemistry
numerical accuracy. Mixed-precision is not acceptable.

## Solution files (all three may be edited)

- `solution.cu` — your main kernel and the `launch_ccsd_t_kernel()` driver.
  Must define a `__global__` kernel and be callable through the fixed C++
  signature declared in `src/kernel_interface.cuh`.
- `ccsd_t_g2s_device_functions.cu` — device helper functions that move tensor
  tiles from global memory into shared memory. Mix in stride/layout logic here.
- `tensor_core_helper.cuh` — any auxiliary device-side helpers you need
  (register layout helpers, inline PTX wrappers, small struct helpers, etc.).

The file names are historical. You are free to reorganize the code within
these three files however you like. If you don't need one of them, leave it as
a small no-op header so the build still succeeds.

## Interface

The driver function you implement must match exactly the signature in
`src/kernel_interface.cuh`. Read that header to see it. The benchmark harness
calls your driver for each orbital-block iteration and expects energy
contributions to be written to `dev_energies`.

## Test Configuration

Three test cases from real molecular systems (`ccsdt_tilesize = 40`):

| Test Case | Molecule          | noab | nvab | h dims     | p dims    | Grid Blocks |
|-----------|-------------------|------|------|------------|-----------|-------------|
| 1         | H2O / cc-pvdz     | 1    | 1    | 5,5,5      | 19,19,19  |       1,000 |
| 2         | Ubiquitin / 6-31g | 4    | 7    | 40,40,40   | 40,40,40  |   1,000,000 |
| 3         | Water-53 / cc-pvdz| 6    | 26   | 40,40,35   | 40,40,7   |     180,000 |

Total GPU memory used per test: up to ~12 GB for tensor data.
Reference total kernel time (all 3 cases combined): ~55 seconds.

## Correctness

Your run must print two energies:

  Energy_T:  <float>
  Energy_T5: <float>

Each must match the reference within **relative tolerance 1e-6**. Returning
zero or wildly wrong energies is rejected.

## Performance

Aim to minimize the total `Kernel time:` (summed across all three test cases).
The reference is the hand-written kernel that ships with this benchmark
(source hidden from you). Speedup = baseline_time / your_kernel_time.

## Allowed

- Any CUDA / PTX / device intrinsics available on SM80+.
- Any memory layout, tile size, thread-block shape, or launch configuration.
- Cooperative groups, async copies, cooperative SM primitives.

## Prohibited

- Using or linking against the upstream ExaChem source files (any file under
  `exachem/cc/ccsd_t/` from the original repo).
- Hardcoding the energies returned to the harness for the 3 test cases.
- Reading precomputed energy files.
"""


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA engineer optimizing a production quantum-chemistry
kernel inside a fixed tool-augmented evaluation harness.

The task description is sent in the planning-phase user message. Read it there.

## Phases

**Phase 0 (planning)** — The first user message sends the task description. Output
a free-text plan only; no JSON actions, no code blocks.

**Phase 1..N (implementation)** — Each round you respond with actions.

## Action format (strict)

Every action MUST start with a JSON object on its own line. Code/text blocks
without a preceding action header are ignored.

## Available actions

1. `write_file` — overwrite one of the three solution files.
   ```
   {"action":"write_file","path":"solution.cu"}
   ```cuda
   // full file contents
   ```
   Allowed paths:
     - solution.cu
     - ccsd_t_g2s_device_functions.cu
     - tensor_core_helper.cuh
   Same file at most once per round.

2. `read_file` — read a file. Allowed:
   - any of the three solution files (returns your latest version)
   - `src/kernel_interface.cuh` — fixed interface declaration
   - `src/standalone_common.cuh` — shared common types / macros
   - `src/main.cu` — the benchmark driver that calls your kernel
   - `src/Makefile` — build configuration
   Reference kernel sources are NOT readable.

3. `profile` — run a profiler over the built test binary (requires at least one
   prior round with `correct=True`). Two tools:
   - `nsys`: timeline + kernel summary
   - `ncu`: per-kernel metrics; requires `"kernel": "<regex>"`
   At most 2 profile calls per round.

4. `update_plan` — replace the stored plan (re-shown each round).
   ```
   {"action":"update_plan"}
   ```text
   revised plan text
   ```

## Round rules

- Round 1: any action allowed. A minimum first build needs `solution.cu`
  implementing the driver, plus non-empty stubs for the other two files so the
  Makefile compiles.
- Rounds 2+: any combination; executed in order.
- If the round contains any `write_file`, the harness runs the full eval.

## Evaluation

Each round with a `write_file` action triggers `eval_solution.py`, which
compiles your three files via the benchmark's Makefile, runs the 3-test-case
benchmark, and reports `compiled / correct / kernel_time / speedup`. Speedup
is relative to the baseline measured once at setup (3 runs, mean).
"""


# ── Action parser (shared pattern) ────────────────────────────────────────────

_JSON_ACTION_RE = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)


def _find_code_after(text: str, start_idx: int) -> tuple[str, int]:
    open_match = re.search(r"```[a-zA-Z0-9+_-]*\n", text[start_idx:])
    if not open_match:
        return "", len(text)
    open_end = start_idx + open_match.end()
    close = text.find("\n```", open_end)
    if close == -1:
        return "", len(text)
    return text[open_end:close], close + 4


def _take_code_until_next_json(text: str, start_idx: int) -> str:
    next_json = _JSON_ACTION_RE.search(text, start_idx)
    end = next_json.start() if next_json else len(text)
    chunk = text[start_idx:end].strip()
    if len(chunk) < 50:
        return ""
    code_markers = ("#include", "__global__", "__device__", "extern \"C\"",
                    "int main", "namespace ", "template ", "void ", "cudaError",
                    "struct ", "class ", "#pragma")
    if not any(mk in chunk for mk in code_markers):
        return ""
    return chunk


def parse_actions(message: str, fallback_round1: bool = False) -> list[dict]:
    actions = []
    for m in _JSON_ACTION_RE.finditer(message):
        raw = m.group(0)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            actions.append({"type": "invalid", "raw_json": raw, "error": f"JSON parse error: {e}"})
            continue
        atype = str(obj.get("action", "")).strip().lower()
        act = {"type": atype, "raw_json": raw}
        if atype == "write_file":
            act["path"] = str(obj.get("path", "")).strip()
            code, _ = _find_code_after(message, m.end())
            if not code:
                code = _take_code_until_next_json(message, m.end())
            act["code"] = code
            if not code:
                act["error"] = "write_file requires a fenced code block (or substantial source text) after the JSON."
        elif atype == "read_file":
            act["path"] = str(obj.get("path", "")).strip()
        elif atype == "profile":
            act["tool"] = str(obj.get("tool", "")).strip().lower()
            act["kernel"] = str(obj.get("kernel", "")).strip()
        elif atype == "update_plan":
            plan_text, _ = _find_code_after(message, m.end())
            act["plan"] = plan_text
            if not plan_text:
                act["error"] = "update_plan requires a fenced text block after the JSON."
        else:
            act["error"] = f"Unknown action '{atype}'. Must be write_file, read_file, profile, or update_plan."
        actions.append(act)

    if fallback_round1 and not actions:
        code, _ = _find_code_after(message, 0)
        if code.strip():
            actions.append({
                "type": "write_file",
                "path": "solution.cu",
                "code": code,
                "raw_json": "(synthesized: no JSON action header, assuming write_file solution.cu)",
            })
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="exachem_toolaug_"))
        self.work_dir = self.tmp_dir / "exachem_ccsd_t"
        self.written_files: dict[str, str] = {}
        self.rounds: list[dict] = []
        self.current_plan: str = ""
        self.baseline_time_ms: float = -1.0
        self.best_speedup: float = -1.0
        self.best_round: int = -1
        self.latest_build_ok: bool = False

    def setup(self, source_task_dir: Path):
        """Copy task dir into tmp; exclude build artifacts."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        ignore = shutil.ignore_patterns(
            "__pycache__", "build", "cmake-build-*",
            "*.o", "*.so", "*.a",
            "ref_benchmark", "sol_benchmark",
        )
        shutil.copytree(source_task_dir, self.work_dir,
                        dirs_exist_ok=True, ignore=ignore)

    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def snapshot_round(self, round_idx: int) -> dict:
        rd = self.run_out_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        for path, content in self.written_files.items():
            target = rd / path.replace("/", "__")
            target.write_text(content)
        return {"dir": str(rd)}


# ── Baseline ─────────────────────────────────────────────────────────────────

_KERNEL_TIME_RE = re.compile(r"Kernel time:\s*([0-9.]+)\s*ms")


def _truncate(text: str, limit: int = MAX_PROFILE_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return text[:half] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-half:]


def extract_compile_errors(stderr: str, max_chars: int = 3000) -> str:
    if not stderr:
        return ""
    error_kw = re.compile(r"\b(error:|fatal error:|undefined reference|note:|warning:)")
    lines = stderr.splitlines()
    kept_idx: set[int] = set()
    for i, line in enumerate(lines):
        if error_kw.search(line):
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                kept_idx.add(j)
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception): ", lines[i]):
            kept_idx.add(i)
            break
    if not kept_idx:
        return _truncate(stderr, max_chars)
    out_lines: list[str] = []
    prev = -2
    for idx in sorted(kept_idx):
        if prev != -2 and idx > prev + 1:
            out_lines.append("  ... (skipped non-error output) ...")
        out_lines.append(lines[idx])
        prev = idx
    return _truncate("\n".join(out_lines), max_chars)


def _run_reference_once(session: Session) -> tuple[bool, float, str]:
    """Run the upstream reference kernel once via `python run.py` on the
    untouched reference solution. We restore the reference solution files via
    a git-free approach: since this evaluator copies the task dir fresh in
    setup(), the initial files ARE the ref placeholders with `#error` — we
    cannot run the reference through run.py with placeholders.

    Instead, we drive `make ref` (builds ref_benchmark) and run it directly.
    This matches what run.py does internally, but skips the solution build.
    """
    src_dir = session.work_dir / "src"
    try:
        build = subprocess.run(
            ["make", "ref"], cwd=src_dir,
            capture_output=True, text=True, timeout=REF_RUN_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, -1.0, f"make ref timed out ({REF_RUN_TIMEOUT_SEC}s)"
    if build.returncode != 0:
        return False, -1.0, f"make ref failed:\n{(build.stderr or build.stdout)[-3000:]}"

    ref_bin = session.work_dir / "ref_benchmark"
    if not ref_bin.is_file():
        return False, -1.0, f"ref_benchmark not produced at {ref_bin}"
    try:
        proc = subprocess.run(
            [str(ref_bin)], cwd=session.work_dir,
            capture_output=True, text=True, timeout=REF_RUN_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, -1.0, f"ref_benchmark timed out ({REF_RUN_TIMEOUT_SEC}s)"
    combined = (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or "")
    m = _KERNEL_TIME_RE.search(combined)
    kernel_time = float(m.group(1)) if m else -1.0
    ok = proc.returncode == 0 and kernel_time > 0
    return ok, kernel_time, combined


def measure_baseline(session: Session, verbose: bool = True) -> float:
    if verbose:
        print(f"[baseline] warmup run ({BASELINE_WARMUP})...", flush=True)
    for _ in range(BASELINE_WARMUP):
        ok, kt, out = _run_reference_once(session)
        if not ok:
            raise RuntimeError(f"baseline warmup failed:\n{out[-3000:]}")
    times = []
    for i in range(BASELINE_RUNS):
        ok, kt, out = _run_reference_once(session)
        if not ok:
            raise RuntimeError(f"baseline run {i+1} failed:\n{out[-3000:]}")
        times.append(kt)
        if verbose:
            print(f"[baseline] run {i+1}/{BASELINE_RUNS}: kernel_time={kt:.2f} ms", flush=True)
    return mean(times)


# ── Action executors ──────────────────────────────────────────────────────────

def _is_safe_relpath(path: str) -> bool:
    p = path.replace("\\", "/")
    return bool(p and not p.startswith("/") and p not in (".", "..")
                and not p.startswith("../") and "/../" not in p)


def do_write_file(session: Session, action: dict) -> str:
    path = action["path"]
    if path not in SOLUTION_FILES:
        return f"ERROR: path '{path}' not in writable set {SOLUTION_FILES}."
    if not _is_safe_relpath(path):
        return f"ERROR: unsafe path '{path}'."
    code = action["code"]
    if not code.strip():
        return f"ERROR: empty code block for {path}."
    for pat, desc in FORBIDDEN_PATTERNS:
        if pat.search(code):
            return f"ERROR: '{path}' contains forbidden pattern — {desc}."
    target = session.work_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)
    session.written_files[path] = code
    return f"OK: wrote {path} ({len(code)} bytes)."


def do_read_file(session: Session, action: dict) -> str:
    path = action["path"]
    if path in session.written_files:
        content = session.written_files[path]
        return f"CONTENT of {path} (your current version, {len(content)} bytes):\n```\n{content}\n```"
    if path in READABLE_INFRA_FILES:
        fs_path = session.work_dir / path
        if not fs_path.is_file():
            return f"ERROR: infra file '{path}' missing on disk (unexpected)."
        content = fs_path.read_text()
        return f"CONTENT of {path} (infra, {len(content)} bytes):\n```\n{content}\n```"
    if path in SOLUTION_FILES:
        return (f"ERROR: '{path}' is a placeholder you have not yet rewritten. "
                f"Write your own implementation first.")
    return (f"ERROR: '{path}' is not in the readable whitelist. "
            f"Writable: {SOLUTION_FILES}. Readable: {sorted(READABLE_INFRA_FILES)}.")


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def do_profile(session: Session, action: dict) -> str:
    if not session.latest_build_ok:
        return ("ERROR: no round has passed correctness yet. Write a working "
                "solution first and let the harness build it.")
    tool = action.get("tool", "")
    env = os.environ.copy()
    env["TMPDIR"] = str(session.tmp_dir)
    sol_bin = session.work_dir / "sol_benchmark"
    if not sol_bin.is_file():
        return f"ERROR: sol_benchmark not present at {sol_bin}."
    ts = int(time.time() * 1000)
    try:
        if tool == "nsys":
            rep = session.tmp_dir / f"nsys_{ts}.nsys-rep"
            cmd = ["nsys", "profile", "-t", "cuda", "--stats=true",
                   "--force-overwrite", "true", "-o", str(rep), str(sol_bin)]
        elif tool == "ncu":
            kregex = action.get("kernel", "").strip()
            if not kregex:
                return "ERROR: ncu requires a 'kernel' regex."
            cmd = ["ncu", "--set", "basic", "--launch-count", "5",
                   "--kernel-name", f"regex:{kregex}", str(sol_bin)]
        else:
            return f"ERROR: unknown profile tool '{tool}'. Use 'nsys' or 'ncu'."
        proc = subprocess.run(
            cmd, cwd=session.work_dir, env=env,
            capture_output=True, text=True, timeout=PROFILE_TIMEOUT_SEC,
        )
        combined = (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or "")
        if tool == "ncu" and "No kernels were profiled" in combined:
            return (f"ncu ERROR: no launches matched regex '{kregex}'. "
                    f"Run nsys first to see actual kernel names.\n" + _truncate(combined, 1500))
        return f"{tool.upper()} PROFILE (exit={proc.returncode}):\n" + _truncate(combined)
    except subprocess.TimeoutExpired:
        return f"ERROR: {tool} timed out after {PROFILE_TIMEOUT_SEC}s."
    except FileNotFoundError as e:
        return f"ERROR: {tool} not installed ({e})."


# ── Per-round evaluation ─────────────────────────────────────────────────────

def run_eval(session: Session) -> dict:
    eval_script = session.work_dir / "eval_solution.py"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), str(session.work_dir),
             "--timeout", str(EVAL_TIMEOUT_SEC - 60)],
            cwd=session.work_dir,
            capture_output=True, text=True, timeout=EVAL_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {"compiled": False, "correct": False, "speedup": -1.0,
                "error": f"eval_solution.py timed out ({EVAL_TIMEOUT_SEC}s)",
                "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    result = {}
    try:
        result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", stdout, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
            except json.JSONDecodeError:
                result = {}
    if not result:
        return {"compiled": False, "correct": False, "speedup": -1.0,
                "exit_code": proc.returncode, "elapsed_s": elapsed,
                "error": "eval_solution.py returned no parseable JSON",
                "stdout_tail": _truncate(stdout, 1500),
                "stderr_errors": extract_compile_errors(stderr, 2000)}
    compiled = bool(result.get("compiled"))
    correct = bool(result.get("correct"))
    kernel_time = float(result.get("kernel_time_ms", -1.0)) if result.get("kernel_time_ms") not in (None, -1) else -1.0

    # Hidden anti-cheat: reject sub-floor kernel_time even if correct. A real
    # FP64 tensor-contraction kernel on 1.2M total grid blocks cannot complete
    # in < KERNEL_TIME_FLOOR_MS — anything faster is a hardcode/shortcut.
    anti_cheat_reason = ""
    if correct and kernel_time > 0 and kernel_time < KERNEL_TIME_FLOOR_MS:
        anti_cheat_reason = (
            f"anti-cheat: kernel_time={kernel_time:.3f} ms is below the minimum "
            f"plausible time ({KERNEL_TIME_FLOOR_MS:.0f} ms) for a real FP64 "
            f"tensor-contraction kernel on these test cases."
        )
        correct = False

    speedup = (session.baseline_time_ms / kernel_time
               if session.baseline_time_ms > 0 and kernel_time > 0 else -1.0)
    if compiled and correct and not session.latest_build_ok:
        session.latest_build_ok = True
    return {
        "compiled": compiled,
        "correct": correct,
        "kernel_time_ms": kernel_time,
        "baseline_time_ms": session.baseline_time_ms,
        "fresh_ref_time_ms": float(result.get("ref_time_ms", -1.0)) if result.get("ref_time_ms") not in (None, -1) else -1.0,
        "speedup": speedup,
        "fresh_speedup": float(result.get("speedup", -1.0)) if result.get("speedup") not in (None, -1) else -1.0,
        "energy_t": result.get("energy_t"),
        "energy_t5": result.get("energy_t5"),
        "anti_cheat_reason": anti_cheat_reason,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": _truncate(result.get("output", "") or stdout, 1500),
        "error": anti_cheat_reason or result.get("output", "")[-500:] if not correct else "",
        "stderr_errors": extract_compile_errors(stderr, 2000) if stderr.strip() else "",
    }


def execute_round(session: Session, round_idx: int, actions: list[dict]) -> tuple[list[str], dict | None]:
    outputs: list[str] = []
    profile_calls = 0
    seen_writes: set[str] = set()
    any_write = False
    for act in actions:
        if "error" in act and act.get("type") not in ("profile", "update_plan"):
            outputs.append(f"REJECTED: {act['error']}  raw={act['raw_json'][:200]}")
            continue
        atype = act.get("type")
        if atype == "write_file":
            if act["path"] in seen_writes:
                outputs.append(f"REJECTED: {act['path']} already written this round.")
                continue
            seen_writes.add(act["path"])
            any_write = True
            outputs.append(do_write_file(session, act))
        elif atype == "read_file":
            outputs.append(do_read_file(session, act))
        elif atype == "profile":
            if profile_calls >= MAX_PROFILE_PER_ROUND:
                outputs.append(f"REJECTED: profile limit ({MAX_PROFILE_PER_ROUND}) reached this round.")
                continue
            profile_calls += 1
            outputs.append(do_profile(session, act))
        elif atype == "update_plan":
            outputs.append(do_update_plan(session, act))
        else:
            outputs.append(f"REJECTED: unknown action '{atype}'.")

    eval_result = run_eval(session) if any_write else None
    return outputs, eval_result


# ── Feedback assembly ────────────────────────────────────────────────────────

def _coverage_block(session: Session, round_idx: int, total_rounds: int) -> str:
    lines = ["=== FILES YOU HAVE WRITTEN ==="]
    if session.written_files:
        for p in sorted(session.written_files.keys()):
            sz = len(session.written_files[p])
            lines.append(f"  {p} ({sz} B)")
    else:
        lines.append("  (none yet)")
    lines.append(f"Round {round_idx}/{total_rounds} ({total_rounds - round_idx} left).")
    return "\n".join(lines)


def build_planning_message(task_description: str, baseline_time_ms: float) -> str:
    return (f"{task_description}\n\n"
            "=== PLANNING PHASE ===\n"
            f"Baseline (reference CCSD(T) kernel): {baseline_time_ms:.2f} ms total "
            f"(mean of {BASELINE_RUNS} runs after {BASELINE_WARMUP} warmup).\n\n"
            "Produce a concise plan covering:\n"
            "  1. Your high-level algorithmic approach to fusing the S1/D1/D2 equations.\n"
            "  2. Tile sizes / thread-block shape / shared-memory budget you'll try.\n"
            "  3. Which instructions / intrinsics you plan to use for the FP64 math and why.\n"
            "  4. How you will partition work across the 3 test cases.\n"
            "  5. Round budget — what you'll have by round 5 / 10 / final.\n\n"
            "Output ONLY plan text — no JSON actions, no code blocks. Revise later with `update_plan`.")


def build_user_message_round1(session: Session, total_rounds: int) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    cov = _coverage_block(session, 1, total_rounds) + "\n\n"
    bl = f"Baseline: {session.baseline_time_ms:.2f} ms.\n"
    return (f"{plan_block}{cov}{bl}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            "Begin writing your solution. A minimum first-round build needs:\n"
            "  solution.cu              (main kernel + launch_ccsd_t_kernel driver)\n"
            "  ccsd_t_g2s_device_functions.cu  (can start as a no-op header if unused)\n"
            "  tensor_core_helper.cuh   (can start as a small header if unused)\n"
            "Reading `src/kernel_interface.cuh` and `src/main.cu` first is recommended\n"
            "to understand the driver signature and how inputs are prepared.")


def build_feedback(session: Session, round_idx: int, total_rounds: int,
                   action_outputs: list[str], eval_result: dict | None) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    cov = _coverage_block(session, round_idx, total_rounds) + "\n\n"
    parts = [f"{plan_block}{cov}=== ROUND {round_idx} / {total_rounds} RESULTS ==="]
    for i, out in enumerate(action_outputs, 1):
        parts.append(f"\n--- action {i} output ---\n{out}")
    if eval_result is not None:
        parts.append("\n--- evaluation (eval_solution.py) ---")
        sp = eval_result.get("speedup", -1.0)
        kt = eval_result.get("kernel_time_ms", -1.0)
        bl = eval_result.get("baseline_time_ms", session.baseline_time_ms)
        summary = (f"compiled={eval_result.get('compiled')}  "
                   f"correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  "
                   f"speedup={sp:.4f}x" if sp > 0 else
                   f"compiled={eval_result.get('compiled')}  correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  speedup=N/A")
        parts.append(summary)
        et, et5 = eval_result.get("energy_t"), eval_result.get("energy_t5")
        if et is not None or et5 is not None:
            parts.append(f"Energy_T={et} Energy_T5={et5}")
        if eval_result.get("anti_cheat_reason"):
            parts.append(f"RUNTIME CHECK: {eval_result['anti_cheat_reason']}")
        if eval_result.get("stderr_errors"):
            parts.append(f"\ncompile errors (extracted):\n{eval_result['stderr_errors']}")
        tail = eval_result.get('stdout_tail', '')
        if tail.strip():
            parts.append(f"\nstdout tail:\n{tail[:1500]}")
    parts.append(f"\n=== ROUND {round_idx+1} / {total_rounds} ===" if round_idx < total_rounds
                 else "\n(final round complete)")
    return "\n".join(parts)


# ── Main driver ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--backend", choices=["openai", "vertex"], default="openai")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"])
    ap.add_argument("--num-rounds", type=int, default=20)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    root = Path(get_project_root())
    src_task = root / TASK_DIR_REL
    out_dir = Path(args.output) if args.output else root / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intentionally use our STRIPPED description, NOT description.txt from disk.
    task_description = STRIPPED_DESCRIPTION

    session = Session(run_out_dir=out_dir)
    session.setup(src_task)

    t_bl = time.time()
    print("[setup] measuring baseline (reference CCSD(T) kernel)...", flush=True)
    session.baseline_time_ms = measure_baseline(session, verbose=True)
    print(f"[setup] baseline = {session.baseline_time_ms:.2f} ms "
          f"({time.time() - t_bl:.1f}s total)", flush=True)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    conv_path = out_dir / "conversation.jsonl"
    conv_log = conv_path.open("w")

    def log(role: str, content: str):
        conv_log.write(json.dumps({"role": role, "content": content}) + "\n")
        conv_log.flush()

    log("system", SYSTEM_PROMPT)

    def call_model(label: str) -> str:
        print(f"\n[{label}] querying model...", flush=True)
        t_q = time.time()
        resp = query_server(
            prompt=messages, model=args.model, system_prompt="",
            temperature=args.temperature, max_tokens=args.max_tokens,
            api_base=args.api_base, api_key=args.api_key, backend=args.backend,
            is_reasoning_model=args.reasoning, reasoning_effort=args.reasoning_effort,
        )
        if isinstance(resp, list):
            resp = resp[0]
        print(f"  model responded in {time.time() - t_q:.1f}s, {len(resp)} chars", flush=True)
        return resp

    t_start = time.time()
    try:
        planning_msg = build_planning_message(task_description, session.baseline_time_ms)
        messages.append({"role": "user", "content": planning_msg})
        log("user", planning_msg)
        plan_response = call_model("Plan phase")
        messages.append({"role": "assistant", "content": plan_response})
        log("assistant", plan_response)
        session.current_plan = plan_response.strip()
        print(f"  plan stored ({len(session.current_plan)} chars)", flush=True)

        first_impl = build_user_message_round1(session, args.num_rounds)
        messages.append({"role": "user", "content": first_impl})
        log("user", first_impl)

        for round_idx in range(1, args.num_rounds + 1):
            response = call_model(f"Round {round_idx}/{args.num_rounds}")
            messages.append({"role": "assistant", "content": response})
            log("assistant", response)

            actions = parse_actions(response, fallback_round1=(round_idx == 1))
            print(f"  parsed {len(actions)} actions: " +
                  ", ".join(a.get("type", "?") for a in actions), flush=True)

            t_r = time.time()
            outputs, eval_result = execute_round(session, round_idx, actions)
            print(f"  executed in {time.time() - t_r:.1f}s", flush=True)

            if eval_result and eval_result.get("correct") and eval_result.get("speedup", -1) > 0:
                if eval_result["speedup"] > session.best_speedup:
                    session.best_speedup = eval_result["speedup"]
                    session.best_round = round_idx

            session.snapshot_round(round_idx)
            session.rounds.append({
                "round": round_idx,
                "actions": [{"type": a.get("type"), "path": a.get("path", ""),
                             "tool": a.get("tool", ""), "kernel": a.get("kernel", ""),
                             "error": a.get("error", "")} for a in actions],
                "outputs_summary": [o.splitlines()[0][:160] for o in outputs],
                "eval": eval_result,
            })

            if eval_result:
                sp = eval_result.get("speedup", -1)
                corr = eval_result.get("correct")
                kt = eval_result.get("kernel_time_ms", -1)
                print(f"  eval: correct={corr} kt={kt:.2f}ms speedup={sp:.4f}x" if sp > 0 else
                      f"  eval: correct={corr} kt={kt:.2f}ms speedup=N/A", flush=True)

            feedback = build_feedback(session, round_idx, args.num_rounds, outputs, eval_result)
            messages.append({"role": "user", "content": feedback})
            log("user", feedback)
    finally:
        conv_log.close()

    elapsed = time.time() - t_start
    final = {
        "task_id": TASK_ID, "model": args.model,
        "num_rounds": args.num_rounds,
        "baseline_time_ms": session.baseline_time_ms,
        "best_speedup": session.best_speedup, "best_round": session.best_round,
        "elapsed_s": elapsed, "rounds": session.rounds,
    }
    (out_dir / "final_report.json").write_text(json.dumps(final, indent=2))

    print(f"\n=== DONE ({elapsed:.1f}s) ===")
    print(f"baseline = {session.baseline_time_ms:.2f} ms")
    print(f"best_speedup = {session.best_speedup:.4f}x (round {session.best_round})")
    print(f"results in: {out_dir}")

    session.cleanup()


if __name__ == "__main__":
    main()
