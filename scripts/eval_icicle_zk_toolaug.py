#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/icicle_zk.

Customized loop:
  - Baseline measured once at setup: `python run.py` × 3 runs, mean of the
    "--- CUDA Baseline (Icicle) ---" E2E time.
  - Planning phase (round 0): model produces a plan; no actions parsed.
  - Rounds 1..N: model may write/read/profile/update_plan. If any write_file
    occurred in a round, the harness stages the candidate directory and runs
    `python eval_solution.py <candidate_dir>` (which compiles + benches).
  - write_file: any path under `custom_cuda_backend/` is allowed (multi-file).
  - read_file: whitelist = {custom_backend_api.h, backend_template.cu,
    build_contract.md} or any file the LLM has written in this session.

Usage:
    python scripts/eval_icicle_zk_toolaug.py \
        --model "Qwen/Qwen3.5-122B-A10B" \
        --api-base http://134.84.150.135:8001/v1 --api-key EMPTY \
        --run-name icicle_tool_qwen
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
TASK_ID = "class3/icicle_zk/blackwell"
TASK_DIR_REL = "tasks/class3/icicle_zk"
SOLUTION_PREFIX = "custom_cuda_backend/"   # every write path must start with this

# Files the LLM may read even without having written them (task-owned helpers).
# Anything else (including run.py, eval_solution.py, cuda_backend/*.so, ref_data/*,
# reference_sources/icicle) is NOT readable via read_file.
READABLE_INFRA_FILES = {
    "custom_backend_api.h",
    "backend_template.cu",
    "build_contract.md",
}

# nvcc / dlopen-time anti-cheat patterns (scanned on write_file).
# The LLM can still #include Icicle *public* headers at compile time; these
# blocked patterns only forbid importing internal CUDA backend implementations
# or re-loading Icicle's prebuilt .so via dlopen.
FORBIDDEN_PATTERNS = [
    (re.compile(r"#include\s*[<\"]icicle/backend/", re.IGNORECASE),
     "#include <icicle/backend/...> — reference backend internals"),
    (re.compile(r"libicicle_backend_cuda", re.IGNORECASE),
     "libicicle_backend_cuda — reference backend binary"),
    (re.compile(r"dlopen\s*\(", re.IGNORECASE),
     "dlopen — loading another backend at runtime"),
    (re.compile(r"dlsym\s*\(", re.IGNORECASE),
     "dlsym — loading another backend at runtime"),
    (re.compile(r"ref_data/", re.IGNORECASE),
     "ref_data/ — reading the reference outputs"),
]

BASELINE_RUNS = 3
MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
REF_RUN_TIMEOUT_SEC = 900       # first run.py invocation may need to prepare data
EVAL_TIMEOUT_SEC = 1200         # matches eval_solution.py default
PROFILE_TIMEOUT_SEC = 1200

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA engineer implementing a custom Icicle-compatible
backend for zero-knowledge benchmarks, inside a fixed tool-augmented harness.

## Task

Implement high-performance CUDA kernels for **NTT** (Number Theoretic Transform)
and **MSM** (Multi-Scalar Multiplication) on the **BN254** elliptic curve (254-bit
prime scalar field, G1 over E/Fp). Your code is compiled into a shared library
and loaded via dlopen by the benchmark.

Operations:
  - NTT forward on BN254 scalar field. Sizes: 2^16, 2^18, 2^20, 2^22, 2^24.
  - MSM on BN254 G1. Sizes: 2^14, 2^16, 2^18, 2^20, 2^22.

Correctness is checked against a CPU reference bit-for-bit. Performance is
compared against Icicle's reference CUDA backend.

## Fixed ABI

Implement exactly these `extern "C"` symbols (declared in `custom_backend_api.h`):

    int  kh_custom_backend_init();
    void kh_custom_backend_shutdown();
    const char* kh_custom_backend_last_error();

    int  kh_custom_ntt_forward_bn254(
        const bn254::scalar_t* input, int log_n, bn254::scalar_t* output);

    int  kh_custom_msm_bn254(
        const bn254::scalar_t* scalars, const bn254::affine_t* points,
        int log_n, bn254::projective_t* output);

Return 0 on success, non-zero on failure (and expose a message via
`kh_custom_backend_last_error`).

Inputs/outputs are HOST pointers. You must copy to device, compute on GPU, copy
back to host. The harness allocates/frees all host buffers.

You may `#include` public Icicle headers (e.g. `icicle/curves/params/bn254.h`,
`icicle/msm.h`) for types and helper math, but you must NOT:
  - `#include <icicle/backend/...>` — reference backend internals
  - link against or dlopen any `libicicle_backend_cuda*` — that is the reference
  - read `ref_data/*` files — those are the reference outputs used for grading

## Phases

**Phase 0 (planning)** — The first message asks for a free-text plan. Output plan
text only; no JSON actions, no code blocks.

**Phase 1..N (implementation)** — Each round you respond with actions.

## Action format (strict)

Every action MUST start with a JSON object on its own line.

1. `write_file` — write a file under `custom_cuda_backend/`. Multi-file projects
   are allowed (you can split kernels into multiple `.cu` / `.cuh` files; the
   harness compiles `custom_cuda_backend/backend.cu` as the primary translation
   unit, and other files can be `#include`'d from it).
   ```
   {"action":"write_file","path":"custom_cuda_backend/backend.cu"}
   ```cuda
   // full file contents (not a diff)
   ```
   Same file at most once per round.

2. `read_file` — read the CURRENT contents of a file. Allowed:
   - any file you have already written this session
   - `custom_backend_api.h` (the fixed ABI header)
   - `backend_template.cu` (a stub showing the 5 required functions)
   - `build_contract.md` (the full build/evaluation contract)

3. `profile` — run a profiler over the compiled bench binary after a successful
   build (requires at least one prior successful write_file round with
   `correct=True`).
   - `nsys`: timeline + kernel summary
     ```
     {"action":"profile","tool":"nsys"}
     ```
   - `ncu`: per-kernel micro-metrics; supply a regex matching kernel name.
     ```
     {"action":"profile","tool":"ncu","kernel":"ntt.*"}
     ```
   At most 2 profile calls per round.

4. `update_plan` — revise your stored plan (replaces old plan, re-shown each round).
   ```
   {"action":"update_plan"}
   ```text
   revised plan text
   ```

## Round rules

- Round 1: any action allowed. A minimum first build needs
  `custom_cuda_backend/backend.cu` implementing the 5 ABI symbols.
- Rounds 2+: any action combination; executed in the written order.
- If a round contains any `write_file`, the harness runs the full eval:
  `python eval_solution.py <candidate_dir>` which compiles + runs the bench
  (CPU ref + CUDA ref + your custom backend), parses JSON, and reports
  `compiled / correct / kernel_time / speedup`.

## Evaluation

Speedup = baseline_time_ms / your_kernel_time_ms, where baseline is the Icicle
reference CUDA backend, measured once at setup (3 runs of the reference, mean).
"""


# ── Action parser (same format as cuszp/llmc) ─────────────────────────────────

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
    """Fallback: grab text between start_idx and the next JSON action block.
    Returns stripped text if it looks like source code (contains common markers)."""
    next_json = _JSON_ACTION_RE.search(text, start_idx)
    end = next_json.start() if next_json else len(text)
    chunk = text[start_idx:end].strip()
    # Heuristic: accept if the chunk is substantial and looks like C/C++/CUDA.
    if len(chunk) < 50:
        return ""
    code_markers = ("#include", "__global__", "__device__", "extern \"C\"",
                    "int main", "namespace ", "template ", "void ", "cudaError")
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
                # Permissive fallback: model emitted raw code without a code fence.
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
                "path": "custom_cuda_backend/backend.cu",
                "code": code,
                "raw_json": "(synthesized: no JSON action header, assuming write_file backend.cu)",
            })
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="icicle_zk_toolaug_"))
        # Mirror the real on-disk layout so run.py's relative path
        # `task_dir/../../../reference_sources/icicle` lands in our tmp_dir.
        self.work_dir = self.tmp_dir / "tasks" / "class3" / "icicle_zk"
        self.candidate_dir = self.tmp_dir / "candidate"  # LLM files staged here
        self.written_files: dict[str, str] = {}
        self.rounds: list[dict] = []
        self.current_plan: str = ""
        self.baseline_time_ms: float = -1.0
        self.best_speedup: float = -1.0
        self.best_round: int = -1
        self.latest_build_ok: bool = False

    def setup(self, source_task_dir: Path):
        """Copy task dir (excluding heavy/pre-built artifacts) to work_dir. Share
        reference_sources/icicle, cuda_backend, ref_data via symlink to avoid
        duplicating multi-GB data."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Files that can be copied cheaply (task scaffolding).
        heavy_or_readonly = {"ref_data", "cuda_backend", "__pycache__"}
        for child in source_task_dir.iterdir():
            name = child.name
            if name in heavy_or_readonly:
                continue
            dst = self.work_dir / name
            if child.is_dir():
                shutil.copytree(child, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "build"))
            else:
                shutil.copy2(child, dst)
        # Symlink the heavy/read-only pieces.
        for name in ("ref_data", "cuda_backend"):
            src = source_task_dir / name
            if src.exists():
                os.symlink(src.resolve(), self.work_dir / name)
        # Mirror reference_sources/icicle at tmp_dir/reference_sources/icicle so
        # run.py's hardcoded `task_dir/../../../reference_sources/icicle` resolves.
        project_root = Path(get_project_root())
        ref_src = project_root / "reference_sources" / "icicle"
        (self.tmp_dir / "reference_sources").mkdir(parents=True, exist_ok=True)
        os.symlink(ref_src.resolve(), self.tmp_dir / "reference_sources" / "icicle")
        # Fresh candidate scratch
        self.candidate_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def snapshot_round(self, round_idx: int) -> dict:
        rd = self.run_out_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        for path, content in self.written_files.items():
            target = rd / path.replace("/", "__")
            target.write_text(content)
        return {"dir": str(rd)}


# ── Baseline measurement ──────────────────────────────────────────────────────

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


_BASELINE_RE = re.compile(
    r"---\s*CUDA Baseline \(Icicle\)\s*---.*?E2E:\s*([0-9.]+)\s*ms",
    re.DOTALL | re.IGNORECASE,
)


def run_baseline_once(session: Session) -> tuple[bool, float, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=session.work_dir,
            capture_output=True, text=True,
            timeout=REF_RUN_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, -1.0, f"run.py timed out ({REF_RUN_TIMEOUT_SEC}s)"
    combined = (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or "")
    m = _BASELINE_RE.search(combined)
    if not m:
        return False, -1.0, combined
    try:
        return True, float(m.group(1)), combined
    except ValueError:
        return False, -1.0, combined


def measure_baseline(session: Session, verbose: bool = True) -> float:
    times = []
    for i in range(BASELINE_RUNS):
        ok, t, out = run_baseline_once(session)
        if not ok:
            raise RuntimeError(f"baseline run {i+1} failed:\n{out[-3000:]}")
        times.append(t)
        if verbose:
            print(f"[baseline] run {i+1}/{BASELINE_RUNS}: Icicle CUDA E2E={t:.2f} ms", flush=True)
    return mean(times)


# ── Action executors ──────────────────────────────────────────────────────────

def _is_safe_relpath(path: str) -> bool:
    p = path.replace("\\", "/")
    return bool(p and not p.startswith("/") and p not in (".", "..")
                and not p.startswith("../") and "/../" not in p)


def do_write_file(session: Session, action: dict) -> str:
    path = action["path"]
    if not path.startswith(SOLUTION_PREFIX):
        return (f"ERROR: path '{path}' must start with '{SOLUTION_PREFIX}'.")
    if not _is_safe_relpath(path):
        return f"ERROR: unsafe path '{path}'."
    code = action["code"]
    if not code.strip():
        return f"ERROR: empty code block for {path}."
    for pat, desc in FORBIDDEN_PATTERNS:
        if pat.search(code):
            return f"ERROR: '{path}' contains forbidden pattern — {desc}."
    # Stage under session.candidate_dir. We drop the SOLUTION_PREFIX so files land
    # directly in candidate_dir (eval_solution.py treats candidate_dir as the
    # contents of custom_cuda_backend/).
    rel_in_candidate = path[len(SOLUTION_PREFIX):]
    target = session.candidate_dir / rel_in_candidate
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)
    session.written_files[path] = code
    return f"OK: wrote {path} ({len(code)} bytes)."


def do_read_file(session: Session, action: dict) -> str:
    path = action["path"]
    # Priority 1: something the LLM has written this session
    if path in session.written_files:
        content = session.written_files[path]
        return f"CONTENT of {path} (your current version, {len(content)} bytes):\n```\n{content}\n```"
    # Priority 2: task-owned infra file
    if path in READABLE_INFRA_FILES:
        fs_path = session.work_dir / path
        if not fs_path.is_file():
            return f"ERROR: infra file '{path}' missing on disk (unexpected)."
        content = fs_path.read_text()
        return f"CONTENT of {path} (infra, {len(content)} bytes):\n```\n{content}\n```"
    return (f"ERROR: '{path}' is not readable. "
            f"You may only read files you have written OR: {sorted(READABLE_INFRA_FILES)}.")


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def _bench_binary(session: Session) -> Path:
    return session.work_dir / "src" / "build" / "bench"


def _custom_so(session: Session) -> Path:
    return session.candidate_dir / "libkh_custom_backend.so"


def do_profile(session: Session, action: dict) -> str:
    if not session.latest_build_ok:
        return "ERROR: no successful build yet. Write code and let the round's eval build it first."
    bench = _bench_binary(session)
    so = _custom_so(session)
    if not bench.is_file():
        return f"ERROR: bench binary not present at {bench}."
    if not so.is_file():
        return f"ERROR: custom backend .so not present at {so}."
    tool = action.get("tool", "")
    env = os.environ.copy()
    env["TMPDIR"] = str(session.tmp_dir)
    env["LD_LIBRARY_PATH"] = (
        str(session.work_dir.parent.parent / "reference_sources" / "icicle" / "build")
    )
    cmd_prefix = [str(bench), "--device", "CUSTOM", "--custom-so", str(so),
                  "--ref-dir", str(session.work_dir / "ref_data")]
    ts = int(time.time() * 1000)
    try:
        if tool == "nsys":
            rep = session.tmp_dir / f"nsys_{ts}.nsys-rep"
            cmd = ["nsys", "profile", "-t", "cuda", "--stats=true",
                   "--force-overwrite", "true", "-o", str(rep)] + cmd_prefix
        elif tool == "ncu":
            kregex = action.get("kernel", "").strip()
            if not kregex:
                return "ERROR: ncu requires a 'kernel' regex."
            cmd = ["ncu", "--set", "basic", "--launch-count", "10",
                   "--kernel-name", f"regex:{kregex}",
                   "--target-processes", "all"] + cmd_prefix
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


# ── Per-round evaluation ──────────────────────────────────────────────────────

def run_eval(session: Session) -> dict:
    """Call the task's eval_solution.py on the current candidate directory."""
    eval_script = session.work_dir / "eval_solution.py"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), str(session.candidate_dir),
             "--timeout", str(EVAL_TIMEOUT_SEC // 2)],
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
    # Our consistent speedup is vs the cached up-front baseline (3-run mean).
    speedup = (session.baseline_time_ms / kernel_time
               if session.baseline_time_ms > 0 and kernel_time > 0 else -1.0)
    fresh_gpu_ref = float(result.get("gpu_ref_time_ms", -1.0)) if result.get("gpu_ref_time_ms") not in (None, -1) else -1.0
    if compiled and correct and not session.latest_build_ok:
        session.latest_build_ok = True
    return {
        "compiled": compiled,
        "correct": correct,
        "kernel_time_ms": kernel_time,
        "baseline_time_ms": session.baseline_time_ms,
        "fresh_gpu_ref_time_ms": fresh_gpu_ref,
        "cpu_time_ms": float(result.get("cpu_time_ms", -1.0)) if result.get("cpu_time_ms") not in (None, -1) else -1.0,
        "speedup": speedup,
        "fresh_speedup": float(result.get("speedup", -1.0)) if result.get("speedup") not in (None, -1) else -1.0,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": _truncate(result.get("output", "") or stdout, 1500),
        "error": result.get("error", ""),
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
            f"Baseline (Icicle reference CUDA backend): {baseline_time_ms:.2f} ms "
            f"(mean of {BASELINE_RUNS} runs).\n\n"
            "Produce a concise plan covering:\n"
            "  1. NTT strategy — algorithm (e.g. Stockham / six-step Cooley-Tukey),\n"
            "     twiddle handling for BN254 scalar field, memory layout.\n"
            "  2. MSM strategy — approach (e.g. Pippenger bucket method, window size,\n"
            "     point addition/doubling in Jacobian/projective coords).\n"
            "  3. BN254 arithmetic — 256-bit modular add/mul/inv, Montgomery form, etc.\n"
            "  4. File organization — single backend.cu vs multi-file split.\n"
            "  5. Round budget — which pieces first; what you'll have by round 5 vs 20.\n\n"
            "Output ONLY plan text — no JSON actions, no code blocks. The controller\n"
            "stores the plan and re-shows it every round. Revise later with `update_plan`.")


def build_user_message_round1(session: Session, task_description: str, total_rounds: int) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    cov = _coverage_block(session, 1, total_rounds) + "\n\n"
    bl = f"Baseline (Icicle reference CUDA): {session.baseline_time_ms:.2f} ms.\n"
    return (f"{plan_block}{cov}{bl}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            "Begin writing your backend. A first-round submission needs at minimum\n"
            "`custom_cuda_backend/backend.cu` implementing the 5 ABI symbols. You may\n"
            "also split supporting code into additional files under\n"
            "`custom_cuda_backend/`. Reading `custom_backend_api.h` and\n"
            "`backend_template.cu` first is recommended.")


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
        fresh_bl = eval_result.get("fresh_gpu_ref_time_ms", -1.0)
        cpu = eval_result.get("cpu_time_ms", -1.0)
        summary = (f"compiled={eval_result.get('compiled')}  "
                   f"correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  "
                   f"speedup={sp:.4f}x" if sp > 0 else
                   f"compiled={eval_result.get('compiled')}  correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  speedup=N/A")
        parts.append(summary)
        if fresh_bl > 0 or cpu > 0:
            parts.append(f"(this run: GPU ref={fresh_bl:.2f}ms, CPU ref={cpu:.2f}ms)")
        if eval_result.get("error"):
            parts.append(f"error: {eval_result['error'][:1000]}")
        if eval_result.get("stderr_errors"):
            parts.append(f"\ncompile errors (extracted):\n{eval_result['stderr_errors']}")
        tail = eval_result.get('stdout_tail', '')
        if tail.strip():
            parts.append(f"\nstdout tail:\n{tail[:1500]}")
    parts.append(f"\n=== ROUND {round_idx+1} / {total_rounds} ===" if round_idx < total_rounds
                 else "\n(final round complete)")
    return "\n".join(parts)


# ── Main driver ───────────────────────────────────────────────────────────────

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

    desc_path = src_task / "blackwell" / "description.txt"
    task_description = desc_path.read_text() if desc_path.exists() else ""

    session = Session(run_out_dir=out_dir)
    session.setup(src_task)

    t_bl = time.time()
    print("[setup] measuring baseline (Icicle CUDA reference)...", flush=True)
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

        first_impl = build_user_message_round1(session, task_description, args.num_rounds)
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
