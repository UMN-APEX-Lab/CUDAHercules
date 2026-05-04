#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/gpumd (black-box MD project generation).

Customized loop:
  - Baseline = upstream GPUMD-inspired `md_bench` reference, measured once at setup
    (3 timed runs of `python run.py`, mean of `Kernel time:` values).
  - Planning phase (round 0): model produces a plan; no actions parsed.
  - Rounds 1..N: model may write/read/profile/update_plan. If any write_file
    occurred in a round, the harness stages the project and runs
    `python eval_solution.py <candidate_dir>` (which builds + runs all 6 MD
    workloads and byte-compares forces vs the benchmark-owned reference).
  - write_file: any path under `generated_project/` is allowed (multi-file).
  - read_file: whitelist = {build_contract.md, project_manifest_example.json}
    or any file the LLM has written this session.

Correctness is enforced by `eval_solution.py` via byte-level force comparison
and energy-drift check, so this evaluator does not add any extra anti-cheat
gates on loss trajectories, param counts, etc.

Usage:
    python scripts/eval_gpumd_toolaug.py \
        --model "Qwen/Qwen3.5-122B-A10B" \
        --api-base http://134.84.150.135:8001/v1 --api-key EMPTY \
        --run-name gpumd_tool_qwen
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
TASK_ID = "class3/gpumd/blackwell"
TASK_DIR_REL = "tasks/class3/gpumd"
SOLUTION_ROOT = "generated_project/"

# Files the LLM may read even without having written them (task-owned helpers).
READABLE_INFRA_FILES = {
    "build_contract.md",
    "project_manifest_example.json",
}

# Anti-cheat: forbid referencing the upstream reference implementation or data.
# Mirrors task.yaml blocked_patterns + eval_solution.py FORBIDDEN_PATTERNS.
FORBIDDEN_PATTERNS = [
    (re.compile(r"tasks/class3/gpumd/src", re.IGNORECASE),
     "tasks/class3/gpumd/src — upstream reference path"),
    (re.compile(r"tasks/class3/gpumd/data", re.IGNORECASE),
     "tasks/class3/gpumd/data — reference data path"),
    (re.compile(r"\bmd_bench\b", re.IGNORECASE),
     "md_bench — reference binary"),
    (re.compile(r"\bref_forces\b", re.IGNORECASE),
     "ref_forces — reference force outputs"),
    (re.compile(r"\.blackbox_inputs_v1\b", re.IGNORECASE),
     ".blackbox_inputs_v1 — benchmark-owned input cache"),
    (re.compile(r"prepare_blackbox_inputs\.py", re.IGNORECASE),
     "prepare_blackbox_inputs.py — benchmark-owned input builder"),
    (re.compile(r"model\.xyz", re.IGNORECASE),
     "model.xyz — upstream XYZ data"),
]

BASELINE_WARMUP = 1
BASELINE_RUNS = 3
MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
REF_RUN_TIMEOUT_SEC = 900
EVAL_TIMEOUT_SEC = 900
PROFILE_TIMEOUT_SEC = 900

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA engineer writing a complete molecular-dynamics project
from scratch, inside a fixed tool-augmented evaluation harness.

## Task

Produce a self-contained CUDA/C++ MD project under `generated_project/` that runs
three force pipelines on a GPU:

  1. **Lennard-Jones** pairwise forces — Argon (256k, 500k atoms)
  2. **Tersoff** three-body potential — Silicon (27k, 64k atoms)
  3. **Coulomb real-space + reciprocal-space (Ewald-style)** — NaCl (8k, 32k atoms)

Your project must handle neighbor-list construction, integration, kinetic
energy, force computation, and output forces in a fixed binary format.

The upstream reference (GPUMD-inspired) is HIDDEN from you; you must implement
every kernel yourself.

## Required project layout (black-box)

- `generated_project/project_manifest.json`
  - keys: `build_script`, `run_script`, `files`, `cuda_sources`, `ops`
  - `ops` must map every required op → the file that implements it:
      neighbor_list, integration, kinetic_energy, lennard_jones, tersoff,
      coulomb_real, coulomb_kspace
- `generated_project/build.sh` — invoked via `bash build.sh`
- `generated_project/run.sh` — invoked via
  `bash run.sh --input-dir <D> --output-dir <O> [--ref-time-ms <F>]`
- one or more CUDA/C++ source files listed in `cuda_sources`, with real
  `__global__` kernels and `<<<>>>` launches

## Benchmark-Owned inputs

The benchmark converts upstream `.xyz` files into a binary format for you. The
input directory contains `workloads.json` (schema below) and one
`workloads/<name>.bin` per workload. Your project MUST consume this binary
format, NOT the upstream `.xyz` files.

`workloads.json`:
  {"version": 1, "format": "kh_gpumd_v1",
   "workloads": [
     {"name": "Ar_256000", "system": "lj", "file": "workloads/Ar_256000.bin",
      "atom_count": 256000, "steps": 100,
      "energy_drift_tol": 0.01, "force_rel_tol": 0.01},
     ... (6 workloads total)
   ]}

Per-workload binary layout (little-endian):
  Header:
    magic[8] = "KHGPMD1\\0"
    version  = uint32
    system_kind = uint32  (1=lj, 2=tersoff, 3=coulomb)
    atom_count = uint32
    steps      = uint32
    dt         = float64
    mass       = float64
    energy_drift_tol = float64
    force_rel_tol    = float64
    box_x, box_y, box_z = float64
    params[16] = float64[16]
  Data:
    x[N], y[N], z[N], vx[N], vy[N], vz[N] = float64
    atom_type[N] = int32
    charge[N]    = float32

`params` usage:
  - LJ: [epsilon, sigma, cutoff, ...]
  - Tersoff: [A, B, lambda, mu, beta, n, c, d, h, R1, R2, m, alpha, gamma]
  - Coulomb: [alpha, cutoff, ...]

## Required candidate outputs

Your `run.sh` must write:

  <OUTPUT_DIR>/benchmark_results.json:
    {"version": 1, "results": [
      {"name": <workload_name>, "force_file": <relative_path>,
       "energy_drift": <float>, "time_ms": <float>}, ... ]}

  <OUTPUT_DIR>/<force_file> per workload, little-endian:
    atom_count = int32
    fx[N], fy[N], fz[N], pe[N] = float64

And print:

  KERNEL <name>: energy_drift=<f> time_ms=<f> force_file=<rel_path>   (one per workload)
  Passed
  Kernel time: <float> ms

## Correctness

Enforced by the benchmark (cannot be faked):
  - byte-level comparison of each workload's force output against the
    benchmark-owned reference, with relative tolerance `force_rel_tol`
  - `energy_drift < energy_drift_tol` for every workload

## Anti-Cheat

Your project must be self-contained. The following names/paths will be
scanned for and rejected if present in any source file:
  - `tasks/class3/gpumd/src`, `tasks/class3/gpumd/data`
  - `md_bench`, `ref_forces`, `.blackbox_inputs_v1`, `prepare_blackbox_inputs.py`
  - `model.xyz`

## Phases

**Phase 0 (planning)** — The first message asks for a free-text plan. Output
plan text only; no JSON actions, no code blocks.

**Phase 1..N (implementation)** — Actions only.

## Action format (strict)

Every action MUST start with a JSON object on its own line.

1. `write_file` — write a file under `generated_project/`. Multi-file projects
   welcome (split kernels per op, helpers, build system).
   ```
   {"action":"write_file","path":"generated_project/project_manifest.json"}
   ```json
   {"build_script":"build.sh","run_script":"run.sh","files":[...],
    "cuda_sources":[...],"ops":{...}}
   ```

2. `read_file` — read your own written files OR: `build_contract.md`,
   `project_manifest_example.json`.

3. `profile` — run a profiler after a successful build.
   - `nsys`: timeline + kernel summary
   - `ncu`: per-kernel metrics; requires `"kernel": "<regex>"`
   At most 2 profile calls per round.

4. `update_plan` — replace your stored plan (re-shown each round).
   ```
   {"action":"update_plan"}
   ```text
   revised plan text
   ```

## Round rules

- Round 1: any actions. A minimum first build needs `project_manifest.json`
  + `build.sh` + `run.sh` + enough CUDA sources to cover all 7 required ops.
- Rounds 2+: any combination; executed in order.
- If the round contains any `write_file`, the harness runs the full eval.

## Evaluation

Speedup = baseline_time_ms / your_kernel_time_ms, where baseline is the
reference MD bench, measured 3 runs at setup (mean).
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
                    "#!/", "{", "[")
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
                "path": "generated_project/project_manifest.json",
                "code": code,
                "raw_json": "(synthesized: no JSON action header, assuming write_file project_manifest.json)",
            })
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="gpumd_toolaug_"))
        self.work_dir = self.tmp_dir / "gpumd"
        self.project_dir = self.work_dir / "generated_project"
        self.written_files: dict[str, str] = {}
        self.rounds: list[dict] = []
        self.current_plan: str = ""
        self.baseline_time_ms: float = -1.0
        self.best_speedup: float = -1.0
        self.best_round: int = -1
        self.latest_build_ok: bool = False

    def setup(self, source_task_dir: Path):
        """Copy task dir (excluding heavy/pre-built artifacts) + symlink heavy pieces."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        heavy_or_symlink = {"data", ".blackbox_inputs_v1", "__pycache__"}
        for child in source_task_dir.iterdir():
            if child.name in heavy_or_symlink:
                continue
            dst = self.work_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "build"))
            else:
                shutil.copy2(child, dst)
        # Symlink heavy dirs so we don't duplicate multi-GB references.
        for name in ("data", ".blackbox_inputs_v1"):
            src = source_task_dir / name
            if src.exists():
                os.symlink(src.resolve(), self.work_dir / name)
        self.project_dir.mkdir(parents=True, exist_ok=True)

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


def run_reference_once(session: Session) -> tuple[bool, float, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=session.work_dir,
            capture_output=True, text=True,
            timeout=REF_RUN_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, -1.0, f"reference run timed out ({REF_RUN_TIMEOUT_SEC}s)"
    combined = (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or "")
    m = _KERNEL_TIME_RE.search(combined)
    kernel_time = float(m.group(1)) if m else -1.0
    ok = proc.returncode == 0 and kernel_time > 0
    return ok, kernel_time, combined


def measure_baseline(session: Session, verbose: bool = True) -> float:
    if verbose:
        print(f"[baseline] warmup run ({BASELINE_WARMUP})...", flush=True)
    for _ in range(BASELINE_WARMUP):
        ok, kt, out = run_reference_once(session)
        if not ok:
            raise RuntimeError(f"baseline warmup failed:\n{out[-3000:]}")
    times = []
    for i in range(BASELINE_RUNS):
        ok, kt, out = run_reference_once(session)
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
    if not path.startswith(SOLUTION_ROOT):
        return f"ERROR: path '{path}' must start with '{SOLUTION_ROOT}'."
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
    return (f"ERROR: '{path}' is not readable. "
            f"You may only read files you have written OR: {sorted(READABLE_INFRA_FILES)}.")


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def do_profile(session: Session, action: dict) -> str:
    if not session.latest_build_ok:
        return ("ERROR: no successful build yet. Write your project files and let the "
                "round's eval build it first.")
    tool = action.get("tool", "")
    env = os.environ.copy()
    env["TMPDIR"] = str(session.tmp_dir)
    # Profile wraps `bash run.sh` with the same CLI eval_solution.py uses.
    input_dir = session.work_dir / ".blackbox_inputs_v1"
    profile_out = session.tmp_dir / "profile_outputs"
    profile_out.mkdir(parents=True, exist_ok=True)
    run_args = [
        "bash", "run.sh",
        "--input-dir", str(input_dir),
        "--output-dir", str(profile_out),
    ]
    ts = int(time.time() * 1000)
    try:
        if tool == "nsys":
            rep = session.tmp_dir / f"nsys_{ts}.nsys-rep"
            cmd = ["nsys", "profile", "-t", "cuda", "--stats=true",
                   "--force-overwrite", "true", "-o", str(rep)] + run_args
        elif tool == "ncu":
            kregex = action.get("kernel", "").strip()
            if not kregex:
                return "ERROR: ncu requires a 'kernel' regex."
            cmd = ["ncu", "--set", "basic", "--launch-count", "10",
                   "--kernel-name", f"regex:{kregex}",
                   "--target-processes", "all"] + run_args
        else:
            return f"ERROR: unknown profile tool '{tool}'. Use 'nsys' or 'ncu'."
        proc = subprocess.run(
            cmd, cwd=session.project_dir, env=env,
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
    eval_script = session.work_dir / "eval_solution.py"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), str(session.project_dir),
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
        "workloads": result.get("workloads", {}),
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
    lines.append(f"Total files: {len(session.written_files)}  "
                 f"Round {round_idx}/{total_rounds} ({total_rounds - round_idx} left).")
    mp = f"{SOLUTION_ROOT}project_manifest.json"
    if mp not in session.written_files:
        lines.append("  NOTE: project_manifest.json not yet written — required before first build.")
    return "\n".join(lines)


def build_planning_message(task_description: str, baseline_time_ms: float) -> str:
    return (f"{task_description}\n\n"
            "=== PLANNING PHASE ===\n"
            f"Baseline (reference MD bench): {baseline_time_ms:.2f} ms total kernel time "
            f"(mean of {BASELINE_RUNS} runs after {BASELINE_WARMUP} warmup).\n\n"
            "Produce a concise plan covering:\n"
            "  1. Project layout — how many files, how ops split across them.\n"
            "  2. Neighbor-list strategy — cell list, Verlet, rebuild frequency.\n"
            "  3. Force kernels — per op (LJ / Tersoff / Coulomb real / Coulomb k-space).\n"
            "  4. Integration / energy accounting strategy.\n"
            "  5. Round budget — what you'll have by round 5 vs 10 vs 20.\n\n"
            "Output ONLY plan text — no JSON actions, no code blocks. Revise later with `update_plan`.")


def build_user_message_round1(session: Session, task_description: str, total_rounds: int) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    cov = _coverage_block(session, 1, total_rounds) + "\n\n"
    bl = f"Baseline: {session.baseline_time_ms:.2f} ms.\n"
    return (f"{plan_block}{cov}{bl}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            "Begin writing your project. A minimum viable first-round submission:\n"
            "  generated_project/project_manifest.json\n"
            "  generated_project/build.sh\n"
            "  generated_project/run.sh\n"
            "  generated_project/<cuda-sources>.cu (covering all 7 required ops)\n"
            "Reading `build_contract.md` and `project_manifest_example.json` first is helpful.")


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
        wl = eval_result.get("workloads") or {}
        if wl:
            wl_lines = []
            for name, d in wl.items():
                if isinstance(d, dict):
                    wl_lines.append(
                        f"  {name}: force_correct={d.get('force_correct')}  "
                        f"energy_correct={d.get('energy_correct')}  "
                        f"energy_drift={d.get('energy_drift', '?')}  "
                        f"time_ms={d.get('time_ms', '?')}  "
                        f"max_rel_err={d.get('max_rel_err', '?')}"
                    )
            if wl_lines:
                parts.append("per-workload:\n" + "\n".join(wl_lines))
        if eval_result.get("error"):
            parts.append(f"error: {eval_result['error'][:1000]}")
        if eval_result.get("stderr_errors"):
            parts.append(f"\ncompile/runtime errors (extracted):\n{eval_result['stderr_errors']}")
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
    print("[setup] measuring baseline (reference MD bench)...", flush=True)
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
