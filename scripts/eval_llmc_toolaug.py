#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/llmc (black-box GPT-2 training project).

Customized loop:
  - LLM produces a complete self-contained project under `generated_project/`
    (build.sh + run.sh + CUDA sources + project_manifest.json).
  - Baseline = upstream llm.c reference, measured once at setup
    (1 warmup + 5 timed runs of `python run.py`, mean kernel_time_ms).
  - Planning phase (round 0): model produces a plan; no actions parsed.
  - Rounds 1..N: model may write/read/profile/update_plan. If any write_file
    occurred in a round, the harness stages the project and runs
    `python eval_solution.py <candidate_dir>` which builds + trains the
    candidate, parses losses, and compares kernel time to the ref.
  - write_file allows any path under `generated_project/`, scans for
    forbidden references (llmc/*.cuh, train_gpt2*, tasks/class3/llmc/src).
  - read_file: only files the LLM has written this session — there is NO
    readable infrastructure (the LLM writes the entire project from scratch).

Usage:
    python scripts/eval_llmc_toolaug.py \
        --model "Qwen/Qwen3.5-122B-A10B" \
        --api-base http://134.84.150.135:8001/v1 --api-key EMPTY \
        --run-name llmc_tool_qwen
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
TASK_ID = "class3/llmc/blackwell"
TASK_DIR_REL = "tasks/class3/llmc"
SOLUTION_ROOT = "generated_project/"  # every write path must live under this
REQUIRED_OPS = ["attention", "layernorm", "gelu", "encoder",
                "fused_classifier", "adamw", "global_norm"]
FORBIDDEN_PATTERNS = [
    (re.compile(r"#include\s*[<\"].*llmc/", re.IGNORECASE),
     "#include <...llmc/...> — upstream reference header"),
    (re.compile(r"\btrain_gpt2cu\b", re.IGNORECASE), "train_gpt2cu — reference binary"),
    (re.compile(r"\btrain_gpt2\.cu\b", re.IGNORECASE), "train_gpt2.cu — reference source"),
    (re.compile(r"tasks/class3/llmc/src", re.IGNORECASE), "tasks/class3/llmc/src — reference path"),
]

BASELINE_WARMUP = 1
BASELINE_RUNS = 5
MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
REF_RUN_TIMEOUT_SEC = 300
EVAL_TIMEOUT_SEC = 600
PROFILE_TIMEOUT_SEC = 900

# Training parameters mirror eval_solution.py / run.py
TRAINING_STEPS = 20
BATCH_SIZE = 4
SEQ_LEN = 256
LEARNING_RATE = 3e-4
VAL_EVERY = 10
SEED = 1337
DTYPE = "bf16"
OVERFIT_SINGLE_BATCH = 1

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA engineer writing a complete black-box GPT-2 training
project from scratch, inside a fixed tool-augmented evaluation harness.

## Task

Produce a self-contained project under `generated_project/` that trains
**GPT-2 large (774M)** on TinyShakespeare for 20 steps in bf16. The required
architecture is:

  - layers = 36
  - hidden / channels = 1280
  - heads = 20
  - head_dim = 64
  - vocab_size = 50257 (GPT-2 BPE)
  - max seq_len supported >= 256
  - ffn = 4 * hidden = 5120

The upstream reference (llm.c by Karpathy) is HIDDEN from you; you must
implement attention + all the GPT-2 training ops with your own CUDA kernels
(plus optional cuBLAS for matmul).

Your project must produce these outputs so the benchmark can parse them:

  Passed
  Kernel time: <ms> ms
  Solution loss: <first_loss> -> <last_loss>
  val loss: <first_val_loss> -> <last_val_loss>
  Solution loss checkpoints: 25%=... | 50%=... | 75%=... | 100%=...
  tok/s: <value>
  Loss per step: v1,v2,...,vN

The `Loss per step` line is comma-separated, no spaces, one value per training step,
in order.

Before `run.sh` exits, it must also write the final trained model weights to
`final_weights.bin` inside the project directory. This file is used by the
benchmark for post-run validation; any binary format containing the bf16/fp32
weights is fine.

Correctness requires:
  - process exits 0
  - training loss decreases from first to last step
  - final loss < 8.0
  - the loss trajectory reflects an honest training run on the specified model.

Fabricated loss curves or solutions that train a different model than what the
task specifies will not be accepted.

## Required project layout

- `generated_project/project_manifest.json` — JSON object with keys:
  - `build_script`: string, usually `"build.sh"`
  - `run_script`: string, usually `"run.sh"`
  - `files`: list of relative file paths that make up your project
  - `cuda_sources`: list of files that contain your __global__ kernels
  - `ops`: map from each required op → file that implements it
- `generated_project/build.sh` — invoked via `bash build.sh` from generated_project/
- `generated_project/run.sh` — invoked via
  `bash run.sh --train-bin <P> --val-bin <P> --steps 20 --batch-size 4 --seq-len 256 \\
   --learning-rate 3e-4 --val-every 10 --seed 1337 --dtype bf16 \\
   --overfit-single-batch 1 [--ref-time-ms <F>]`
- one or more CUDA/C++ source files listed in `cuda_sources`, containing real
  `__global__` kernels and `<<<>>>` launches

Required ops (must be keys in `manifest.ops`):
    attention, layernorm, gelu, encoder, fused_classifier, adamw, global_norm

`matmul` may use cuBLAS (not required in `ops`).

## Evaluation

The harness measures the upstream reference once at setup (1 warmup + 5 runs of its
GPT-2 trainer) to get `baseline_time_ms`. Each round with a write_file triggers:

  1. Stage your `generated_project/` directory to a tmp sandbox.
  2. Validate `project_manifest.json`, scan for forbidden references.
  3. `bash build.sh` in the sandbox.
  4. `bash run.sh --train-bin ... --ref-time-ms <baseline>` in the sandbox.
  5. Parse output; `speedup = baseline_time_ms / kernel_time_ms`.

## Anti-cheat (enforced both at write_file and at eval time)

Your code must NOT contain:
  - `#include` of any `llmc/...` header
  - references to `train_gpt2cu`, `train_gpt2.cu`
  - the literal path `tasks/class3/llmc/src`

These names belong to the hidden reference. Your project must be fully self-contained
with your own kernels.

## Phases

**Phase 0 (planning)** — The first message asks for a free-text plan. Output plan text
only; no JSON actions, no code blocks. Describe your architecture: forward kernels,
backward kernels, optimizer, data pipeline, fused ops, bf16 handling, etc.

**Phase 1..N (implementation)** — Each round you respond with actions. The controller
executes them in order and replies with results.

## Action format (strict)

Every action MUST start with a JSON object on its own line:
`{"action":"<name>", ...}`. Code/text blocks without a preceding action header are
ignored.

## Available actions

1. `write_file` — write a file under `generated_project/`. The path is relative to
   the project root. Any project layout is allowed.
   ```
   {"action":"write_file","path":"generated_project/project_manifest.json"}
   ```json
   {"build_script":"build.sh","run_script":"run.sh","files":[...],"cuda_sources":[...],"ops":{...}}
   ```
   ```
   {"action":"write_file","path":"generated_project/build.sh"}
   ```bash
   #!/bin/bash
   nvcc -O3 -o train src/train.cu
   ```
   ```
   {"action":"write_file","path":"generated_project/src/train.cu"}
   ```cuda
   // ...
   ```
   Same file at most once per round.

2. `read_file` — read back a file you have written earlier in THIS session.
   ```
   {"action":"read_file","path":"generated_project/src/attention.cu"}
   ```
   Returns your latest version. No readable reference infrastructure exists — you
   must remember what you wrote, or write a plan in `update_plan`.

3. `profile` — run a profiler over your BUILT project (requires a prior successful
   build in an earlier round).
   - `nsys`: timeline + kernel summary.
     ```
     {"action":"profile","tool":"nsys"}
     ```
   - `ncu`: per-kernel micro-metrics. Requires a regex matching kernel name.
     ```
     {"action":"profile","tool":"ncu","kernel":"attention.*"}
     ```
   Profile uses a shortened run (fewer steps) and is capped at 2 calls per round.

4. `update_plan` — revise your stored plan. Replaces the previous plan; shown to you
   every round.
   ```
   {"action":"update_plan"}
   ```text
   revised plan text
   ```

## Round rules

- Round 1: any action allowed. Minimum viable project to trigger a first build is:
  `project_manifest.json` + `build.sh` + `run.sh` + at least one CUDA source file
  + manifest.ops covering all 7 required ops.
- Rounds 2+: any action combination; execute in the written order.
- If the round contains any `write_file`, the harness runs the full eval.
"""


# ── Action parser (same as cuszp) ────────────────────────────────────────────

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
            act["code"] = code
            if not code:
                act["error"] = "write_file requires a fenced code block after the JSON."
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
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="llmc_toolaug_"))
        # work_dir/ mirrors the task dir for referencing run.py + data/; the LLM's
        # files all live under work_dir/generated_project/.
        self.work_dir = self.tmp_dir / "llmc"
        self.project_dir = self.work_dir / "generated_project"
        self.written_files: dict[str, str] = {}  # key: path relative to work_dir
        self.rounds: list[dict] = []
        self.current_plan: str = ""
        self.baseline_time_ms: float = -1.0
        self.best_speedup: float = -1.0
        self.best_round: int = -1
        self.latest_build_ok: bool = False

    def setup(self, source_task_dir: Path):
        """Copy run.py + eval_solution.py + data/; empty generated_project/."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Files we need for eval_solution.py to work (it calls run.py, which
        # builds src/train_gpt2cu and trains the reference).
        keep = ["run.py", "eval_solution.py", "prepare_data.sh",
                "build_contract.md", "project_manifest_example.json"]
        for name in keep:
            src = source_task_dir / name
            if src.is_file():
                shutil.copy2(src, self.work_dir / name)
        # src/ is the reference; symlink so run.py's build() finds it
        src_src = source_task_dir / "src"
        if src_src.exists():
            os.symlink(src_src.resolve(), self.work_dir / "src")
        # data/ holds the TinyShakespeare bins (~20MB); symlink
        data_src = source_task_dir / "data"
        if data_src.exists():
            os.symlink(data_src.resolve(), self.work_dir / "data")
        # task config dirs
        for name in ("general", "hopper", "blackwell"):
            d = source_task_dir / name
            if d.is_dir():
                shutil.copytree(d, self.work_dir / name)
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


# ── Baseline measurement ──────────────────────────────────────────────────────

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
    """Run `python run.py` to train the reference. Returns (ok, kernel_time_ms, stdout)."""
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
        if kt > 0:
            times.append(kt)
            if verbose:
                print(f"[baseline] run {i+1}/{BASELINE_RUNS}: kernel_time={kt:.2f} ms", flush=True)
    if not times:
        raise RuntimeError("no valid baseline runs")
    return mean(times)


# ── Action executors ──────────────────────────────────────────────────────────

def _is_safe_relpath(path: str) -> bool:
    p = path.replace("\\", "/")
    return bool(p and not p.startswith("/") and p not in (".", "..")
                and not p.startswith("../") and "/../" not in p)


def do_write_file(session: Session, action: dict) -> str:
    path = action["path"]
    if not path.startswith(SOLUTION_ROOT):
        return (f"ERROR: path '{path}' must start with '{SOLUTION_ROOT}'. "
                f"All project files live under generated_project/.")
    if not _is_safe_relpath(path):
        return f"ERROR: unsafe path '{path}'."
    code = action["code"]
    if not code.strip():
        return f"ERROR: empty code block for {path}."
    # Anti-cheat scan
    for pat, desc in FORBIDDEN_PATTERNS:
        if pat.search(code):
            return f"ERROR: '{path}' contains forbidden pattern — {desc}. Your project must be self-contained."
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
    return (f"ERROR: '{path}' has not been written in this session. "
            f"Available: {sorted(session.written_files.keys()) or 'none'}. "
            f"Reference files are hidden.")


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def do_profile(session: Session, action: dict) -> str:
    if not session.latest_build_ok:
        return ("ERROR: no successful build yet. Trigger a build by writing your "
                "project files (manifest + build.sh + run.sh + sources) and letting "
                "the harness run the eval at round end.")
    tool = action.get("tool", "")
    env = os.environ.copy()
    env["TMPDIR"] = str(session.tmp_dir)
    env["NO_MULTI_GPU"] = "1"
    # Profile via bash run.sh (matches how eval_solution.py runs it)
    data_dir = session.work_dir / "data"
    train_bin = str((data_dir / "tiny_shakespeare_train.bin").resolve())
    val_bin = str((data_dir / "tiny_shakespeare_val.bin").resolve())
    # Short profile run — fewer steps to stay within PROFILE_TIMEOUT_SEC
    run_args = [
        "bash", "run.sh",
        "--train-bin", train_bin, "--val-bin", val_bin,
        "--steps", "5", "--batch-size", str(BATCH_SIZE),
        "--seq-len", str(SEQ_LEN), "--learning-rate", str(LEARNING_RATE),
        "--val-every", "10", "--seed", str(SEED), "--dtype", DTYPE,
        "--overfit-single-batch", str(OVERFIT_SINGLE_BATCH),
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
                return "ERROR: ncu requires a 'kernel' regex. Use nsys first to see actual kernel names."
            cmd = ["ncu", "--set", "basic", "--launch-count", "5",
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
        status = f"exit={proc.returncode}"
        return f"{tool.upper()} PROFILE ({status}):\n" + _truncate(combined)
    except subprocess.TimeoutExpired:
        return f"ERROR: {tool} timed out after {PROFILE_TIMEOUT_SEC}s."
    except FileNotFoundError as e:
        return f"ERROR: {tool} not installed on this machine ({e})."


# ── Per-round evaluation ──────────────────────────────────────────────────────

def run_eval(session: Session) -> dict:
    """Run eval_solution.py on the current generated_project/. Parses its JSON output."""
    eval_script = session.work_dir / "eval_solution.py"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), str(session.project_dir),
             "--timeout", str(EVAL_TIMEOUT_SEC // 2)],
            cwd=session.work_dir,
            capture_output=True, text=True, timeout=EVAL_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {"compiled": False, "correct": False, "speedup": -1.0,
                "error": f"eval_solution.py timed out after {EVAL_TIMEOUT_SEC}s",
                "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    # eval_solution.py prints a JSON object (or minimal _fail JSON) to stdout.
    result = {}
    # Find the LAST JSON object in stdout (eval_solution may pretty-print with
    # indentation, so json.loads on everything works in most cases)
    try:
        # Try parsing whole stdout first
        result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        # Fall back to scanning for a top-level object
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
    compiled = bool(result.get("compiled", False))
    correct = bool(result.get("correct", False))
    kernel_time = float(result.get("kernel_time_ms", -1.0)) if result.get("kernel_time_ms") not in (None, -1) else -1.0
    ref_time = float(result.get("ref_time_ms", session.baseline_time_ms)) if result.get("ref_time_ms") not in (None, -1) else session.baseline_time_ms
    # Our reported speedup is relative to OUR cached baseline (stable across rounds).
    speedup = (session.baseline_time_ms / kernel_time
               if session.baseline_time_ms > 0 and kernel_time > 0 else -1.0)
    if compiled and correct and not session.latest_build_ok:
        session.latest_build_ok = True
    return {
        "compiled": compiled,
        "correct": correct,
        "kernel_time_ms": kernel_time,
        "baseline_time_ms": session.baseline_time_ms,
        "fresh_ref_time_ms": ref_time,  # eval_solution.py's per-call ref (noisy)
        "speedup": speedup,
        "fresh_speedup": float(result.get("speedup", -1.0)) if result.get("speedup") not in (None, -1) else -1.0,
        "loss_ratio": float(result.get("loss_ratio", -1.0)) if result.get("loss_ratio") not in (None, -1) else -1.0,
        "solution_loss": result.get("solution_loss", {}),
        "ref_loss": result.get("ref_loss", {}),
        "tok_per_sec": float(result.get("tok_per_sec", -1.0)) if result.get("tok_per_sec") not in (None, -1) else -1.0,
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
    # ops coverage summary
    lines.append(f"Total files: {len(session.written_files)}  "
                 f"Round {round_idx}/{total_rounds} ({total_rounds - round_idx} left).")
    # Has manifest?
    mp = f"{SOLUTION_ROOT}project_manifest.json"
    if mp not in session.written_files:
        lines.append("  NOTE: project_manifest.json not yet written — required before first build.")
    return "\n".join(lines)


def build_planning_message(task_description: str, baseline_time_ms: float) -> str:
    return (f"{task_description}\n\n"
            "=== PLANNING PHASE ===\n"
            f"Baseline (upstream llm.c reference): {baseline_time_ms:.2f} ms total "
            f"kernel time (mean of {BASELINE_RUNS} runs after {BASELINE_WARMUP} warmup).\n\n"
            "You are writing a complete GPT-2 training project from scratch. Your plan "
            "should cover:\n"
            "  1. Overall architecture — which files, how they are laid out.\n"
            "  2. Required ops coverage — one bullet per op (attention / layernorm / gelu / "
            "encoder / fused_classifier / adamw / global_norm), pointing at the file that "
            "will implement it and a short note on the algorithm.\n"
            "  3. Round budget — in which round do you plan to have the first fully "
            "compiling + loss-converging version? What comes after?\n"
            "  4. bf16 + cuBLAS strategy — how mixed precision is handled; do you use "
            "cuBLAS for matmul or roll your own.\n"
            "  5. Data pipeline — how run.sh parses args and feeds the trainer.\n\n"
            "Output ONLY plan text — no JSON actions, no code blocks. The controller stores "
            "the plan and re-shows it every round. Revise later with `update_plan`.")


def build_user_message_round1(session: Session, task_description: str, total_rounds: int) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    cov = _coverage_block(session, 1, total_rounds) + "\n\n"
    bl = f"Baseline: {session.baseline_time_ms:.2f} ms.\n"
    return (f"{plan_block}{cov}{bl}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            "Begin writing your project. A complete first-round project would include:\n"
            "  generated_project/project_manifest.json\n"
            "  generated_project/build.sh\n"
            "  generated_project/run.sh\n"
            "  generated_project/<your_cuda_source>.cu\n"
            "If you write all of these this round (plus optionally more), the harness will\n"
            "build + run your project and report results. Partial round 1 is also OK —\n"
            "you can finish in round 2. No eval runs if no write_file occurred.")


def build_feedback(session: Session, round_idx: int, total_rounds: int,
                   action_outputs: list[str], eval_result: dict | None) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    coverage_block = _coverage_block(session, round_idx, total_rounds) + "\n\n"
    parts = [f"{plan_block}{coverage_block}=== ROUND {round_idx} / {total_rounds} RESULTS ==="]
    for i, out in enumerate(action_outputs, 1):
        parts.append(f"\n--- action {i} output ---\n{out}")
    if eval_result is not None:
        parts.append("\n--- evaluation (eval_solution.py) ---")
        sp = eval_result.get("speedup", -1.0)
        kt = eval_result.get("kernel_time_ms", -1.0)
        bl = eval_result.get("baseline_time_ms", session.baseline_time_ms)
        lr = eval_result.get("loss_ratio", -1)
        summary = (f"compiled={eval_result.get('compiled')}  "
                   f"correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  "
                   f"speedup={sp:.4f}x" if sp > 0 else
                   f"compiled={eval_result.get('compiled')}  correct={eval_result.get('correct')}  "
                   f"kernel_time={kt:.2f}ms  baseline={bl:.2f}ms  speedup=N/A")
        parts.append(summary)
        sl = eval_result.get("solution_loss") or {}
        rl = eval_result.get("ref_loss") or {}
        if sl and rl and "first" in sl and "last" in sl:
            parts.append(f"solution_loss: {sl.get('first', '?')} -> {sl.get('last', '?')}  "
                         f"val: {sl.get('first_val', '?')} -> {sl.get('last_val', '?')}")
        if rl and "first" in rl:
            parts.append(f"ref_loss: {rl.get('first', '?')} -> {rl.get('last', '?')}")
        if lr and lr > 0:
            parts.append(f"loss_ratio (sol/ref): {lr:.4f}")
        if eval_result.get("tok_per_sec", -1) > 0:
            parts.append(f"tok/s: {eval_result['tok_per_sec']:.0f}")
        if eval_result.get("error"):
            parts.append(f"error: {eval_result['error'][:1000]}")
        if eval_result.get("stderr_errors"):
            parts.append(f"\ncompile/runtime errors (extracted):\n{eval_result['stderr_errors']}")
        stdout_tail = eval_result.get('stdout_tail', '')
        if stdout_tail.strip():
            parts.append(f"\nstdout tail:\n{stdout_tail[:1500]}")
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
    ap.add_argument("--num-rounds", type=int, default=30)
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
    print("[setup] measuring baseline (upstream llm.c reference)...", flush=True)
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
        # Planning phase
        planning_msg = build_planning_message(task_description, session.baseline_time_ms)
        messages.append({"role": "user", "content": planning_msg})
        log("user", planning_msg)
        plan_response = call_model("Plan phase")
        messages.append({"role": "assistant", "content": plan_response})
        log("assistant", plan_response)
        session.current_plan = plan_response.strip()
        print(f"  plan stored ({len(session.current_plan)} chars)", flush=True)

        # Implementation rounds
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
