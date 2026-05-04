#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/tcgnn_gcn.

Per-task customized controller loop:
  - Round 1: model writes all solution files (solution.cu + wrapper.cpp).
  - Rounds 2..N: model may issue any combination of write_file / read_file / profile.
  - If a round contains any write_file, run.py is executed once at round end.
  - profile is limited to 2 calls per round (e.g. 1 nsys + 1 ncu).
  - ncu requires a kernel-name regex; nsys does not.

Usage:
    python scripts/eval_tcgnn_gcn_toolaug.py \
        --model Qwen/Qwen3.5-122B-A10B \
        --api-base http://localhost:8000/v1 --api-key EMPTY \
        --num-rounds 9 --run-name tcgnn_tool_qwen
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.llm_api import query_server
from cuda_hercules.utils import get_project_root

# ── Task-specific constants ──────────────────────────────────────────────────
TASK_ID = "class3/tcgnn_gcn/general"
TASK_DIR_REL = "tasks/class3/tcgnn_gcn"
SOLUTION_FILES = ["solution.cu"]

# Profile cost reduction: single graph, fewer bench epochs.
PROFILE_ENV = {"TCGNN_GRAPHS": "amazon0505", "KH_BENCHMARK": "30"}
EVAL_ENV = {"KH_BENCHMARK": "200"}  # matches task.yaml

MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
EVAL_TIMEOUT_SEC = 900
PROFILE_TIMEOUT_SEC = 600

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA kernel engineer working inside a fixed evaluation harness.

You are optimizing the SpMM (Sparse Matrix-Matrix Multiplication) kernel used in 2-layer GCN
training. The evaluation has two phases:

**Phase 0 (planning)** — The very first message from the harness asks you to produce a plan
in plain text. No actions are parsed in this phase. Write a concise plan describing your
algorithmic approach, data layout choices, expected kernel structure, and how you'll
validate correctness. Do NOT emit any JSON actions or code blocks in this phase.

**Phase 1..N (implementation)** — Each round you respond with one or more **actions**. The
controller parses your message, executes actions in order, and replies with results.

## Action format (strict)

Every action MUST begin with a JSON object on its own line: `{"action":"<name>", ...}`.
Any code/text block without a preceding JSON action header is ignored. If you want to
write code, you MUST precede it with a write_file JSON header.

## Available actions

1. `write_file` — overwrite a solution file.
   ```
   {"action":"write_file","path":"solution.cu"}
   ```cuda
   // full file contents (not a diff)
   ```
   Allowed path: `solution.cu`. Same file may be written at most once per round.

2. `read_file` — read back a file you wrote earlier in this session.
   ```
   {"action":"read_file","path":"solution.cu"}
   ```
   Returns current contents. No following code block is needed.

3. `profile` — run a profiler over the current solution.
   - `nsys`: timeline + kernel summary (shows which kernels dominate and their names).
     ```
     {"action":"profile","tool":"nsys"}
     ```
   - `ncu`: per-kernel micro-metrics. You MUST supply a regex via `kernel` matching the
     kernel name you want profiled (use nsys first to discover names).
     ```
     {"action":"profile","tool":"ncu","kernel":"spmm.*"}
     ```
   At most 2 profile calls per round.

4. `update_plan` — revise your stored plan. The new plan replaces the old one and is
   re-shown at the start of every future round.
   ```
   {"action":"update_plan"}
   ```text
   revised plan here
   ```

## Round rules

- Round 1: only `write_file` and `update_plan` allowed. Produce the required file(s).
- Rounds 2+: any combination is allowed. Actions execute in the order you write them.
- If a round contains any `write_file`, the controller runs the full evaluation at the
  end (`python run.py`, all 4 graphs, 200 bench epochs). If no `write_file`, no eval.

## Evaluation

The full eval compiles solution.cu + the fixed wrapper.cpp, builds a 2-layer GCN, and
trains on 4 graphs. Correctness requires the solution to converge and the final loss to
be within 10% of the reference's. Performance = geomean speedup over 4 graphs against
the TC-GNN WMMA reference.

## Required file

- `solution.cu` implements:
  ```cpp
  std::vector<torch::Tensor> spmm_forward_cuda(
      torch::Tensor nodePointer, torch::Tensor edgeList,
      torch::Tensor blockPartition, torch::Tensor edgeToColumn, torch::Tensor edgeToRow,
      int num_nodes, int num_edges, int embedding_dim,
      torch::Tensor input);
  ```
  Returns `{output}` where `output[i,:] = sum_{j in N(i)} input[j,:]`.

The `wrapper.cpp` (pybind bindings + CPU preprocess) is fixed harness code; it is NOT
writable. Your function signature above must match exactly.
"""


# ── Action parser ─────────────────────────────────────────────────────────────

_JSON_ACTION_RE = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)


def _find_code_after(text: str, start_idx: int) -> tuple[str, int]:
    """Find the next fenced code block starting at/after start_idx. Returns (code, end_pos).
    end_pos is the character position right after the closing fence (or len(text) on miss).
    """
    open_match = re.search(r"```[a-zA-Z0-9+_-]*\n", text[start_idx:])
    if not open_match:
        return "", len(text)
    open_end = start_idx + open_match.end()
    close = text.find("\n```", open_end)
    if close == -1:
        return "", len(text)
    return text[open_end:close], close + 4


def parse_actions(message: str, fallback_round1: bool = False) -> list[dict]:
    """Extract ordered action list from model response.

    Each action is a dict with keys:
      - type: "write_file" | "read_file" | "profile" | "update_plan"
      - path, code, tool, kernel, plan (as applicable)
      - raw_json: the JSON source for logging
      - error: set if the action is malformed (included in output so controller can report)

    If fallback_round1 is True and the message has no JSON action blocks but contains at
    least one code fence, we synthesize a `write_file solution.cu` action from the first
    code block. This covers models that emit a code block without the JSON header in the
    very first coding round.
    """
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
                act["error"] = "write_file requires a fenced code block immediately after the JSON."
        elif atype == "read_file":
            act["path"] = str(obj.get("path", "")).strip()
        elif atype == "profile":
            act["tool"] = str(obj.get("tool", "")).strip().lower()
            act["kernel"] = str(obj.get("kernel", "")).strip()
        elif atype == "update_plan":
            plan_text, _ = _find_code_after(message, m.end())
            act["plan"] = plan_text
            if not plan_text:
                act["error"] = "update_plan requires a fenced text block immediately after the JSON."
        else:
            act["error"] = f"Unknown action '{atype}'. Must be write_file, read_file, profile, or update_plan."
        actions.append(act)

    # Fallback: round-1 model emitted a code block without the JSON header.
    if fallback_round1 and not actions:
        code, _ = _find_code_after(message, 0)
        if code.strip():
            actions.append({
                "type": "write_file",
                "path": "solution.cu",
                "code": code,
                "raw_json": "(synthesized: no JSON action header found, assuming write_file solution.cu)",
            })
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    """Holds tmp workdir, written-file state, and round history for a single run."""

    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="tcgnn_toolaug_"))
        self.work_dir = self.tmp_dir / "tcgnn_gcn"
        self.written_files: dict[str, str] = {}
        self.rounds: list[dict] = []
        self.best_speedup = -1.0
        self.best_round = -1
        self.current_plan: str = ""

    def setup(self, source_task_dir: Path):
        """Copy task dir to tmp_dir, but symlink heavy dirs (data, __pycache__)."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Files and dirs to physically copy (harness + initial solution scaffolding).
        copy_items = ["run.py", "eval_solution.py", "ref_kernel.cu",
                      "solution.cu", "wrapper.cpp", "prepare_data.sh",
                      "general", "hopper", "blackwell"]
        # Data dir is big (~150MB of graphs) — symlink.
        for name in copy_items:
            src = source_task_dir / name
            if not src.exists():
                continue
            dst = self.work_dir / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        data_src = source_task_dir / "data"
        if data_src.exists():
            os.symlink(data_src.resolve(), self.work_dir / "data")

    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def snapshot_round(self, round_idx: int) -> dict:
        """Persist per-round artifacts into run_out_dir/round_XX/."""
        rd = self.run_out_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        for path, content in self.written_files.items():
            target = rd / path.replace("/", "__")
            target.write_text(content)
        return {"dir": str(rd)}


# ── Action executors ──────────────────────────────────────────────────────────

def do_write_file(session: Session, action: dict) -> str:
    path = action["path"]
    if path not in SOLUTION_FILES:
        return f"ERROR: path '{path}' not in allowed solution files {SOLUTION_FILES}."
    code = action["code"]
    if not code.strip():
        return f"ERROR: empty code block for {path}."
    # Basic anti-cheat: forbid reading ref_kernel.cu via #include
    if re.search(r"#\s*include\s*[\"<]ref_kernel", code):
        return f"ERROR: solution cannot include ref_kernel.cu (reference code)."
    target = session.work_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)
    session.written_files[path] = code
    return f"OK: wrote {path} ({len(code)} bytes)."


def do_read_file(session: Session, action: dict) -> str:
    path = action["path"]
    if path not in session.written_files:
        return (f"ERROR: '{path}' has not been written in this session. "
                f"Available: {sorted(session.written_files.keys()) or 'none'}.")
    content = session.written_files[path]
    return f"CONTENT of {path} ({len(content)} bytes):\n```\n{content}\n```"


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text in the code fence."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def _truncate(text: str, limit: int = MAX_PROFILE_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return text[:half] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-half:]


def extract_compile_errors(stderr: str, max_chars: int = 3000) -> str:
    """Pull real nvcc/g++ errors out of 10M-char ninja-verbose noise.

    Keeps lines containing ``error:``/``fatal error:``/``note:`` plus a few
    lines of surrounding context, and the Python-side final exception line.
    Falls back to a truncated tail if no error line matched.
    """
    if not stderr:
        return ""
    error_kw = re.compile(r"\b(error:|fatal error:|undefined reference|note:|warning:)")
    lines = stderr.splitlines()
    kept_idx: set[int] = set()
    for i, line in enumerate(lines):
        if error_kw.search(line):
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                kept_idx.add(j)
    # Always keep the last Python-level "XxxError: ..." / "Exception:" line — that
    # often summarizes what made run.py exit nonzero.
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
    joined = "\n".join(out_lines)
    return _truncate(joined, max_chars)


def do_profile(session: Session, action: dict) -> str:
    tool = action.get("tool", "")
    env = os.environ.copy()
    env.update(PROFILE_ENV)
    # Redirect nsys's scratch dir (defaults to /tmp/nvidia/nsight_systems which may be
    # owned by another user) into our session tmp so we don't hit permission errors.
    env["TMPDIR"] = str(session.tmp_dir)
    # Unique nsys output file per call to avoid clobbering.
    ts = int(time.time() * 1000)
    try:
        if tool == "nsys":
            rep = session.tmp_dir / f"nsys_{ts}.nsys-rep"
            cmd = ["nsys", "profile", "-t", "cuda", "--stats=true",
                   "--force-overwrite", "true", "-o", str(rep),
                   sys.executable, "run.py"]
        elif tool == "ncu":
            kregex = action.get("kernel", "").strip()
            if not kregex:
                return "ERROR: ncu requires a 'kernel' regex. Use nsys first to see actual kernel names."
            cmd = ["ncu", "--set", "basic", "--launch-count", "5",
                   "--kernel-name", f"regex:{kregex}",
                   "--target-processes", "all",
                   sys.executable, "run.py"]
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
        status = f"exit={proc.returncode}"
        return f"{tool.upper()} PROFILE ({status}, env: {PROFILE_ENV}):\n" + _truncate(combined)
    except subprocess.TimeoutExpired:
        return f"ERROR: {tool} timed out after {PROFILE_TIMEOUT_SEC}s."
    except FileNotFoundError as e:
        return f"ERROR: {tool} not installed on this machine ({e})."


# ── Evaluation ────────────────────────────────────────────────────────────────

_SUMMARY_RE = re.compile(r"^RUN_SUMMARY_JSON\s+(\{.*\})\s*$", re.MULTILINE)


def run_eval(session: Session) -> dict:
    env = os.environ.copy()
    env.update(EVAL_ENV)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "run.py"], cwd=session.work_dir, env=env,
            capture_output=True, text=True, timeout=EVAL_TIMEOUT_SEC,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        return {"compiled": False, "correct": False, "speedup": -1.0,
                "error": f"eval timed out after {EVAL_TIMEOUT_SEC}s", "elapsed_s": time.time() - t0}

    out, err = proc.stdout or "", proc.stderr or ""
    # Heuristic: compile failed if exit nonzero and no RUN_SUMMARY_JSON line.
    m = _SUMMARY_RE.search(out)
    if m:
        try:
            summary = json.loads(m.group(1))
        except Exception:
            summary = {}
        agg = summary.get("aggregate", {})
        return {
            "compiled": True,
            "correct": bool(summary.get("correct")),
            "speedup": float(agg.get("speedup", -1.0)),
            "ref_time_ms": float(agg.get("ref_time_ms", -1.0)),
            "sol_time_ms": float(agg.get("kernel_time_ms", -1.0)),
            "loss_ratio": float(agg.get("loss_ratio", -1.0)),
            "exit_code": proc.returncode,
            "elapsed_s": elapsed,
            "stdout_tail": _truncate(out, 1500),
            "stderr_errors": extract_compile_errors(err, 2000),
        }
    # No summary — compile or run failed early. Extract real nvcc/Python errors.
    return {
        "compiled": False,
        "correct": False,
        "speedup": -1.0,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "error": "run.py did not produce RUN_SUMMARY_JSON",
        "stdout_tail": _truncate(out, 1500),
        "stderr_errors": extract_compile_errors(err, 3000),
    }


# ── Round-level controller ────────────────────────────────────────────────────

def build_planning_message(task_description: str) -> str:
    return (f"{task_description}\n\n"
            "=== PLANNING PHASE ===\n"
            "Before writing any code, produce a concise plan for your approach. Cover:\n"
            "  1. High-level kernel strategy (naive CSR SpMM vs Tensor Core / WMMA vs other)\n"
            "  2. Data layout / memory access pattern for input features\n"
            "  3. How you will parallelize across nodes / edges / feature dims\n"
            "  4. How you will validate correctness (the reference uses TC-GNN WMMA TF32)\n\n"
            "Output ONLY plan text — no actions, no code blocks. The controller stores your\n"
            "plan and will show it at the start of every implementation round. You can revise\n"
            "the plan later with the `update_plan` action.")


def build_user_message_round1(session: Session, task_description: str) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    return (f"{plan_block}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            f"Produce the required file(s): {', '.join(SOLUTION_FILES)}\n"
            "Only `write_file` and `update_plan` actions are allowed this round. After this\n"
            "round the harness will evaluate and report results back to you.")


def build_feedback(session: Session, round_idx: int, total_rounds: int,
                   action_outputs: list[str], eval_result: dict | None) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    parts = [f"{plan_block}=== ROUND {round_idx} / {total_rounds} RESULTS ==="]
    for i, out in enumerate(action_outputs, 1):
        parts.append(f"\n--- action {i} output ---\n{out}")
    if eval_result is not None:
        parts.append("\n--- evaluation (python run.py) ---")
        sp = eval_result.get("speedup", -1.0)
        parts.append(f"compiled={eval_result.get('compiled')}  "
                     f"correct={eval_result.get('correct')}  "
                     f"speedup={sp:.4f}x" if sp >= 0 else
                     f"compiled={eval_result.get('compiled')}  correct={eval_result.get('correct')}  speedup=N/A")
        if eval_result.get("ref_time_ms", -1) > 0:
            parts.append(f"ref_time={eval_result['ref_time_ms']:.2f}ms  "
                         f"sol_time={eval_result['sol_time_ms']:.2f}ms  "
                         f"loss_ratio={eval_result['loss_ratio']:.4f}")
        if eval_result.get("error"):
            parts.append(f"error: {eval_result['error']}")
        parts.append(f"\nstdout tail:\n{eval_result.get('stdout_tail','')[:1500]}")
        errs = eval_result.get("stderr_errors", "")
        if errs:
            parts.append(f"\ncompile/runtime errors (extracted):\n{errs}")
    parts.append(f"\n=== ROUND {round_idx+1} / {total_rounds} ===" if round_idx < total_rounds
                 else "\n(final round complete)")
    return "\n".join(parts)


def execute_round(session: Session, round_idx: int, actions: list[dict]) -> tuple[list[str], dict | None]:
    """Run a round's actions in order. Return (per-action outputs, eval_result-or-None)."""
    outputs = []
    profile_calls = 0
    seen_writes = set()
    any_write = False
    for act in actions:
        if "error" in act and act.get("type") not in ("profile", "update_plan"):
            outputs.append(f"REJECTED: {act['error']}  raw={act['raw_json'][:200]}")
            continue
        atype = act.get("type")
        if round_idx == 1 and atype not in ("write_file", "update_plan"):
            outputs.append(f"REJECTED (round 1 allows write_file / update_plan only): {atype}")
            continue
        if atype == "write_file":
            if act["path"] in seen_writes:
                outputs.append(f"REJECTED: {act['path']} already written this round (same file cannot appear twice).")
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
    ap.add_argument("--num-rounds", type=int, default=10)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(get_project_root())
    src_task = root / TASK_DIR_REL
    out_dir = Path(args.output) if args.output else root / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_path = src_task / "general" / "description.txt"
    task_description = desc_path.read_text() if desc_path.exists() else ""

    session = Session(run_out_dir=out_dir)
    session.setup(src_task)

    # Conversation: system + rolling user/assistant turns.
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    conversation_log_path = out_dir / "conversation.jsonl"
    conversation_log = conversation_log_path.open("w")

    def log(role: str, content: str):
        conversation_log.write(json.dumps({"role": role, "content": content}) + "\n")
        conversation_log.flush()

    log("system", SYSTEM_PROMPT)

    def call_model(label: str) -> str:
        print(f"\n[{label}] querying model...", flush=True)
        t_q = time.time()
        resp = query_server(
            prompt=messages,
            model=args.model,
            system_prompt="",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_base=args.api_base,
            api_key=args.api_key,
            backend=args.backend,
            is_reasoning_model=args.reasoning,
            reasoning_effort=args.reasoning_effort,
        )
        if isinstance(resp, list):
            resp = resp[0]
        print(f"  model responded in {time.time() - t_q:.1f}s, {len(resp)} chars", flush=True)
        return resp

    t_start = time.time()
    try:
        # --- Planning phase (round 0) ---
        planning_msg = build_planning_message(task_description)
        messages.append({"role": "user", "content": planning_msg})
        log("user", planning_msg)
        plan_response = call_model("Plan phase")
        messages.append({"role": "assistant", "content": plan_response})
        log("assistant", plan_response)
        # Store plan text (strip fenced blocks if model accidentally used them).
        session.current_plan = plan_response.strip()
        print(f"  plan stored ({len(session.current_plan)} chars)", flush=True)

        # --- Implementation rounds 1..N ---
        first_impl_msg = build_user_message_round1(session, task_description)
        messages.append({"role": "user", "content": first_impl_msg})
        log("user", first_impl_msg)

        for round_idx in range(1, args.num_rounds + 1):
            response = call_model(f"Round {round_idx}/{args.num_rounds}")
            messages.append({"role": "assistant", "content": response})
            log("assistant", response)

            # Round-1 fallback: allow code block without JSON header → write solution.cu.
            actions = parse_actions(response, fallback_round1=(round_idx == 1))
            print(f"  parsed {len(actions)} actions: " +
                  ", ".join(a.get("type", "?") for a in actions), flush=True)

            t_r = time.time()
            outputs, eval_result = execute_round(session, round_idx, actions)
            print(f"  executed in {time.time() - t_r:.1f}s", flush=True)

            # Track best.
            if eval_result and eval_result.get("correct") and eval_result.get("speedup", -1) > 0:
                if eval_result["speedup"] > session.best_speedup:
                    session.best_speedup = eval_result["speedup"]
                    session.best_round = round_idx

            # Snapshot (always save current written files).
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
                print(f"  eval: correct={eval_result.get('correct')}  speedup={sp:.4f}x"
                      if sp >= 0 else f"  eval: correct={eval_result.get('correct')}  speedup=N/A",
                      flush=True)

            feedback = build_feedback(session, round_idx, args.num_rounds, outputs, eval_result)
            messages.append({"role": "user", "content": feedback})
            log("user", feedback)
    finally:
        conversation_log.close()

    elapsed = time.time() - t_start

    final = {
        "task_id": TASK_ID,
        "model": args.model,
        "num_rounds": args.num_rounds,
        "best_speedup": session.best_speedup,
        "best_round": session.best_round,
        "elapsed_s": elapsed,
        "rounds": session.rounds,
    }
    (out_dir / "final_report.json").write_text(json.dumps(final, indent=2))

    # Persist best-so-far solution files for convenience.
    if session.best_round > 0:
        best_dir = out_dir / f"round_{session.best_round:02d}"
        for fname in SOLUTION_FILES:
            src = best_dir / fname.replace("/", "__")
            if src.exists():
                shutil.copy2(src, out_dir / f"best_{fname}")

    print(f"\n=== DONE ({elapsed:.1f}s) ===")
    print(f"best_speedup = {session.best_speedup:.4f}x (round {session.best_round})")
    print(f"results in: {out_dir}")

    session.cleanup()


if __name__ == "__main__":
    main()
