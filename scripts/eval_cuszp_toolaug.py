#!/usr/bin/env python3
"""Tool-augmented evaluator for class3/cuszp.

Customized loop:
  - Baseline measured once at setup: full cmake+make, then 1 warmup + 5 bench runs.
    Mean bench kernel time = baseline_time_ms. Used to report speedup to the model.
  - Planning phase (round 0): model produces a plan; no actions parsed.
  - Rounds 1..N: model may write/read/profile/update_plan. If any write_file occurred,
    the harness does an incremental `make` + single bench run at round end.
  - write_file paths restricted to the 6 `src/cuSZp_kernels_{1D,2D,3D}_{f32,f64}.cu`.
  - read_file whitelist: either (a) something the model has written this session,
    or (b) readable infrastructure (entries, main, timer, headers, CMakeLists).
    Baseline kernel `.cu` contents are NEVER returned until the model writes its own.

Usage:
    python scripts/eval_cuszp_toolaug.py \
        --model "Qwen/Qwen3.5-122B-A10B" \
        --api-base http://134.84.150.135:8001/v1 --api-key EMPTY \
        --run-name cuszp_tool_qwen
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
TASK_ID = "class3/cuszp/blackwell"
TASK_DIR_REL = "tasks/class3/cuszp"

SOLUTION_FILES = [
    "src/cuSZp_kernels_1D_f32.cu",
    "src/cuSZp_kernels_1D_f64.cu",
    "src/cuSZp_kernels_2D_f32.cu",
    "src/cuSZp_kernels_2D_f64.cu",
    "src/cuSZp_kernels_3D_f32.cu",
    "src/cuSZp_kernels_3D_f64.cu",
]

# Per-family scoring (Plan C):
#   speedup = arithmetic mean over 6 families of per-family speedup
#   per-family speedup = 0 if family not submitted by the model OR fails correctness
#                      = baseline_family_time / solution_family_time otherwise
VARIANTS = ["1D_f32", "1D_f64", "2D_f32", "2D_f64", "3D_f32", "3D_f64"]
VARIANT_TO_FILE = {v: f"src/cuSZp_kernels_{v}.cu" for v in VARIANTS}
FILE_TO_VARIANT = {v: k for k, v in VARIANT_TO_FILE.items()}

# Files the model may read (interface / infrastructure) even without having written them.
# Baseline kernel .cu bodies are intentionally NOT here.
READABLE_INFRA_FILES = {
    "src/main.cu",
    "src/cuSZp_timer.cu",
    "src/CMakeLists.txt",
    # entry wrappers that show how kernels are launched
    "src/cuSZp_entry_1D_f32.cu",
    "src/cuSZp_entry_1D_f64.cu",
    "src/cuSZp_entry_2D_f32.cu",
    "src/cuSZp_entry_2D_f64.cu",
    "src/cuSZp_entry_3D_f32.cu",
    "src/cuSZp_entry_3D_f64.cu",
    # headers (kernel signatures + constants)
    "src/include/cuSZp/cuSZp_entry_1D_f32.h",
    "src/include/cuSZp/cuSZp_entry_1D_f64.h",
    "src/include/cuSZp/cuSZp_entry_2D_f32.h",
    "src/include/cuSZp/cuSZp_entry_2D_f64.h",
    "src/include/cuSZp/cuSZp_entry_3D_f32.h",
    "src/include/cuSZp/cuSZp_entry_3D_f64.h",
    "src/include/cuSZp/cuSZp_kernels_1D_f32.h",
    "src/include/cuSZp/cuSZp_kernels_1D_f64.h",
    "src/include/cuSZp/cuSZp_kernels_2D_f32.h",
    "src/include/cuSZp/cuSZp_kernels_2D_f64.h",
    "src/include/cuSZp/cuSZp_kernels_3D_f32.h",
    "src/include/cuSZp/cuSZp_kernels_3D_f64.h",
    "src/include/cuSZp/cuSZp_timer.h",
}

BUILD_SUBPATH = "src/cmake-build-release"
EXECUTABLE_NAME = "cuSZp_bench"
# Reference implementations (PTX-tuned cuSZp originals) live OUTSIDE the LLM-writable
# `src/` tree so the LLM cannot inherit them by leaving a kernel file unwritten.
# We measure the SC'23 baseline by overlaying them into a separate baseline work_dir.
REFERENCE_SUBDIR = "reference"
BASELINE_WARMUP = 1
BASELINE_RUNS = 5
MAX_PROFILE_PER_ROUND = 2
MAX_PROFILE_OUTPUT = 4000
BUILD_TIMEOUT_SEC = 600
BENCH_TIMEOUT_SEC = 600
PROFILE_TIMEOUT_SEC = 900

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CUDA kernel engineer optimizing cuSZp, a GPU error-bounded
lossy compression library, inside a fixed tool-augmented evaluation harness.

## Task

Six kernel files together implement compression + decompression for all combinations of
{1D, 2D, 3D} × {float32, float64}. Each file exposes 3 encoding modes
(fixed / plain / outlier), each with a `compress` and `decompress` kernel (6 kernels per
file, 36 kernels total). You will rewrite kernel implementations from scratch; the
original SC'23 baseline is NOT shown to you. You have full access to the kernel
signatures (headers), the entry wrappers that call the kernels, the benchmark driver,
and CMake build configuration.

## Evaluation

The harness builds `cuSZp_bench` from source and runs it. It tests all 108 kernel
invocations (6 variants × 3 modes × (3 REL error bounds | 3-5 runs)) on 2 GB synthetic
data. Correctness is checked per kernel call: every kernel call must satisfy the
error bound (max |original − decompressed| ≤ eb). The harness measures the original
SC'23 baseline once at startup (5 timed runs after 1 warmup) and records its
per-family time, which never changes during your session.

## Scoring metric (per-family arithmetic mean — Plan C)

The headline `speedup` reported every round is computed as follows. Let
`b_v` = baseline mean time for family v, `s_v` = your kernel time summed over the
9 tests of family v, `submitted_v` = whether you have ever rewritten that family's
file in this session, `correct_v` = whether ALL 9 tests of family v are within the
error bound this round. Define

    speedup_v = baseline_v / solution_v   if  submitted_v AND correct_v AND solution_v > 0
              = 0                          otherwise

    speedup   = mean(speedup_v) over the 6 families v ∈
                {1D_f32, 1D_f64, 2D_f32, 2D_f64, 3D_f32, 3D_f64}

This means:
  - Families you have NOT rewritten contribute 0, no matter how fast the baseline is.
  - Families with any failed test contribute 0.
  - The maximum achievable score requires correct kernels for ALL six families.
  - To reach speedup ≥ 1.0 you need a per-family geometric/arithmetic profile such
    that the average matches or beats baseline parity across the six.

Each round's feedback also prints a per-family breakdown (submitted? correct? time,
speedup) so you can see exactly which families are dragging the score down.

## Coverage matters (strict mode)

The 6 kernel files in `src/cuSZp_kernels_*.cu` are ALL pre-populated with `#error
"CUDA_HERCULES_PLACEHOLDER_..."` directives. The build will hard-fail with that
error message until you rewrite every file that needs to be measured. There is
NO fallback to a reference implementation — leaving a kernel file unwritten
means the entire `cuSZp_bench` binary cannot link, and the round's eval reports
`make failed` (no per-family speedup data is produced for that round).

Practical consequence: you must rewrite ALL six families before you can score
any non-zero speedup at all. Iterating endlessly on a single family is not just
suboptimal — it produces a build error every round and zero data.

## Phases

**Phase 0 (planning)** — The very first message asks for a free-text plan. Output plan
text only; no JSON actions, no code blocks. Describe your high-level strategy for
each of the 6 kernel families.

**Phase 1..N (implementation)** — Each round you respond with actions. The controller
executes them in order and replies with results.

## Action format (strict)

Every action MUST start with a JSON object on its own line:
`{"action":"<name>", ...}`. Code/text blocks without a preceding action header are
ignored.

## Available actions

1. `write_file` — overwrite one of the 6 kernel files with your implementation.
   ```
   {"action":"write_file","path":"src/cuSZp_kernels_1D_f32.cu"}
   ```cuda
   // full file contents (not a diff)
   ```
   Allowed paths:
     - src/cuSZp_kernels_1D_f32.cu
     - src/cuSZp_kernels_1D_f64.cu
     - src/cuSZp_kernels_2D_f32.cu
     - src/cuSZp_kernels_2D_f64.cu
     - src/cuSZp_kernels_3D_f32.cu
     - src/cuSZp_kernels_3D_f64.cu
   Same file at most once per round.

2. `read_file` — read the CURRENT contents of a file. Allowed:
   - Any kernel file you have already written in THIS session (returns your version).
   - Any of the readable infrastructure files listed below (always readable).
   - The original baseline contents of kernel files are NEVER returned.
   ```
   {"action":"read_file","path":"src/cuSZp_entry_1D_f32.cu"}
   ```

   Readable infrastructure files:
     - src/main.cu, src/cuSZp_timer.cu, src/CMakeLists.txt
     - src/cuSZp_entry_{1D,2D,3D}_{f32,f64}.cu (6 entry wrappers)
     - src/include/cuSZp/*.h (all 13 headers)

3. `profile` — run a profiler over the compiled cuSZp_bench binary.
   - `nsys`: timeline + kernel summary.
     ```
     {"action":"profile","tool":"nsys"}
     ```
   - `ncu`: per-kernel micro-metrics. Requires a regex matching kernel name (use nsys
     first to discover actual names; they look like `cuSZp_compress_kernel_1D_fixed_f32`).
     ```
     {"action":"profile","tool":"ncu","kernel":"cuSZp_compress_kernel_1D_fixed_f32"}
     ```
   At most 2 profile calls per round.

4. `update_plan` — revise your stored plan. Replaces the previous plan; shown to you
   every round.
   ```
   {"action":"update_plan"}
   ```text
   revised plan text
   ```

## Round rules

- Round 1: may use any action. If you don't write, no bench runs (we keep the baseline).
  You may use round 1 purely to read_file headers/entries and plan.
- Rounds 2+: any action combination; execute in the written order.
- If the round contains any `write_file`, the harness runs `make` (incremental) + one
  bench run and reports compile/correctness/kernel-time/speedup back to you.

## Hints

- Start by reading a header (`src/include/cuSZp/cuSZp_kernels_1D_f32.h`) to see the
  kernel signatures, then the entry wrapper (`src/cuSZp_entry_1D_f32.cu`) to see how
  they are launched. From the signatures you can see all `__global__` kernels use a
  flag-based asynchronous chunking scheme.
- You may choose to replace just one kernel family first (e.g. 1D f32) and iterate
  before touching the others — unwritten files keep the baseline implementation, so
  the build still succeeds.
"""


# ── Action parser (same format as tcgnn evaluator, extended to update_plan) ──

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

    if fallback_round1 and not actions:
        code, _ = _find_code_after(message, 0)
        if code.strip():
            actions.append({
                "type": "write_file",
                "path": "src/cuSZp_kernels_1D_f32.cu",
                "code": code,
                "raw_json": "(synthesized: no JSON action header, assuming write_file 1D_f32)",
            })
    return actions


# ── Session / sandbox ─────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_out_dir: Path):
        self.run_out_dir = run_out_dir
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="cuszp_toolaug_"))
        # LLM session work_dir: src/cuSZp_kernels_*.cu start as #error placeholders.
        # Any kernel file the LLM does not rewrite will fail to compile — there is
        # NO fallback to the reference implementation.
        self.work_dir = self.tmp_dir / "cuszp"
        # Separate baseline work_dir: reference/ files are overlaid into src/ so
        # the SC'23 baseline can be measured. Never exposed to the LLM.
        self.baseline_work_dir = self.tmp_dir / "cuszp_baseline"
        self.written_files: dict[str, str] = {}
        self.rounds: list[dict] = []
        self.current_plan: str = ""
        self.baseline_time_ms: float = -1.0
        self.baseline_per_family: dict[str, float] = {}
        self.best_speedup: float = -1.0
        self.best_round: int = -1

    def setup(self, source_task_dir: Path):
        """Provision two work_dirs: one for the LLM (placeholders in src/) and a
        private one for baseline measurement (reference/ overlaid into src/).
        """
        def _ignore(src, names):
            return [n for n in names if n in ("cmake-build-release", "__pycache__")]

        # 1. LLM session — drop the entire `reference/` tree; LLM never sees it
        def _ignore_with_reference(src, names):
            ignored = [n for n in names if n in ("cmake-build-release", "__pycache__")]
            if Path(src).resolve() == source_task_dir.resolve():
                ignored.append(REFERENCE_SUBDIR)
            return ignored

        self.work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_task_dir, self.work_dir,
                        dirs_exist_ok=True, ignore=_ignore_with_reference)

        # 2. Baseline work_dir — copy task INCLUDING reference/, then overlay
        #    reference/cuSZp_kernels_*.cu over src/cuSZp_kernels_*.cu so cmake
        #    builds the SC'23 reference instead of the placeholders.
        self.baseline_work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_task_dir, self.baseline_work_dir,
                        dirs_exist_ok=True, ignore=_ignore)
        ref_dir = self.baseline_work_dir / REFERENCE_SUBDIR
        if not ref_dir.is_dir():
            raise RuntimeError(
                f"reference dir missing: {ref_dir} — task layout broken; "
                f"reference/cuSZp_kernels_*.cu must be present for baseline measurement.")
        for f in SOLUTION_FILES:
            ref_file = ref_dir / Path(f).name
            dst_file = self.baseline_work_dir / f
            if not ref_file.is_file():
                raise RuntimeError(f"reference file missing: {ref_file}")
            shutil.copy2(ref_file, dst_file)
        # Sanity: drop reference/ from baseline work_dir post-overlay so the build
        # only sees src/, identical to the layout cmake expects.
        shutil.rmtree(ref_dir, ignore_errors=True)

    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def snapshot_round(self, round_idx: int) -> dict:
        rd = self.run_out_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        for path, content in self.written_files.items():
            target = rd / path.replace("/", "__")
            target.write_text(content)
        return {"dir": str(rd)}


# ── Build / bench / parsing helpers ──────────────────────────────────────────

_KERNEL_TIME_RE = re.compile(r"Kernel time:\s*([0-9.]+)\s*ms")
_MODE_RE = re.compile(
    r"(\w+):\s+cmp\s+([0-9.]+)\s+GB/s,\s+dec\s+([0-9.]+)\s+GB/s,\s+"
    r"ratio\s+([0-9.]+)x,\s+(PASS|FAIL)"
)
# Per-test KERNEL line emitted by src/main.cu, e.g.:
#   KERNEL 1D_f32 fixed eb=1E-2: correct=1 errors=0 max_error=... error_bound=...
#   err_ratio=... cmp_ms=12.34 dec_ms=5.67 ratio=5.96 cmp_gbps=... dec_gbps=... nbEle=...
_KERNEL_PER_TEST_RE = re.compile(
    r"KERNEL\s+(\S+)\s+\S+\s+eb=\S+:\s+correct=(\d+)\s+errors=\d+\s+"
    r"max_error=[0-9.eE+\-]+\s+error_bound=[0-9.eE+\-]+\s+err_ratio=[0-9.]+\s+"
    r"cmp_ms=([0-9.]+)\s+dec_ms=([0-9.]+)"
)


def parse_per_family(output: str) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Return (time_ms_sum, fail_count, test_count) per variant ('1D_f32' etc.)."""
    times = {v: 0.0 for v in VARIANTS}
    fails = {v: 0 for v in VARIANTS}
    counts = {v: 0 for v in VARIANTS}
    for m in _KERNEL_PER_TEST_RE.finditer(output):
        variant = m.group(1)
        if variant not in times:
            continue
        correct = int(m.group(2))
        cmp_ms = float(m.group(3))
        dec_ms = float(m.group(4))
        times[variant] += cmp_ms + dec_ms
        counts[variant] += 1
        if correct == 0:
            fails[variant] += 1
    return times, fails, counts


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


def _build_dir(work_dir: Path) -> Path:
    return work_dir / BUILD_SUBPATH


def _executable(work_dir: Path) -> Path:
    return _build_dir(work_dir) / EXECUTABLE_NAME


def run_cmake_configure(work_dir: Path) -> tuple[bool, str]:
    build = _build_dir(work_dir)
    build.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build, capture_output=True, text=True, timeout=BUILD_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, "cmake configure timed out"
    if proc.returncode != 0:
        return False, extract_compile_errors(proc.stderr + "\n" + proc.stdout, 3000)
    return True, ""


def run_make(work_dir: Path) -> tuple[bool, str]:
    """Incremental make; returns (ok, errors)."""
    build = _build_dir(work_dir)
    try:
        proc = subprocess.run(
            ["make", f"-j{os.cpu_count() or 4}"],
            cwd=build, capture_output=True, text=True, timeout=BUILD_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, "make timed out"
    if proc.returncode != 0:
        return False, extract_compile_errors(proc.stderr + "\n" + proc.stdout, 3000)
    return True, ""


def run_bench(work_dir: Path) -> tuple[dict, str, str]:
    """Run cuSZp_bench once, return (parsed result dict, stdout, stderr)."""
    exe = _executable(work_dir)
    if not exe.is_file():
        return {"ok": False, "error": f"executable not built at {exe}"}, "", ""
    try:
        proc = subprocess.run(
            [str(exe)], cwd=work_dir,
            capture_output=True, text=True, timeout=BENCH_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "bench timed out"}, "", ""
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    combined = stdout + stderr
    passed = ("Passed" in combined) and (proc.returncode == 0)
    m = _KERNEL_TIME_RE.search(combined)
    kernel_time_ms = float(m.group(1)) if m else -1.0
    modes = _MODE_RE.findall(combined)
    n_fail = sum(1 for mode in modes if mode[4] == "FAIL")
    pf_time, pf_fails, pf_counts = parse_per_family(combined)
    return {
        "ok": passed,
        "kernel_time_ms": kernel_time_ms,
        "mode_count": len(modes),
        "mode_fail": n_fail,
        "exit_code": proc.returncode,
        "per_family_time_ms": pf_time,
        "per_family_fails": pf_fails,
        "per_family_counts": pf_counts,
    }, stdout, stderr


def measure_baseline(session: Session, verbose: bool = True) -> tuple[float, dict[str, float]]:
    """Build and bench the SC'23 reference in `session.baseline_work_dir`.

    The reference impls were overlaid into baseline_work_dir/src/ during setup —
    they do NOT live in the LLM session work_dir. Returns
    (mean_total_kernel_time_ms, per_family_baseline_ms).
    """
    bl_dir = session.baseline_work_dir
    if verbose:
        print(f"[baseline] cmake configure + full make in {bl_dir} ...", flush=True)
    ok, err = run_cmake_configure(bl_dir)
    if not ok:
        raise RuntimeError(f"baseline cmake configure failed:\n{err}")
    ok, err = run_make(bl_dir)
    if not ok:
        raise RuntimeError(f"baseline initial make failed:\n{err}")

    if verbose:
        print(f"[baseline] warmup bench ({BASELINE_WARMUP} run)...", flush=True)
    for _ in range(BASELINE_WARMUP):
        run_bench(bl_dir)

    times: list[float] = []
    pf_acc: dict[str, list[float]] = {v: [] for v in VARIANTS}
    for i in range(BASELINE_RUNS):
        r, _stdout, _stderr = run_bench(bl_dir)
        if not r.get("ok"):
            raise RuntimeError(f"baseline bench run {i+1} failed: {r}")
        if r["kernel_time_ms"] > 0:
            times.append(r["kernel_time_ms"])
            if verbose:
                print(f"[baseline] run {i+1}/{BASELINE_RUNS}: kernel_time={r['kernel_time_ms']:.2f} ms", flush=True)
        for v, t in r.get("per_family_time_ms", {}).items():
            if t > 0:
                pf_acc[v].append(t)
    if not times:
        raise RuntimeError("no valid baseline runs")
    pf_baseline = {v: mean(ts) for v, ts in pf_acc.items() if ts}
    if verbose:
        bl_table = "  ".join(f"{v}={pf_baseline.get(v, 0.0):.1f}ms" for v in VARIANTS)
        print(f"[baseline] per-family: {bl_table}", flush=True)
    return mean(times), pf_baseline


# ── Action executors ──────────────────────────────────────────────────────────

def do_write_file(session: Session, action: dict) -> str:
    path = action["path"]
    if path not in SOLUTION_FILES:
        return f"ERROR: path '{path}' not in writable set {SOLUTION_FILES}."
    code = action["code"]
    if not code.strip():
        return f"ERROR: empty code block for {path}."
    target = session.work_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)
    session.written_files[path] = code
    return f"OK: wrote {path} ({len(code)} bytes)."


def do_read_file(session: Session, action: dict) -> str:
    path = action["path"]
    # Priority 1: model has written this file in the current session → show its version
    if path in session.written_files:
        content = session.written_files[path]
        return f"CONTENT of {path} (your current version, {len(content)} bytes):\n```\n{content}\n```"
    # Priority 2: infrastructure file — read from work_dir
    if path in READABLE_INFRA_FILES:
        fs_path = session.work_dir / path
        if not fs_path.is_file():
            return f"ERROR: infra file '{path}' missing on disk (unexpected)."
        content = fs_path.read_text()
        return f"CONTENT of {path} (infra, {len(content)} bytes):\n```\n{content}\n```"
    # Priority 3: writable kernel file, not yet written by model → baseline hidden
    if path in SOLUTION_FILES:
        return (f"ERROR: '{path}' is a baseline kernel file you have not yet rewritten. "
                f"Baseline contents are hidden. Write your own implementation first.")
    return (f"ERROR: '{path}' is not in the readable whitelist. "
            f"Writable: {SOLUTION_FILES}. "
            f"Readable infra: {sorted(READABLE_INFRA_FILES)}.")


def do_update_plan(session: Session, action: dict) -> str:
    plan = action.get("plan", "").strip()
    if not plan:
        return "ERROR: update_plan requires non-empty plan text."
    session.current_plan = plan
    return f"OK: plan updated ({len(plan)} chars)."


def do_profile(session: Session, action: dict) -> str:
    tool = action.get("tool", "")
    exe = _executable(session)
    if not exe.is_file():
        return f"ERROR: binary not built yet at {exe}. Write code first so the harness rebuilds."
    env = os.environ.copy()
    env["TMPDIR"] = str(session.tmp_dir)
    ts = int(time.time() * 1000)
    try:
        if tool == "nsys":
            rep = session.tmp_dir / f"nsys_{ts}.nsys-rep"
            cmd = ["nsys", "profile", "-t", "cuda", "--stats=true",
                   "--force-overwrite", "true", "-o", str(rep), str(exe)]
        elif tool == "ncu":
            kregex = action.get("kernel", "").strip()
            if not kregex:
                return "ERROR: ncu requires a 'kernel' regex. Use nsys first to see actual kernel names."
            cmd = ["ncu", "--set", "basic", "--launch-count", "5",
                   "--kernel-name", f"regex:{kregex}", str(exe)]
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
        return f"{tool.upper()} PROFILE ({status}):\n" + _truncate(combined)
    except subprocess.TimeoutExpired:
        return f"ERROR: {tool} timed out after {PROFILE_TIMEOUT_SEC}s."
    except FileNotFoundError as e:
        return f"ERROR: {tool} not installed on this machine ({e})."


# ── Per-round evaluation ──────────────────────────────────────────────────────

def _compute_per_family(session: Session, bench: dict) -> tuple[dict, float]:
    """Plan C: per-family speedup, arithmetic mean over 6 families with missing→0.

    Per-family rules:
      - Family file not in session.written_files          → speedup = 0 (not_submitted)
      - Family has any failed test                        → speedup = 0 (incorrect)
      - Family has no parsed timing or no baseline timing → speedup = 0 (no_timing)
      - Otherwise                                          → baseline_t / solution_t
    """
    pf_time = bench.get("per_family_time_ms", {}) or {}
    pf_fails = bench.get("per_family_fails", {}) or {}
    pf_counts = bench.get("per_family_counts", {}) or {}
    written_variants = {FILE_TO_VARIANT[f] for f in session.written_files
                        if f in FILE_TO_VARIANT}
    per_family: dict[str, dict] = {}
    for v in VARIANTS:
        submitted = v in written_variants
        family_time = float(pf_time.get(v, 0.0))
        family_fails = int(pf_fails.get(v, 0))
        family_count = int(pf_counts.get(v, 0))
        baseline_v = float(session.baseline_per_family.get(v, 0.0))
        if not submitted:
            speedup_v, family_correct, reason = 0.0, None, "not_submitted"
        elif family_count == 0 or family_time <= 0 or baseline_v <= 0:
            speedup_v, family_correct, reason = 0.0, False, "no_timing"
        elif family_fails > 0:
            speedup_v, family_correct, reason = 0.0, False, "incorrect"
        else:
            speedup_v = baseline_v / family_time
            family_correct, reason = True, "ok"
        per_family[v] = {
            "submitted": submitted,
            "correct": family_correct,
            "tests_passed": max(0, family_count - family_fails),
            "tests_total": family_count,
            "time_ms": family_time,
            "baseline_ms": baseline_v,
            "speedup": speedup_v,
            "reason": reason,
        }
    mean_speedup = sum(p["speedup"] for p in per_family.values()) / len(VARIANTS)
    return per_family, mean_speedup


def run_eval(session: Session) -> dict:
    """Build & bench the LLM's session work_dir.

    Any kernel file the LLM has not yet rewritten still contains the
    `#error CUDA_HERCULES_PLACEHOLDER_*` directive — `make` will hard-fail
    until the candidate provides an implementation for every solution_file
    they intend to score on (Plan-C scoring then aggregates per-family).
    """
    t0 = time.time()
    ok, err = run_make(session.work_dir)
    t_make = time.time() - t0
    if not ok:
        return {
            "compiled": False, "correct": False,
            "kernel_time_ms": -1.0, "speedup": -1.0,
            "speedup_total": -1.0,
            "per_family": {},
            "stderr_errors": err, "elapsed_s": t_make,
            "error": "make failed",
        }
    t_b = time.time()
    bench, stdout, stderr = run_bench(session.work_dir)
    t_bench = time.time() - t_b
    elapsed = time.time() - t0
    kernel_time = bench.get("kernel_time_ms", -1.0)
    speedup_total = (session.baseline_time_ms / kernel_time
                     if session.baseline_time_ms > 0 and kernel_time > 0 else -1.0)
    per_family, speedup = _compute_per_family(session, bench)
    exit_code = bench.get("exit_code", 0)

    # Summarize crash signal for the feedback when the benchmark exited abnormally
    # and produced little or no output (otherwise the model sees an empty message).
    bench_crash_info = ""
    if exit_code is not None and exit_code != 0:
        if exit_code < 0:
            sig = -exit_code
            sig_name_map = {6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL",
                            11: "SIGSEGV", 15: "SIGTERM"}
            sig_name = sig_name_map.get(sig, f"signal {sig}")
            bench_crash_info = f"bench killed by {sig_name} (exit={exit_code})"
        else:
            bench_crash_info = f"bench exited with code {exit_code}"
    return {
        "compiled": True,
        "correct": bool(bench.get("ok")),
        "kernel_time_ms": kernel_time,
        "baseline_time_ms": session.baseline_time_ms,
        # `speedup` is the headline metric: arithmetic mean of per-family speedup,
        # with not-submitted / incorrect families scored 0 (Plan C).
        "speedup": speedup,
        # Legacy aggregate (baseline_total / solution_total). Kept for diagnosis only.
        "speedup_total": speedup_total,
        "per_family": per_family,
        "mode_count": bench.get("mode_count", 0),
        "mode_fail": bench.get("mode_fail", 0),
        "exit_code": exit_code,
        "bench_crash_info": bench_crash_info,
        "stdout_tail": _truncate(stdout, 1500),
        "stderr_errors": extract_compile_errors(stderr, 2000) if stderr.strip() else "",
        "stderr_tail": _truncate(stderr, 1500) if stderr.strip() else "",
        "make_s": t_make,
        "bench_s": t_bench,
        "elapsed_s": elapsed,
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

def build_planning_message(task_description: str, baseline_time_ms: float,
                           per_family_baseline: dict[str, float] | None = None) -> str:
    pf_block = ""
    if per_family_baseline:
        pf_block = "Per-family baseline (sum cmp+dec across 9 tests/family):\n" + "\n".join(
            f"  {v}: {per_family_baseline.get(v, 0.0):.2f} ms" for v in VARIANTS) + "\n\n"
    return (f"{task_description}\n\n"
            "=== PLANNING PHASE ===\n"
            f"Baseline measured: {baseline_time_ms:.2f} ms total kernel time "
            f"(mean of {BASELINE_RUNS} runs after {BASELINE_WARMUP} warmup).\n"
            f"{pf_block}"
            "SCORING (Plan C, per-family arithmetic mean): each round's reported speedup\n"
            "is mean(speedup_v) over the 6 families, where speedup_v = baseline_v /\n"
            "solution_v if you have rewritten family v AND every test of family v passes,\n"
            "else 0. Families you never touch are 0 — there is NO 'fall back to baseline'\n"
            "credit. Maximum score requires all 6 families correct; touching only 1 caps\n"
            "your score at ≤ 1/6.\n\n"
            "Your plan MUST explicitly address all SIX kernel families. The 6 families are:\n"
            "  - src/cuSZp_kernels_1D_f32.cu\n"
            "  - src/cuSZp_kernels_1D_f64.cu\n"
            "  - src/cuSZp_kernels_2D_f32.cu\n"
            "  - src/cuSZp_kernels_2D_f64.cu\n"
            "  - src/cuSZp_kernels_3D_f32.cu\n"
            "  - src/cuSZp_kernels_3D_f64.cu\n\n"
            "Produce a concise plan covering:\n"
            "  1. Overall algorithmic approach — what strategy per (dim, precision)? Note\n"
            "     that 1D→2D→3D mostly differ in block/tile decomposition of the input,\n"
            "     and f32↔f64 share structure but differ in quantization precision and\n"
            "     mantissa widths.\n"
            "  2. Per-family intent — ONE bullet PER file listing what kind of kernel you\n"
            "     plan to implement there (can share code patterns across families).\n"
            "  3. Execution order — which family first, which next, and a rough round\n"
            "     budget so all 6 are covered within the 30-round session.\n"
            "  4. What information to read first (headers, entry wrappers).\n"
            "  5. Correctness plan — error-bound-respecting quantization/reconstruction.\n\n"
            "Output ONLY plan text — no JSON actions, no code blocks. The controller\n"
            "stores the plan and re-shows it every round. Revise later with `update_plan`.")


def build_user_message_round1(session: Session, task_description: str, total_rounds: int) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    coverage_block = _coverage_block(session, 1, total_rounds) + "\n\n"
    bl = f"Baseline (total): {session.baseline_time_ms:.2f} ms.\n"
    return (f"{plan_block}{coverage_block}{bl}"
            "=== ROUND 1 / IMPLEMENTATION PHASE ===\n"
            "You may do any combination of actions this round. If you don't yet understand\n"
            "the interface, begin by reading `src/include/cuSZp/cuSZp_kernels_1D_f32.h`\n"
            "(kernel signatures) and `src/cuSZp_entry_1D_f32.cu` (how kernels are invoked).\n"
            "Strict-mode reminder: every kernel file in src/cuSZp_kernels_*.cu currently\n"
            "contains a `#error CUDA_HERCULES_PLACEHOLDER_*` directive. The build will\n"
            "hard-fail until you rewrite ALL SIX. There is no fallback to a reference\n"
            "implementation. You will not see a non-zero speedup for ANY family until\n"
            "every one of the six kernel files compiles.")


def _coverage_block(session: Session, round_idx: int, total_rounds: int) -> str:
    covered = set(session.written_files.keys())
    lines = ["=== COVERAGE ==="]
    for f in SOLUTION_FILES:
        mark = "✓ your version" if f in covered else "✗ #error PLACEHOLDER (build will fail)"
        lines.append(f"  {f}: {mark}")
    lines.append(f"Covered {len(covered)}/{len(SOLUTION_FILES)} families. "
                 f"Round {round_idx}/{total_rounds} ({total_rounds - round_idx} left). "
                 f"Build only succeeds once ALL 6 are rewritten.")
    return "\n".join(lines)


def build_feedback(session: Session, round_idx: int, total_rounds: int,
                   action_outputs: list[str], eval_result: dict | None) -> str:
    plan_block = (f"=== YOUR PLAN ===\n{session.current_plan}\n\n"
                  if session.current_plan else "")
    coverage_block = _coverage_block(session, round_idx, total_rounds) + "\n\n"
    parts = [f"{plan_block}{coverage_block}=== ROUND {round_idx} / {total_rounds} RESULTS ==="]
    for i, out in enumerate(action_outputs, 1):
        parts.append(f"\n--- action {i} output ---\n{out}")
    if eval_result is not None:
        parts.append("\n--- evaluation (make + cuSZp_bench) ---")
        sp = eval_result.get("speedup", -1.0)
        sp_total = eval_result.get("speedup_total", -1.0)
        kt = eval_result.get("kernel_time_ms", -1.0)
        baseline_time = eval_result.get("baseline_time_ms", session.baseline_time_ms)
        sp_str = f"{sp:.4f}x" if sp > 0 else "0.0000x"
        sp_total_str = f"{sp_total:.4f}x" if sp_total > 0 else "N/A"
        parts.append(
            f"compiled={eval_result.get('compiled')}  correct={eval_result.get('correct')}  "
            f"kernel_time={kt:.2f}ms  baseline={baseline_time:.2f}ms")
        parts.append(
            f"SCORE (Plan C, mean of per-family speedup, missing/incorrect=0): {sp_str}  "
            f"[diagnostic-only legacy total-time speedup: {sp_total_str}]")
        pf = eval_result.get("per_family") or {}
        if pf:
            parts.append("per-family breakdown:")
            for v in VARIANTS:
                e = pf.get(v, {})
                fname = VARIANT_TO_FILE[v]
                tag = e.get("reason", "n/a")
                tt = e.get("tests_total", 0)
                tp = e.get("tests_passed", 0)
                t_ms = e.get("time_ms", 0.0)
                bl_ms = e.get("baseline_ms", 0.0)
                spv = e.get("speedup", 0.0)
                parts.append(
                    f"  {v} ({fname}): {tag:14s}  tests {tp}/{tt}  "
                    f"time={t_ms:7.2f}ms  baseline={bl_ms:7.2f}ms  speedup={spv:.4f}x")
        if eval_result.get("mode_count"):
            parts.append(f"mode_count={eval_result['mode_count']}  mode_fail={eval_result['mode_fail']}")
        if eval_result.get("error"):
            parts.append(f"error: {eval_result['error']}")
        if eval_result.get("bench_crash_info"):
            parts.append(f"RUNTIME CRASH: {eval_result['bench_crash_info']}")
        if eval_result.get("stderr_errors"):
            parts.append(f"\ncompile/runtime errors (extracted):\n{eval_result['stderr_errors']}")
        stdout_tail = eval_result.get('stdout_tail', '')
        if stdout_tail.strip():
            parts.append(f"\nstdout tail:\n{stdout_tail[:1500]}")
        stderr_tail = eval_result.get('stderr_tail', '')
        if stderr_tail.strip() and not eval_result.get('stderr_errors'):
            parts.append(f"\nstderr tail:\n{stderr_tail[:1200]}")
        # If bench produced nothing at all (common with silent segfaults), tell the model explicitly
        if (not stdout_tail.strip() and not stderr_tail.strip()
                and eval_result.get('kernel_time_ms', 0) <= 0
                and eval_result.get('compiled')):
            parts.append("\n(bench produced no output — solution likely crashed before "
                         "printing; check for null pointer deref, out-of-bounds access, "
                         "incorrect launch bounds, or missing __syncthreads.)")
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
    print("[setup] measuring baseline (private reference work_dir)...", flush=True)
    session.baseline_time_ms, session.baseline_per_family = measure_baseline(session, verbose=True)
    print(f"[setup] baseline = {session.baseline_time_ms:.2f} ms "
          f"({time.time() - t_bl:.1f}s total)", flush=True)

    # cmake configure the LLM session work_dir up front. It contains six #error
    # placeholders for cuSZp_kernels_*.cu — cmake configure (Makefile generation)
    # succeeds, but `make` will fail with a CUDA_HERCULES_PLACEHOLDER_* #error
    # until the LLM rewrites every kernel file the build needs.
    print(f"[setup] cmake configure LLM work_dir {session.work_dir}", flush=True)
    ok, err = run_cmake_configure(session.work_dir)
    if not ok:
        raise RuntimeError(f"LLM work_dir cmake configure failed:\n{err}")

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
        planning_msg = build_planning_message(task_description, session.baseline_time_ms,
                                              session.baseline_per_family)
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

            # Plan C: best_speedup tracks the per-family arithmetic-mean speedup.
            # Per-family correctness is already folded into the metric (incorrect → 0),
            # so we don't gate on the bench-wide `correct` flag (which would reject any
            # round where a single submitted family fails, even if others are valid wins).
            if eval_result and eval_result.get("speedup", -1) > 0:
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
        "scoring": {
            "metric": "plan_c_per_family_arithmetic_mean",
            "description": ("speedup = mean over 6 families of (baseline_v / solution_v) "
                            "if family v was submitted AND all 9 of its tests pass, else 0"),
            "variants": VARIANTS,
        },
        "baseline_time_ms": session.baseline_time_ms,
        "baseline_per_family_ms": session.baseline_per_family,
        "best_speedup": session.best_speedup, "best_round": session.best_round,
        "elapsed_s": elapsed, "rounds": session.rounds,
    }
    (out_dir / "final_report.json").write_text(json.dumps(final, indent=2))
    if session.best_round > 0:
        best_dir = out_dir / f"round_{session.best_round:02d}"
        for fname in SOLUTION_FILES:
            src = best_dir / fname.replace("/", "__")
            if src.exists():
                shutil.copy2(src, out_dir / f"best_{os.path.basename(fname)}")

    print(f"\n=== DONE ({elapsed:.1f}s) ===")
    print(f"baseline = {session.baseline_time_ms:.2f} ms")
    print(f"best_speedup = {session.best_speedup:.4f}x (round {session.best_round})")
    print(f"results in: {out_dir}")

    session.cleanup()


if __name__ == "__main__":
    main()
