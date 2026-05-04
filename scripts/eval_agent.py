#!/usr/bin/env python3
"""
CUDA-Hercules Tool-Augmented Evaluation.

Fixed controller loop:
  - The script deterministically handles prompt building, solution writing,
    compile/test, scoring, resume, and reporting.
  - The model may optionally request profiling of the current passing solution.
  - Profiling is the only interactive "tool"; everything else is workflow.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_agent.py \
        --model gemma-4-31B-it --provider openai \
        --api-base http://localhost:8004/v1 \
        --filter backend=class1_make --filter arch=general \
        --max-iterations 10 \
        --task-timeout 900 \
        --run-name gemma_toolaug_c1gen
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.backend_class1 import (
    UNSUPPORTED_ARCH_PREFIX,
    _current_arch_make_vars,
    _format_make_cmd,
    _is_class1_general_task,
)
from cuda_hercules.eval import (
    _current_arch_gencode_flags,
    _get_device_sm,
    _is_class2_general_task,
    get_solution_include_dirs,
    load_task_def,
)
from cuda_hercules.compiler import compile_cuda_module
from cuda_hercules.llm_api import query_server
from cuda_hercules.prompt_builder import build_prompt
from cuda_hercules.result import TaskResult, TaskStatus
from cuda_hercules.runner import discover_tasks, get_gpu_sm, parse_filters, run_task
from cuda_hercules.score import compute_scores, format_score
from cuda_hercules.static_checker import validate_cuda_solution
from cuda_hercules.task_schema import TaskConfig
from cuda_hercules.utils import extract_cuda_code, get_project_root

MAX_PROFILE_OUTPUT = 4000
# Per-round profile budget. Model may chain nsys → ncu → ncu (different
# metrics/sections) before writing code. Keep a reasonable cap so a buggy
# prompt loop can't burn the whole round on profiling alone.
MAX_PROFILE_CALLS_PER_ROUND = 8

VALID_PROFILE_TOOLS = {"nsys", "ncu"}


def _build_system_prompt(allowed_tools: set[str]) -> str:
    """Return SYSTEM_PROMPT, optionally annotated when tools are restricted.

    The base prompt documents both nsys and ncu so the model knows what each
    one does. When a subset is configured for this run, we append a short
    constraint so the model doesn't waste rounds requesting an unavailable
    tool.
    """
    if allowed_tools == VALID_PROFILE_TOOLS:
        return SYSTEM_PROMPT
    missing = sorted(VALID_PROFILE_TOOLS - allowed_tools)
    allowed = sorted(allowed_tools)
    if not allowed:
        restriction = (
            "\n\n## Environment Restriction\n\n"
            "No profilers are available in this evaluation run. Do NOT emit "
            "any `{\"action\":\"profile\", ...}` request — they will be rejected."
        )
    else:
        restriction = (
            "\n\n## Environment Restriction\n\n"
            f"Only the following profiler(s) are available in this evaluation "
            f"run: {', '.join('`' + t + '`' for t in allowed)}. "
            f"Do NOT request {', '.join('`' + t + '`' for t in missing)} — it "
            f"is not installed and the request will be rejected."
        )
    return SYSTEM_PROMPT + restriction

SYSTEM_PROMPT = """You are an expert CUDA kernel engineer operating inside a fixed evaluation controller.

## Profiling Tools Available

You have access to TWO NVIDIA profilers. Both run your current passing
solution on real GPU hardware and return real metrics. You decide which one
to use — you may use just one, or call them in sequence (a common workflow
is `nsys` first for an overview, then `ncu` on the hottest kernel for
micro-level counters).

- **nsys** (Nsight Systems): wall-clock timeline of kernel launches, memcpy,
  and CUDA API calls. Best for: which kernel dominates runtime, how many
  launches happen, are there gaps between kernels, is the workload
  compute-bound at the API level.

- **ncu** (Nsight Compute): per-kernel SM occupancy, memory throughput, SOL
  (speed-of-light) utilization, register/shared-memory usage, stall reasons,
  bank conflicts, warp cycles. Best for: why is a specific kernel slow, is
  it memory-bound or compute-bound, is occupancy limited by register or
  shared memory, are there bank conflicts.

## Action Set (one per round)

1. Return a full replacement solution file.
2. Request profiling of the current passing solution (nsys or ncu).
3. Submit and stop.

## Response Format

To write code:
```json
{"action":"write_code"}
```
```cuda
// complete solution file here
```

To profile with nsys (kernel timing overview):
```json
{"action":"profile","tool":"nsys"}
```

To profile with ncu (per-kernel micro metrics). The kernel_name is a regex
matched against the DEMANGLED kernel name — use a substring of your kernel's
function name, e.g. `"my_gemm_kernel"`:
```json
{"action":"profile","tool":"ncu","kernel_name":"your_kernel_name_substring"}
```

To stop:
```json
{"action":"submit"}
```

## Optional Profile Customization

You can narrow or expand what the profiler reports by adding optional fields
to the action JSON:

- `"set"` (ncu only, default `"detailed"`): one of `basic`, `default`,
  `detailed`, `full`, `roofline`, `source`. Heavier sets replay more kernels.
- `"sections"` (ncu only): extra `--section` values, e.g.
  `["MemoryWorkloadAnalysis", "WarpStateStats"]`.
- `"metrics"` (ncu only): custom metric regex fragments, e.g.
  `["sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard"]`.
- `"launch_count"` (ncu only, default 10, clamped to 1..50): how many kernel
  launches to replay.
- `"trace"` (nsys only, default `"cuda,nvtx"` for class2 / default for
  class1): e.g. `"cuda,cublas,cudnn,osrt,nvtx"`.

Omit any field to accept defaults.

## Rules
- If you choose write_code, return the complete file, not a diff.
- Prefer correctness first — you won't get a speedup score until correctness
  passes. But profile is available any time there is compilable code.
- Profile requires SOME code to have compiled at least once. If nothing has
  compiled yet (e.g. first round), you must write_code first.
- You may call profile multiple times in a row — e.g. `nsys` → read result
  → `ncu` on the hottest kernel → `ncu` again with different `sections` —
  before returning a new write_code. A per-round profile budget caps chains
  at {max_profile_calls_per_round} calls.
- Do not add explanations outside the required blocks.
""".replace("{max_profile_calls_per_round}", str(MAX_PROFILE_CALLS_PER_ROUND))

FIX_TEMPLATE = """Your previous CUDA solution failed. Here is the latest feedback:

## Error
{error}

{correctness_diff}

## Instructions
Refer to the original task description earlier in this conversation.

You have two options:

1. **Return a fixed complete solution** (most common choice when the error is a
   clear code bug):

   ```json
   {{"action":"write_code"}}
   ```
   ```cuda
   // complete solution file
   ```

2. **Profile the latest code that compiled** (useful when the error is a
   correctness mismatch — e.g. shape or numeric diff — and you want ncu /
   nsys to tell you what your kernel is actually doing on the GPU before
   you rewrite it). This works as long as at least one earlier round
   compiled, even if correctness did not pass:

   ```json
   {{"action":"profile","tool":"nsys"}}
   ```
   or
   ```json
   {{"action":"profile","tool":"ncu","kernel_name":"your_kernel_name_substring"}}
   ```

If the failure is a plain compile error (e.g. undefined symbol, bad
template), go straight to option 1 — profiling cannot help until the code
compiles.
"""

OPTIMIZE_TEMPLATE = """Your previous CUDA solution compiled and passed correctness.

## Current Performance
- Overall speedup vs reference: {speedup:.2f}x
{per_size_detail}
{perf_detail}

## Strongly Recommended: Profile Before Rewriting

You have two profilers available — decide which one (or both, in sequence)
fits your current question:

- `nsys` — kernel-level timeline: which kernel takes the most time, how many
  launches, any launch-gap overhead. Use this first if you don't yet know
  which kernel to target.
- `ncu` — per-kernel counters: SM occupancy, memory throughput, compute-vs-
  memory SOL %, stall reasons, bank conflicts, register / shared-memory
  pressure. Use this to diagnose WHY a specific kernel is slow.

A common and effective pattern is: run `nsys` first → identify the hot
kernel → run `ncu` on that kernel → then rewrite. You can also skip
straight to `ncu` if you already know which kernel to inspect.

Guessing at optimizations without profile data frequently breaks correctness
or yields no speedup. **Unless you already have strong profile evidence,
profile first.**

## Available Actions (pick exactly one)

Now that you have a passing solution, **include a short `"reason"` field**
(one sentence, ≤ 30 words) in the action JSON explaining why you chose this
action. The reason is logged for later analysis.

### 1. Profile with nsys (timeline / kernel breakdown)
```json
{{"action":"profile","tool":"nsys","reason":"<why profiling now>"}}
```

### 2. Profile with ncu (per-kernel micro metrics)
`kernel_name` is a regex matched against the DEMANGLED kernel symbol — use a
substring of your kernel's function name.
```json
{{"action":"profile","tool":"ncu","kernel_name":"your_kernel_name_substring","reason":"<what you hope to learn>"}}
```

### 3. Write a full improved solution (only after you know what to fix)
```json
{{"action":"write_code","reason":"<what you're changing and why, with or without profile data>"}}
```
followed by a complete ```cuda``` code block.

### 4. Submit (stop if further gains are unlikely)
```json
{{"action":"submit","reason":"<why no further gain is likely>"}}
```
"""

PROFILE_FOLLOWUP_TEMPLATE = """Profiling result for the current passing solution:

## Profiling
{profile_output}

## Instructions
Based on this profiling result, either:
- request one more profiling run,
- return a full improved solution, or
- submit.

Include a short `"reason"` field (≤ 30 words) in your action JSON explaining
your choice — especially if you skip further profiling and go straight to
code changes. Use the same response format rules as before.
"""

MALFORMED_RESPONSE_TEMPLATE = """Your previous response did not follow the required format.

Return exactly one of:
- a ```json``` block with {{"action":"submit"}}
- a ```json``` block with {{"action":"profile","tool":"nsys"}} or {{"action":"profile","tool":"ncu"}}
- a ```json``` block with {{"action":"write_code"}} followed by a complete ```cuda``` code block
"""


def _round_timeout_trigger(pid: int, flag: dict) -> None:
    """Interrupt a stuck round via SIGINT."""
    flag["timed_out"] = True
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass


def _normalize_model_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _query_model(
    messages: list[dict],
    provider: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    vertex_region: str,
    vertex_project: str,
    is_reasoning_model: bool = False,
    reasoning_effort: str = "",
) -> str:
    if provider == "openai":
        return query_server(
            prompt=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            backend="openai",
            is_reasoning_model=is_reasoning_model,
            reasoning_effort=reasoning_effort,
        )
    if provider == "anthropic_vertex":
        return query_server(
            prompt=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            backend="vertex",
            vertex_region=vertex_region,
            vertex_project=vertex_project,
            is_reasoning_model=is_reasoning_model,
            reasoning_effort=reasoning_effort,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or None,
        )
        response = llm.invoke(lc_messages)
        return _normalize_model_text(response.content)
    raise ValueError(f"Unknown provider: {provider}")


def _extract_first_json_object(text: str) -> dict | None:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))

    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_model_response(text: str) -> dict:
    parsed = _extract_first_json_object(text) or {}
    action = parsed.get("action", "").strip().lower()
    code = extract_cuda_code(text) if text else ""

    if not action and code:
        action = "write_code"

    if action not in {"write_code", "profile", "submit"}:
        action = "invalid"

    reason = str(parsed.get("reason", "")).strip()
    # Trim to single line / cap length for logging
    reason = " ".join(reason.split())[:300]

    def _as_list(value) -> list[str]:
        """Accept either a list or a comma/space-separated string."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value or "").strip()
        if not text:
            return []
        # Split on comma or whitespace
        return [part.strip() for part in re.split(r"[,\s]+", text) if part.strip()]

    return {
        "action": action,
        "tool": str(parsed.get("tool", "")).strip().lower(),
        "kernel_name": str(parsed.get("kernel_name", "")).strip(),
        "reason": reason,
        "code": code,
        # Optional profile customization. Model may omit any of these.
        "set": str(parsed.get("set", "")).strip(),            # ncu: --set (basic|default|detailed|full|roofline|source)
        "sections": _as_list(parsed.get("sections", "")),      # ncu: --section repeated
        "metrics": _as_list(parsed.get("metrics", "")),        # ncu: --metrics (comma-joined)
        "trace": str(parsed.get("trace", "")).strip(),        # nsys: -t value (e.g. "cuda,nvtx,cublas")
        "launch_count": parsed.get("launch_count"),            # ncu: override --launch-count
    }


def _build_messages(
    task_prompt: str,
    previous_code: str,
    latest_feedback: str,
    allowed_tools: Optional[set[str]] = None,
) -> list[dict]:
    system_prompt = _build_system_prompt(allowed_tools or VALID_PROFILE_TOOLS)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    if previous_code and latest_feedback:
        messages.append({"role": "assistant", "content": f"```cuda\n{previous_code}\n```"})
        messages.append({"role": "user", "content": latest_feedback})
    return messages


def _current_gpu_text() -> str:
    gpu_sm = get_gpu_sm()
    if gpu_sm <= 0:
        return ""
    try:
        import torch

        return f"\n\nTarget GPU: {torch.cuda.get_device_name()} (SM {gpu_sm})"
    except Exception:
        return f"\n\nTarget GPU SM: {gpu_sm}"


def _profile_state_init(config: TaskConfig) -> dict:
    root = get_project_root()
    task_dir = os.path.join(root, config.runner.workdir) if config.runner.workdir else ""

    tmp_dir = tempfile.mkdtemp(prefix="cuda_toolaug_")
    backend = config.runner.backend

    if backend == "class1_make":
        # Class 1: copy task dir, solution is a header replaced in place
        shutil.copytree(
            task_dir,
            os.path.join(tmp_dir, "task"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("ref", "test", "*.o"),
        )
        sol_file = config.runner.solution_file or "solution.h"
        return {
            "backend": backend,
            "tmp_dir": tmp_dir,
            "work_dir": os.path.join(tmp_dir, "task"),
            "sol_file": sol_file,
            "sol_path": os.path.join(tmp_dir, "task", sol_file),
            "task_dir_abs": task_dir,
        }

    # Class 2: solution is a standalone .cu compiled via torch cpp_extension.
    # No need to clone the task dir — def.py is read in place.
    sol_file = config.runner.solution_file or "solution.cu"
    return {
        "backend": backend,
        "tmp_dir": tmp_dir,
        "work_dir": tmp_dir,
        "sol_file": sol_file,
        "sol_path": os.path.join(tmp_dir, sol_file),
        "task_dir_abs": task_dir,
    }


def _profile_state_cleanup(state: dict) -> None:
    tmp_dir = state.get("tmp_dir", "")
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _profile_write_solution(config: TaskConfig, state: dict, code: str) -> tuple[bool, str]:
    check = validate_cuda_solution(
        code,
        blocked_patterns=config.anti_cheat.blocked_patterns or [],
        required_patterns=config.anti_cheat.required_patterns or [],
    )
    if not check.valid:
        return False, "STATIC CHECK FAILED:\n" + "\n".join(check.errors)

    with open(state["sol_path"], "w") as f:
        f.write(code)
    return True, ""


def _profile_compile(config: TaskConfig, state: dict) -> tuple[bool, str]:
    backend = config.runner.backend
    if backend == "class1_make":
        return _profile_compile_class1(config, state)
    if backend == "class2_defpy":
        return _profile_compile_class2(config, state)
    return False, f"PROFILING UNSUPPORTED: backend {backend} not supported."


def _profile_compile_class2(config: TaskConfig, state: dict) -> tuple[bool, str]:
    """Sanity-compile the class2 solution via torch cpp_extension.

    Also populates state with the parameters the profiling harness needs so it
    can recompile inside the profiled subprocess (torch's build cache will
    skip the actual nvcc work since the hash hasn't changed).
    """
    task_dir_abs = state["task_dir_abs"]
    try:
        task_def = load_task_def(task_dir_abs)
    except Exception as e:
        return False, f"task def load failed: {e}"

    cuda_flags: list[str] = []
    if _is_class2_general_task(task_dir_abs):
        target_sm = _get_device_sm()
        if target_sm > 0:
            cuda_flags = _current_arch_gencode_flags(target_sm)

    include_dirs = get_solution_include_dirs(get_project_root())

    state["fn_sig"] = task_def["FUNCTION_SIGNATURE"]
    state["cuda_flags"] = cuda_flags
    state["include_dirs"] = include_dirs

    build_dir = os.path.join(state["tmp_dir"], "torch_build")
    state["build_dir"] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    try:
        compile_cuda_module(
            cuda_source=state["sol_path"],
            function_signature=task_def["FUNCTION_SIGNATURE"],
            extra_include_dirs=include_dirs or None,
            extra_cuda_cflags=cuda_flags or None,
            build_dir=build_dir,
            verbose=False,
        )
    except Exception as e:
        return False, str(e)[-2000:]
    return True, ""


def _profile_compile_class1(config: TaskConfig, state: dict) -> tuple[bool, str]:
    work_dir = state["work_dir"]
    env = os.environ.copy()
    root = get_project_root()
    ref_sources = os.path.join(root, "reference_sources")
    env["CUTLASS_DIR"] = os.path.join(ref_sources, "cutlass")
    env["TK_DIR"] = os.path.join(ref_sources, "ThunderKittens")
    env["DEEPGEMM_DIR"] = os.path.join(ref_sources, "deep_gemm")

    try:
        if config.build.clean_cmd:
            subprocess.run(
                shlex.split(config.build.clean_cmd),
                cwd=work_dir,
                capture_output=True,
                timeout=30,
                env=env,
            )
    except Exception:
        pass

    build_cmd = config.build.cmd or ""
    if not build_cmd:
        return False, "No build command configured."

    build_cmd_args = shlex.split(build_cmd)
    target_sm = 0
    if _is_class1_general_task(config):
        target_sm = get_gpu_sm()
        if target_sm > 0:
            make_vars = _current_arch_make_vars(target_sm)
            build_cmd_args = _format_make_cmd(build_cmd, make_vars)
            ref_cmd = ["make", "ref"] + [f"{k}={v}" for k, v in make_vars.items()]
            try:
                ref_proc = subprocess.run(
                    ref_cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=config.runner.timeout_sec,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return False, (
                    f"{UNSUPPORTED_ARCH_PREFIX} class1/general reference build timed out "
                    f"for target sm_{target_sm}"
                )
            if ref_proc.returncode != 0:
                err = ref_proc.stderr[-2000:] if ref_proc.stderr else "Unknown reference compilation error"
                return False, (
                    f"{UNSUPPORTED_ARCH_PREFIX} class1/general reference build failed for target "
                    f"sm_{target_sm}: {err}"
                )

    try:
        proc = subprocess.run(
            build_cmd_args,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=config.runner.timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out during profiling setup."

    if proc.returncode != 0:
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return False, output[-2000:] if output else "Unknown compilation error"
    return True, ""


def _run_profiler_with_pgroup(
    cmd: list[str],
    *,
    cwd: str,
    env: dict,
    timeout_sec: int,
) -> tuple[int, str, str, bool]:
    """Run a profiler (nsys / ncu) as a new session leader and reap the whole
    process group on timeout.

    Plain subprocess.run(..., timeout=...) only kills the top-level process.
    nsys/ncu spawn helpers (nsys-agent, nsys-launcher, nsys-tee, the target
    Python + its CUDA context) that become orphans holding GPU memory when
    the top-level gets SIGKILLed in isolation. start_new_session=True puts
    the child and all descendants into one process group keyed to the
    child's PID, and killpg sweeps them all together.

    Returns (returncode, stdout, stderr, timed_out).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            # Something is stuck even after SIGKILL — best-effort
            stdout, stderr = "", ""
        return -9, stdout, stderr, True


def _run_profile(
    config: TaskConfig,
    code: str,
    tool_name: str,
    kernel_name: str = "",
    profile_opts: Optional[dict] = None,
) -> str:
    """Run nsys or ncu on the supplied solution code.

    profile_opts (optional) is a dict of model-chosen customization:
        set:          ncu --set value (basic|default|detailed|full|roofline|source)
        sections:     list of ncu --section names (each passed separately)
        metrics:      list of ncu --metrics regex fragments (comma-joined)
        launch_count: int override for ncu --launch-count (default 10)
        trace:        nsys -t value (e.g. "cuda,cublas,nvtx")
    """
    backend = config.runner.backend
    if backend not in ("class1_make", "class2_defpy"):
        return f"PROFILING UNSUPPORTED: backend {backend} not supported in eval_agent.py."

    state = _profile_state_init(config)
    try:
        ok, msg = _profile_write_solution(config, state, code)
        if not ok:
            return msg

        ok, msg = _profile_compile(config, state)
        if not ok:
            if msg.startswith(UNSUPPORTED_ARCH_PREFIX):
                return msg
            return f"PROFILING PREP FAILED:\n{msg}"

        if backend == "class1_make":
            return _run_profile_class1(config, state, tool_name, kernel_name, profile_opts or {})
        return _run_profile_class2(config, state, tool_name, kernel_name, profile_opts or {})
    finally:
        _profile_state_cleanup(state)


def _format_nsys_output(output: str, tool_name: str) -> str:
    lines = output.splitlines()
    if tool_name == "nsys":
        relevant = [
            line for line in lines
            if any(key in line.lower() for key in (
                "kernel", "memcpy", "memset", "cuda api", "time(%)", "avg", "total"
            ))
        ]
        if relevant:
            return "Nsight Systems Profile Summary:\n" + "\n".join(relevant[:50])
    else:
        relevant = [
            line for line in lines
            if any(key in line for key in (
                "Achieved Occupancy", "Memory Throughput", "Compute (SM)",
                "SM [%]", "Memory [%]", "Duration", "Registers",
                "Shared Memory", "Warp Cycles",
            ))
        ]
        if relevant:
            return "Nsight Compute Metrics:\n" + "\n".join(relevant[:30])
    return output[-MAX_PROFILE_OUTPUT:] if output else f"{tool_name.upper()} produced no output."


def _profiler_env_extras() -> dict:
    """Env vars that make nsys / ncu robust on shared multi-user systems.

    Problem: nsys's default session scratch path is ``$TMPDIR/nvidia/nsight_systems``
    (TMPDIR defaults to /tmp). On shared systems where another user — or a
    previous install running as root — has already created
    ``/tmp/nvidia/nsight_systems``, nsys aborts at startup with
    'Failed to create directory ... : Permission denied', producing no
    profiling output at all. The model then sees a short permission error
    instead of real data.

    Fix: give every invocation a per-user TMPDIR inside the user's home so
    nsys creates its session subdir on ground we own. We try the XDG cache
    dir first and fall back to an in-home path if that isn't writable
    (NFS home, read-only XDG_CACHE_HOME, etc.). Only set TMPDIR once a
    writable location is confirmed — leaving the user's existing TMPDIR
    intact if we can't improve on it.
    """
    candidates = [
        os.path.join(
            os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
            "cuda_hercules_profiler_tmp",
        ),
        os.path.expanduser("~/.cuda_hercules_profiler_tmp"),
    ]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            # Probe writability so we don't set TMPDIR to a dir that
            # `open(..., 'w')` will later reject (NFS mount w/ weird acls).
            probe = os.path.join(path, ".probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return {"TMPDIR": path}
        except OSError:
            continue
    # All preferred paths failed; leave TMPDIR alone. Better to preserve the
    # user's existing TMPDIR than point it somewhere unwritable.
    return {}


def _build_ncu_cmd(kernel_name: str, opts: dict, demangle: bool = False) -> list[str]:
    """Compose an ncu invocation applying optional model-specified params.

    Defaults (used if opts is empty): --set detailed, --launch-count 10.
    """
    set_val = (opts.get("set") or "detailed").strip().lower()
    if set_val not in {"basic", "default", "detailed", "full", "roofline", "source"}:
        set_val = "detailed"
    try:
        launch_count = int(opts.get("launch_count") or 10)
    except (TypeError, ValueError):
        launch_count = 10
    # Clamp launch count to a sane range so a bad model suggestion doesn't
    # blow ncu past the subprocess timeout.
    launch_count = max(1, min(launch_count, 50))

    cmd = ["ncu", "--set", set_val, "--csv", "--launch-count", str(launch_count)]
    for section in opts.get("sections") or []:
        cmd.extend(["--section", section])
    metrics = opts.get("metrics") or []
    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    if demangle:
        cmd.extend(["--kernel-name-base", "demangled"])
    if kernel_name:
        # For class2 harness (demangle=True) the caller wraps in regex:; for
        # class1 the raw name usually matches fine.
        cmd.extend(["--kernel-name", f"regex:{kernel_name}" if demangle else kernel_name])
    return cmd


def _build_nsys_cmd(opts: dict, extra_flags: Optional[list[str]] = None) -> list[str]:
    cmd = ["nsys", "profile", "--stats=true", "--force-overwrite=true",
           "-o", "/tmp/cuda_toolaug_nsys"]
    trace = (opts.get("trace") or "").strip()
    if trace:
        cmd.extend(["-t", trace])
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def _run_profile_class1(config: TaskConfig, state: dict, tool_name: str, kernel_name: str, opts: dict) -> str:
    env = os.environ.copy()
    env.update(config.runner.env)
    env.update(_profiler_env_extras())
    run_cmd = shlex.split(config.execute.cmd or "./test")

    if tool_name == "nsys":
        cmd = _build_nsys_cmd(opts) + run_cmd
        timeout_sec = min(max(config.runner.timeout_sec, 120), 180)
    elif tool_name == "ncu":
        cmd = _build_ncu_cmd(kernel_name, opts) + run_cmd
        timeout_sec = min(max(config.runner.timeout_sec, 180), 240)
    else:
        return f"Unsupported profiling tool: {tool_name}"

    try:
        rc, stdout, stderr, timed_out = _run_profiler_with_pgroup(
            cmd, cwd=state["work_dir"], env=env, timeout_sec=timeout_sec,
        )
    except FileNotFoundError:
        return f"{tool_name.upper()} NOT FOUND in PATH"
    if timed_out:
        return f"{tool_name.upper()} TIMED OUT"

    output = ((stdout or "") + "\n" + (stderr or "")).strip()
    return _format_nsys_output(output, tool_name)


# Harness that runs inside the profiled subprocess for class2.
# - Skips import / CUDA-init / warmup noise via torch.cuda.profiler.start/stop.
# - Reuses the sanity-built .so (same build_dir → torch's JIT cache hits).
_CLASS2_HARNESS_TEMPLATE = """\
import os, sys
sys.path.insert(0, os.path.join({project_root!r}, "src"))

import torch
from cuda_hercules.eval import load_task_def, _build_call_args
from cuda_hercules.compiler import compile_cuda_module

TASK_DIR = {task_dir!r}
SOLUTION_CU = {sol_path!r}
BUILD_DIR = {build_dir!r}
EXTRA_CUDA_CFLAGS = {cuda_flags!r}
EXTRA_INCLUDE_DIRS = {include_dirs!r}
FUNCTION_SIGNATURE = {fn_sig!r}
NUM_WARMUP = {num_warmup}
NUM_TRIALS = {num_trials}

task_def = load_task_def(TASK_DIR)

sol_fn = compile_cuda_module(
    cuda_source=SOLUTION_CU,
    function_signature=FUNCTION_SIGNATURE,
    extra_include_dirs=EXTRA_INCLUDE_DIRS or None,
    extra_cuda_cflags=EXTRA_CUDA_CFLAGS or None,
    build_dir=BUILD_DIR,
    verbose=False,
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
inputs = task_def["get_inputs"]()
outputs = task_def["get_outputs"](inputs)
args = _build_call_args(task_def, inputs, outputs)

# Warmup — NOT profiled
for _ in range(NUM_WARMUP):
    sol_fn(*args)
torch.cuda.synchronize()

# Measured runs — inside profiler window
torch.cuda.profiler.start()
for _ in range(NUM_TRIALS):
    sol_fn(*args)
torch.cuda.synchronize()
torch.cuda.profiler.stop()

print("HARNESS DONE", flush=True)
"""


def _run_profile_class2(config: TaskConfig, state: dict, tool_name: str, kernel_name: str, opts: dict) -> str:
    root = get_project_root()
    harness_code = _CLASS2_HARNESS_TEMPLATE.format(
        project_root=root,
        task_dir=state["task_dir_abs"],
        sol_path=state["sol_path"],
        build_dir=state["build_dir"],
        cuda_flags=state.get("cuda_flags", []),
        include_dirs=state.get("include_dirs", []),
        fn_sig=state["fn_sig"],
        num_warmup=3,
        num_trials=20,
    )
    harness_path = os.path.join(state["tmp_dir"], "profile_harness.py")
    with open(harness_path, "w") as f:
        f.write(harness_code)

    env = os.environ.copy()
    env.update(config.runner.env)
    env.update(_profiler_env_extras())
    python_exe = sys.executable

    if tool_name == "nsys":
        # Default trace set for class2 harness: cuda + nvtx covers kernels
        # and the profiler.start/stop NVTX markers. Model can override via
        # opts["trace"] (e.g. "cuda,cublas,cudnn,osrt,nvtx").
        default_trace = (opts.get("trace") or "cuda,nvtx").strip()
        cmd = _build_nsys_cmd(
            {"trace": default_trace},
            extra_flags=["--capture-range=cudaProfilerApi", "--capture-range-end=stop"],
        ) + [python_exe, harness_path]
        timeout_sec = 300
    elif tool_name == "ncu":
        # --profile-from-start off + torch.cuda.profiler.start/stop limits
        # measurement to our measured window (skips CUDA init + warmup noise).
        # Model can override set / sections / metrics / launch_count via opts.
        cmd = _build_ncu_cmd(kernel_name, opts, demangle=True) + [
            "--profile-from-start", "off",
            python_exe, harness_path,
        ]
        timeout_sec = 360
    else:
        return f"Unsupported profiling tool: {tool_name}"

    try:
        rc, stdout, stderr, timed_out = _run_profiler_with_pgroup(
            cmd, cwd=state["tmp_dir"], env=env, timeout_sec=timeout_sec,
        )
    except FileNotFoundError:
        return f"{tool_name.upper()} NOT FOUND in PATH"
    if timed_out:
        return f"{tool_name.upper()} TIMED OUT"

    output = ((stdout or "") + "\n" + (stderr or "")).strip()
    if "HARNESS DONE" not in output and rc != 0:
        tail = output[-2000:] if output else "(no output)"
        return f"{tool_name.upper()} FAILED (exit={rc}):\n{tail}"
    return _format_nsys_output(output, tool_name)


def _format_correctness_diff(tr: TaskResult) -> str:
    if not tr.compiled or tr.correct:
        return ""

    detail = tr.correctness_detail or {}
    if not detail:
        return ""

    diff_lines = []
    for name, info in detail.items():
        if not isinstance(info, dict) or info.get("correct", True):
            continue
        if "max_diff" in info:
            diff_lines.append(
                f"- Output '{name}': max_diff={info['max_diff']:.6g}, "
                f"mean_diff={info['mean_diff']:.6g}, "
                f"mismatched={info.get('num_mismatched', '?')}/{info.get('total_elements', '?')}\n"
                f"  At worst element: expected={info.get('ref_at_max', '?')}, got={info.get('sol_at_max', '?')}"
            )
        elif "error" in info:
            diff_lines.append(f"- Output '{name}': {info['error']}")

    return "## Correctness Details\n" + "\n".join(diff_lines) if diff_lines else ""


def _save_task_result(task_out: str, result: dict) -> None:
    with open(os.path.join(task_out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)


_ISOLATED_RUN_TASK_TEMPLATE = r"""
import json, os, sys
sys.path.insert(0, os.path.join({root!r}, 'src'))

from cuda_hercules.runner import discover_tasks, run_task
from cuda_hercules.utils import get_project_root

root = get_project_root()
tasks = {{t.task_id: t for t in discover_tasks(root)}}
config = tasks[{task_id!r}]
tr = run_task(config, {sol_path!r}, measure_perf=True, verbose=False)

payload = {{
    "compiled": tr.compiled,
    "correct": tr.correct,
    "correctness_detail": tr.correctness_detail or {{}},
    "speedup": tr.speedup,
    "latency_mean_ms": tr.latency_mean_ms,
    "ref_latency_mean_ms": tr.ref_latency_mean_ms,
    "per_size": tr.per_size or [],
    "status": tr.status.value if tr.status else "ERROR",
    "error_msg": tr.error_msg or "",
}}
sys.stdout.write("__RESULT_JSON__" + json.dumps(payload) + "\n")
"""


def _run_task_isolated(
    config: TaskConfig,
    sol_path: str,
    timeout_sec: int = 900,
) -> TaskResult:
    """Run `run_task` in a fresh subprocess.

    Why: class2_defpy loads the solution via torch.cpp_extension in the
    current process. A single illegal memory access in any round poisons
    the CUDA context for the entire controller, cascading FAILs through
    every subsequent task (torch.manual_seed starts raising
    cudaErrorIllegalAddress). Running each eval in a subprocess gives
    each round a clean CUDA context.
    """
    project_root = get_project_root()
    script = _ISOLATED_RUN_TASK_TEMPLATE.format(
        root=project_root,
        task_id=config.task_id,
        sol_path=sol_path,
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=project_root,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return TaskResult(
            task_id=config.task_id,
            status=TaskStatus.ERROR,
            error_msg=f"subprocess timeout ({timeout_sec}s)",
            domain=config.domain,
            level=config.task_class,
        )

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("__RESULT_JSON__"):
            try:
                payload = json.loads(line[len("__RESULT_JSON__"):])
            except Exception as e:
                return TaskResult(
                    task_id=config.task_id,
                    status=TaskStatus.ERROR,
                    error_msg=f"could not parse subprocess result: {e}",
                    domain=config.domain,
                    level=config.task_class,
                )
            return TaskResult(
                task_id=config.task_id,
                compiled=payload.get("compiled", False),
                correct=payload.get("correct", False),
                correctness_detail=payload.get("correctness_detail", {}),
                speedup=payload.get("speedup", 0.0),
                latency_mean_ms=payload.get("latency_mean_ms", 0.0),
                ref_latency_mean_ms=payload.get("ref_latency_mean_ms", 0.0),
                per_size=payload.get("per_size", []),
                status=TaskStatus(payload.get("status", "ERROR")),
                error_msg=payload.get("error_msg", ""),
                domain=config.domain,
                level=config.task_class,
            )

    err_tail = (proc.stderr or "")[-800:]
    return TaskResult(
        task_id=config.task_id,
        status=TaskStatus.ERROR,
        error_msg=f"subprocess failed (exit={proc.returncode}): {err_tail}",
        domain=config.domain,
        level=config.task_class,
    )


def eval_task_tool_aug(
    config: TaskConfig,
    max_iterations: int,
    provider: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    task_timeout: int,
    run_dir: str,
    verbose: bool = True,
    vertex_region: str = "global",
    vertex_project: str = "neu-research",
    allowed_tools: Optional[set[str]] = None,
    is_reasoning_model: bool = False,
    reasoning_effort: str = "",
) -> dict:
    if allowed_tools is None:
        allowed_tools = set(VALID_PROFILE_TOOLS)
    root = get_project_root()
    task_dir = os.path.join(root, config.runner.workdir) if config.runner.workdir else ""
    task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
    os.makedirs(task_out, exist_ok=True)

    result = {
        "task_id": config.task_id,
        "task_class": config.task_class,
        "domain": config.domain,
        "model": model,
        "provider": provider,
        "final_status": TaskStatus.FAIL.value,
        "max_iterations": max_iterations,
        "passed_at_round": -1,
        "compiled_at_round": -1,
        "submitted_at_round": -1,
        "total_rounds": 0,
        "best_speedup": 0.0,
        "best_solution_file": "",
        "profile_calls_total": 0,
        "profile_calls_by_tool": {"nsys": 0, "ncu": 0},
        "rounds": [],
    }

    try:
        yaml_dir = getattr(config, "_yaml_dir", task_dir)
        task_prompt = build_prompt(task_dir, config, description_dir=yaml_dir) + _current_gpu_text()
        # Up-front anti-cheat warnings so the model doesn't spend rounds
        # discovering the rules via BLOCKED responses. Tailored per-backend.
        if config.runner.backend == "class1_make":
            task_prompt += (
                "\n\n## Important: Library Restrictions\n\n"
                "You MUST implement the kernel from scratch in hand-written "
                "CUDA. The following are **NOT ALLOWED** and will fail the "
                "static check:\n"
                "- `#include <cutlass/...>` or any CUTLASS template\n"
                "- `#include <cublas*>` / `cublasSgemm` / `cublasGemmEx` and "
                "any other cuBLAS wrapper\n"
                "- `#include <cudnn*>` / cuDNN API calls\n\n"
                "You MAY use:\n"
                "- Raw CUDA kernels (`__global__`) with shared memory, warp "
                "shuffles, async copies, etc.\n"
                "- Tensor-core intrinsics via `mma.h` (wmma / mma.sync) and "
                "`wgmma` on Hopper+\n"
                "- Inline PTX (`asm volatile (\"...\")`) for anything the "
                "intrinsics don't expose.\n\n"
                "Copying the reference CUTLASS/cuBLAS implementation defeats "
                "the benchmark's purpose — the reference is what you are "
                "being compared against."
            )
        elif config.runner.backend == "class2_defpy":
            task_prompt += (
                "\n\n## Important: Library Restrictions\n\n"
                "You MUST implement the kernel from scratch. The following "
                "domain-specific reference libraries are **BLOCKED** by the "
                "static check:\n"
                "- Flash Attention: `#include <flash_attn*>`, `flash_fwd*`, "
                "`flash_bwd*` headers\n"
                "- FFT: `#include <cufft*>`, `cufftPlan*`, `cufftExec*`\n"
                "- cuDNN: `#include <cudnn*>` and its wrapper calls\n\n"
                "You MAY use:\n"
                "- `#include <cutlass/...>` — CUTLASS templates ARE permitted "
                "here (the headers are on the include path). This is "
                "different from class1.\n"
                "- Raw CUDA, tensor-core intrinsics (`mma.h`, `wgmma`), "
                "inline PTX.\n\n"
                "Copying the reference (cuFFT / flash-attn / cuDNN) "
                "defeats the benchmark — those are what you are being "
                "compared against. Hand-rolled Cooley-Tukey FFT, "
                "flash-attention in shared memory, Welford layernorm, etc. "
                "are the intended paths."
            )
    except Exception as e:
        result["final_status"] = TaskStatus.ERROR.value
        result["error"] = f"Prompt build failed: {e}"
        return result

    with open(os.path.join(task_out, "prompt.txt"), "w") as f:
        f.write(task_prompt)

    previous_code = ""
    latest_feedback = ""
    current_passing_code = ""
    # last_compiled_code: latest code that cleared compilation (correctness
    # may or may not have passed). Used as the profile target so the model
    # can inspect any compilable attempt, not just a passing one.
    last_compiled_code = ""
    best_solution_code = ""
    # Sticky anti-cheat hint: once a task triggers BLOCKED with a
    # library-level hint, every subsequent FIX feedback re-appends that
    # hint. Without stickiness, the model "forgets" the rule whenever
    # correctness fails on a hand-written attempt and goes back to the
    # banned library on the next round.
    sticky_anticheat_hint = ""

    for round_idx in range(max_iterations):
        round_label = f"r{round_idx}"
        if verbose:
            print(f"    [ROUND {round_idx + 1}/{max_iterations}] ", end="", flush=True)

        timeout_flag = {"timed_out": False}
        timer = threading.Timer(
            task_timeout,
            _round_timeout_trigger,
            args=(os.getpid(), timeout_flag),
        )
        timer.daemon = True
        timer.start()

        llm_calls = 0
        llm_time_s = 0.0
        profile_calls = []
        response_repairs = 0
        profile_budget = MAX_PROFILE_CALLS_PER_ROUND
        current_code = ""

        messages = _build_messages(task_prompt, previous_code, latest_feedback, allowed_tools)
        round_info = {
            "round": round_idx,
            "llm_calls": 0,
            "llm_time_s": 0.0,
            "model_action": "",
            "submitted": False,
            "profile_calls": [],
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "status": TaskStatus.FAIL.value,
            "error": "",
        }

        try:
            while True:
                t0 = time.time()
                response = _query_model(
                    messages=messages,
                    provider=provider,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    vertex_region=vertex_region,
                    vertex_project=vertex_project,
                    is_reasoning_model=is_reasoning_model,
                    reasoning_effort=reasoning_effort,
                )
                llm_time_s += time.time() - t0
                llm_calls += 1

                parsed = _parse_model_response(response)
                action = parsed["action"]
                round_info["model_action"] = action
                round_info["raw_response_preview"] = response[:400]
                # Only record the model's stated reason in the post-PASS phase
                # (when we actually asked for one via OPTIMIZE_TEMPLATE /
                # PROFILE_FOLLOWUP_TEMPLATE). Pre-PASS rounds are fix-the-bug
                # mode where a reason field adds noise without insight.
                if current_passing_code and parsed["reason"]:
                    round_info["model_reason"] = parsed["reason"]
                    if verbose:
                        reason_preview = parsed["reason"][:180]
                        print(f"\n    REASON[{action}]: {reason_preview}", flush=True)

                if action == "profile":
                    tool_name = parsed["tool"]
                    if tool_name not in allowed_tools:
                        if allowed_tools:
                            allowed_list = ", ".join(f"`{t}`" for t in sorted(allowed_tools))
                            if tool_name in VALID_PROFILE_TOOLS:
                                hint = (
                                    f"Profiler `{tool_name}` is not available in this evaluation run. "
                                    f"Only {allowed_list} may be used. Return code or submit."
                                )
                            else:
                                hint = f"Invalid profiling tool. Use only {allowed_list}."
                        else:
                            hint = "No profilers are available in this evaluation run. Return code or submit."
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": MALFORMED_RESPONSE_TEMPLATE + "\n" + hint,
                        })
                        response_repairs += 1
                        if response_repairs > 2:
                            round_info["error"] = "Malformed profiling request."
                            break
                        continue

                    if profile_budget <= 0:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": "Profiling budget exhausted for this round. Return code or submit.",
                        })
                        continue

                    # Profile uses the latest CODE that compiled — correctness
                    # may or may not have passed. This lets the model profile
                    # any compilable attempt to understand bottlenecks, not
                    # just solutions that already clear correctness.
                    profile_target_code = last_compiled_code or current_passing_code
                    if not profile_target_code:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": "No code has compiled yet, so there is nothing to profile. Return a complete solution first (action=write_code).",
                        })
                        continue

                    # Profiling (esp. ncu) replays the target binary multiple
                    # times for counter collection and can easily run for
                    # several minutes. Exempt it from the per-round wall-clock
                    # timer — each profile subprocess already has its own
                    # subprocess.run(timeout=...) guarding against hangs, and
                    # the next loop iteration will arm a fresh round timer.
                    timer.cancel()
                    profile_opts = {
                        "set": parsed.get("set", ""),
                        "sections": parsed.get("sections", []),
                        "metrics": parsed.get("metrics", []),
                        "launch_count": parsed.get("launch_count"),
                        "trace": parsed.get("trace", ""),
                    }
                    if verbose:
                        krn = parsed["kernel_name"] or "-"
                        opts_suffix = ""
                        if profile_opts["set"] or profile_opts["sections"] or profile_opts["metrics"] or profile_opts["trace"]:
                            parts = []
                            if profile_opts["set"]:
                                parts.append(f"set={profile_opts['set']}")
                            if profile_opts["sections"]:
                                parts.append(f"sec={'+'.join(profile_opts['sections'])}")
                            if profile_opts["metrics"]:
                                parts.append(f"met={len(profile_opts['metrics'])}")
                            if profile_opts["trace"]:
                                parts.append(f"trace={profile_opts['trace']}")
                            opts_suffix = " " + " ".join(parts)
                        print(f"PROFILE[{tool_name}:{krn}{opts_suffix}]", end=" → ", flush=True)
                    profile_t0 = time.time()
                    profile_text = _run_profile(
                        config=config,
                        code=profile_target_code,
                        tool_name=tool_name,
                        kernel_name=parsed["kernel_name"],
                        profile_opts=profile_opts,
                    )
                    if verbose:
                        print(f"done ({time.time() - profile_t0:.0f}s)", flush=True)
                    profile_text = profile_text[-MAX_PROFILE_OUTPUT:]
                    profile_calls.append({
                        "tool": tool_name,
                        "kernel_name": parsed["kernel_name"],
                        "reason": parsed["reason"],
                        "opts": {k: v for k, v in profile_opts.items() if v},
                        "output_preview": profile_text[:600],
                    })
                    result["profile_calls_total"] += 1
                    result["profile_calls_by_tool"][tool_name] += 1
                    profile_budget -= 1

                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": PROFILE_FOLLOWUP_TEMPLATE.format(profile_output=profile_text),
                    })
                    continue

                if action == "submit":
                    round_info["submitted"] = True
                    round_info["status"] = (
                        TaskStatus.PASS.value
                        if result["passed_at_round"] >= 0
                        else result["final_status"]
                    )
                    result["submitted_at_round"] = round_idx
                    result["total_rounds"] = round_idx + 1
                    if result["passed_at_round"] >= 0:
                        result["final_status"] = TaskStatus.PASS.value
                    if verbose:
                        print("SUBMIT", flush=True)
                    result["rounds"].append(round_info)
                    _save_task_result(task_out, result)
                    return result

                if action == "write_code":
                    current_code = parsed["code"]
                    if not current_code or len(current_code.strip()) < 10:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": MALFORMED_RESPONSE_TEMPLATE})
                        response_repairs += 1
                        if response_repairs > 2:
                            round_info["error"] = "Empty code in write_code response."
                            break
                        continue
                    break

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": MALFORMED_RESPONSE_TEMPLATE})
                response_repairs += 1
                if response_repairs > 2:
                    round_info["error"] = "Malformed response."
                    break

            round_info["llm_calls"] = llm_calls
            round_info["llm_time_s"] = llm_time_s
            round_info["profile_calls"] = profile_calls

            if round_info["submitted"]:
                return result

            if not current_code:
                round_info["status"] = TaskStatus.ERROR.value
                if not round_info["error"]:
                    round_info["error"] = "No valid code produced."
                result["rounds"].append(round_info)
                result["total_rounds"] = round_idx + 1
                latest_feedback = MALFORMED_RESPONSE_TEMPLATE
                result["final_status"] = TaskStatus.ERROR.value
                if verbose:
                    print("FORMAT_ERROR", flush=True)
                continue

            sol_name = config.runner.solution_file or "solution.cu"
            sol_path = os.path.join(task_out, f"{round_label}_{sol_name}")
            with open(sol_path, "w") as f:
                f.write(current_code)

            round_info["solution_file"] = os.path.basename(sol_path)
            round_info["lines"] = len(current_code.splitlines())
            previous_code = current_code

            check = validate_cuda_solution(
                current_code,
                blocked_patterns=config.anti_cheat.blocked_patterns or [],
                required_patterns=config.anti_cheat.required_patterns or [],
            )
            if not check.valid:
                error_msg = "; ".join(check.errors)
                round_info["status"] = TaskStatus.FAIL.value
                round_info["error"] = f"Static check failed: {error_msg}"
                result["rounds"].append(round_info)
                result["total_rounds"] = round_idx + 1
                feedback_err = f"Static check failed: {error_msg}"
                # Decide which anti-cheat note fits this task. Same text is
                # also used as the sticky hint so later FIX rounds (triggered
                # by correctness failures, not static-check blocks) keep
                # reminding the model what's off-limits — otherwise the
                # model alternates between "hand-written (fails correctness)"
                # and "use the banned library again (BLOCKED again)".
                anticheat_note = ""
                if config.runner.backend == "class1_make":
                    anticheat_note = (
                        "NOTE: You cannot directly call CUTLASS or cuBLAS "
                        "APIs in this benchmark. You must implement the "
                        "current task with hand-written CUDA code. You MAY "
                        "use `mma` / `wgmma` intrinsics and inline PTX "
                        "(asm volatile (\"...\")) to drive tensor cores "
                        "directly, but must not include CUTLASS or cuBLAS "
                        "headers or call their high-level wrappers."
                    )
                elif config.runner.backend == "class2_defpy" and "/fft_" in config.task_id:
                    anticheat_note = (
                        "NOTE: You cannot call cuFFT (`#include <cufft*>`, "
                        "`cufftPlan*`, `cufftExec*`) in this "
                        "benchmark — those are the reference libraries you "
                        "are being benchmarked against. Implement the FFT "
                        "by hand: a radix-2 / radix-4 Cooley-Tukey butterfly "
                        "in shared memory with `__sincosf` or precomputed "
                        "twiddle tables is the intended path. CUTLASS "
                        "(`#include <cutlass/...>`) is PERMITTED if you "
                        "want its reusable utilities, but it is not "
                        "required — raw CUDA is usually simpler for FFT."
                    )
                if anticheat_note:
                    feedback_err += "\n\n" + anticheat_note
                    sticky_anticheat_hint = anticheat_note
                latest_feedback = FIX_TEMPLATE.format(
                    error=feedback_err,
                    correctness_diff="",
                )
                if verbose:
                    print("BLOCKED", flush=True)
                continue

            tr = _run_task_isolated(config, sol_path, timeout_sec=max(task_timeout, 900))
            round_info["compiled"] = tr.compiled
            round_info["correct"] = tr.correct
            round_info["speedup"] = tr.speedup
            round_info["status"] = tr.status.value
            round_info["error"] = tr.error_msg or ""
            round_info["per_size"] = tr.per_size or []

            result["total_rounds"] = round_idx + 1
            if tr.compiled:
                # Keep the latest compilable code around so the model can
                # profile it via the profile action regardless of correctness.
                last_compiled_code = current_code
                if result["compiled_at_round"] < 0:
                    result["compiled_at_round"] = round_idx
            if tr.correct and result["passed_at_round"] < 0:
                result["passed_at_round"] = round_idx
            if tr.correct:
                current_passing_code = current_code
                result["final_status"] = TaskStatus.PASS.value
                if not best_solution_code or tr.speedup > result["best_speedup"]:
                    result["best_speedup"] = tr.speedup
                    result["best_solution_file"] = os.path.basename(sol_path)
                    best_solution_code = current_code

            result["rounds"].append(round_info)

            report_path = os.path.join(task_out, "final_report.json")
            history = []
            if os.path.exists(report_path):
                with open(report_path) as f:
                    existing = json.load(f)
                history = existing.get("history", [])
            history.append({
                "round": round_idx,
                "status": tr.status.value,
                "compiled": tr.compiled,
                "correct": tr.correct,
                "speedup": tr.speedup,
                "solution_file": os.path.basename(sol_path),
                "profile_calls": profile_calls,
            })
            with open(report_path, "w") as f:
                json.dump({
                    "task_id": config.task_id,
                    "model": model,
                    "provider": provider,
                    "final_status": result["final_status"],
                    "first_compiled_round": result["compiled_at_round"],
                    "first_correct_round": result["passed_at_round"],
                    "submitted_at_round": result["submitted_at_round"],
                    "best_speedup": result["best_speedup"],
                    "best_solution_file": result["best_solution_file"],
                    "history": history,
                }, f, indent=2, default=str)

            if tr.status == TaskStatus.SKIP_ARCH:
                result["final_status"] = TaskStatus.SKIP_ARCH.value
                if verbose:
                    print("SKIP_ARCH", flush=True)
                break

            if tr.correct:
                perf_detail = ""
                if tr.ref_latency_mean_ms > 0:
                    perf_detail = f"- Reference time: {tr.ref_latency_mean_ms:.4f} ms"
                if tr.speedup >= 1.0:
                    perf_detail += "\n- The solution already beats the reference; optimize only if you see a clear next step."
                else:
                    perf_detail += "\n- The solution is still slower than the reference."

                per_size_detail = ""
                if tr.per_size:
                    per_size_lines = []
                    for entry in tr.per_size:
                        per_size_lines.append(
                            f"  - {entry['name']}: {entry['speedup']:.2f}x "
                            f"(kernel={entry['kernel_min_ms']:.4f}ms, ref={entry['ref_min_ms']:.4f}ms)"
                        )
                    per_size_detail = "- Per-size breakdown:\n" + "\n".join(per_size_lines)

                latest_feedback = OPTIMIZE_TEMPLATE.format(
                    speedup=tr.speedup,
                    per_size_detail=per_size_detail,
                    perf_detail=perf_detail,
                )
                status_str = f"PASS {tr.speedup:.2f}x" if tr.speedup > 0 else "PASS"
                if verbose:
                    print(status_str, flush=True)
            else:
                feedback_error = tr.error_msg or "Compilation or correctness check failed."
                if tr.compiled and not tr.correct:
                    feedback_error = "Code compiled successfully but produced incorrect results."
                if sticky_anticheat_hint:
                    feedback_error += "\n\n" + sticky_anticheat_hint
                latest_feedback = FIX_TEMPLATE.format(
                    error=feedback_error,
                    correctness_diff=_format_correctness_diff(tr),
                )
                status_str = "COMPILED" if tr.compiled else tr.status.value
                if verbose:
                    print(status_str, flush=True)

        except KeyboardInterrupt:
            if not timeout_flag["timed_out"]:
                raise
            round_info["llm_calls"] = llm_calls
            round_info["llm_time_s"] = llm_time_s
            round_info["profile_calls"] = profile_calls
            round_info["status"] = TaskStatus.ERROR.value
            round_info["error"] = f"Round timed out after {task_timeout}s"
            result["rounds"].append(round_info)
            result["total_rounds"] = round_idx + 1
            latest_feedback = FIX_TEMPLATE.format(
                error=f"Your previous attempt timed out after {task_timeout}s.",
                correctness_diff="",
            )
            if verbose:
                print("TIMEOUT", flush=True)
            continue
        finally:
            timer.cancel()

    if result["passed_at_round"] < 0 and result["compiled_at_round"] >= 0 and result["final_status"] != TaskStatus.SKIP_ARCH.value:
        result["final_status"] = TaskStatus.FAIL.value

    if best_solution_code and result["best_solution_file"]:
        best_path = os.path.join(task_out, f"solution_best{os.path.splitext(result['best_solution_file'])[1]}")
        with open(best_path, "w") as f:
            f.write(best_solution_code)

    _save_task_result(task_out, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="CUDA-Hercules Tool-Augmented Evaluation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["anthropic", "anthropic_vertex", "openai"],
        help="LLM provider",
    )
    parser.add_argument("--api-base", default="", help="API base URL for OpenAI-compatible servers")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--vertex-region", default="global")
    parser.add_argument("--vertex-project", default="neu-research")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable reasoning/thinking mode (skips temperature for o1/o3/GPT-5/Claude thinking)")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                        help="Reasoning effort level (for o1/o3/GPT-5, default: low)")
    parser.add_argument("--filter", action="append", default=[], help="Task filters")
    parser.add_argument("--max-problems", "--max-tasks", dest="max_tasks", type=int, default=0, help="0=all")
    parser.add_argument("--task-list", default="", help="File with task names to include (one per line)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max controller rounds per task")
    parser.add_argument("--task-timeout", type=int, default=900, help="Per-round wall-clock timeout in seconds")
    parser.add_argument(
        "--profile-tools",
        default="nsys,ncu",
        help="Comma-separated list of allowed profilers (subset of: nsys, ncu). "
             "Pass an empty string to disable profiling entirely. Default: nsys,ncu.",
    )
    parser.add_argument("--run-name", default="", help="Run name for output dir")
    parser.add_argument("--output", default="results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = get_project_root()

    raw_tools = [t.strip().lower() for t in (args.profile_tools or "").split(",") if t.strip()]
    invalid_tools = [t for t in raw_tools if t not in VALID_PROFILE_TOOLS]
    if invalid_tools:
        print(f"Invalid --profile-tools entries: {', '.join(invalid_tools)}. "
              f"Must be subset of: {', '.join(sorted(VALID_PROFILE_TOOLS))}.")
        sys.exit(1)
    allowed_tools = set(raw_tools)

    filters = parse_filters(args.filter)
    tasks = discover_tasks(root, filters)

    if not tasks:
        print("No tasks found.")
        sys.exit(1)

    gpu_sm = get_gpu_sm()
    tasks = [t for t in tasks if t.hardware.min_sm <= gpu_sm]
    if not tasks:
        print(f"No tasks are runnable on GPU SM {gpu_sm}.")
        sys.exit(1)

    if args.task_list:
        with open(args.task_list) as f:
            allowed = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed.add(line)
        tasks = [
            t for t in tasks
            if t.name in allowed or t.task_id in allowed or t.task_id.split("/")[-1] in allowed
        ]
        if not tasks:
            print(f"No tasks matched task-list file: {args.task_list}")
            sys.exit(1)

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    run_name = args.run_name or f"toolaug_{args.model.replace('/', '_')}"
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tools_label = ", ".join(sorted(allowed_tools)) if allowed_tools else "(none)"
    print("CUDA-Hercules Tool-Augmented Evaluation")
    print(f"  Model:         {args.model}")
    print(f"  Provider:      {args.provider}")
    print(f"  Tasks:         {len(tasks)}")
    print(f"  Max rounds:    {args.max_iterations}")
    print(f"  Round timeout: {args.task_timeout}s")
    print(f"  Profilers:     {tools_label}")
    print(f"  GPU:           SM {gpu_sm}")
    print(f"  Output:        {run_dir}")
    print()

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "model": args.model,
            "provider": args.provider,
            "api_base": args.api_base,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_iterations": args.max_iterations,
            "task_timeout": args.task_timeout,
            "filters": args.filter,
            "task_list": args.task_list,
            "num_tasks": len(tasks),
            "gpu_sm": gpu_sm,
            "allowed_profile_tools": sorted(allowed_tools),
        }, f, indent=2)

    all_raw = []
    task_results = []
    completed_ids = set()
    for config in tasks:
        task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
        task_result_file = os.path.join(task_out, "result.json")
        if os.path.isfile(task_result_file):
            with open(task_result_file) as f:
                raw = json.load(f)
            all_raw.append(raw)
            raw_status = raw.get("final_status", TaskStatus.FAIL.value)
            task_results.append(TaskResult(
                task_id=raw["task_id"],
                compiled=raw.get("compiled_at_round", -1) >= 0,
                correct=raw.get("passed_at_round", -1) >= 0,
                speedup=raw.get("best_speedup", 0.0),
                domain=raw.get("domain", ""),
                level=raw.get("task_class", 0),
                status=TaskStatus(raw_status),
            ))
            completed_ids.add(config.task_id)
    if completed_ids:
        print(f"  Resume: {len(completed_ids)} tasks already done, skipping\n")

    for i, config in enumerate(tasks):
        if config.task_id in completed_ids:
            continue

        if not args.quiet:
            print(f"[{i + 1}/{len(tasks)}] {config.task_id}", flush=True)

        raw = eval_task_tool_aug(
            config=config,
            max_iterations=args.max_iterations,
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            task_timeout=args.task_timeout,
            run_dir=run_dir,
            verbose=not args.quiet,
            vertex_region=args.vertex_region,
            vertex_project=args.vertex_project,
            allowed_tools=allowed_tools,
            is_reasoning_model=args.reasoning,
            reasoning_effort=args.reasoning_effort,
        )
        all_raw.append(raw)
        task_results.append(TaskResult(
            task_id=raw["task_id"],
            compiled=raw.get("compiled_at_round", -1) >= 0,
            correct=raw.get("passed_at_round", -1) >= 0,
            speedup=raw.get("best_speedup", 0.0),
            domain=raw.get("domain", ""),
            level=raw.get("task_class", 0),
            status=TaskStatus(raw.get("final_status", TaskStatus.FAIL.value)),
        ))

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(all_raw, f, indent=2, default=str)

    score = compute_scores(task_results)
    report_text = format_score(score)
    print(f"\n{report_text}")

    total = len(all_raw)
    pass_by_round = {}
    compile_by_round = {}
    profile_calls_total = 0
    profile_calls_by_tool = {"nsys": 0, "ncu": 0}
    for raw in all_raw:
        pr = raw.get("passed_at_round", -1)
        cr = raw.get("compiled_at_round", -1)
        if pr >= 0:
            pass_by_round[pr] = pass_by_round.get(pr, 0) + 1
        if cr >= 0:
            compile_by_round[cr] = compile_by_round.get(cr, 0) + 1
        profile_calls_total += raw.get("profile_calls_total", 0)
        raw_by_tool = raw.get("profile_calls_by_tool", {})
        for tool_name in ("nsys", "ncu"):
            profile_calls_by_tool[tool_name] += raw_by_tool.get(tool_name, 0)

    print(f"\nTool-Augmented Statistics (max {args.max_iterations} rounds):")
    cumulative_pass = 0
    cumulative_compile = 0
    for r in range(args.max_iterations):
        p = pass_by_round.get(r, 0)
        c = compile_by_round.get(r, 0)
        cumulative_pass += p
        cumulative_compile += c
        print(
            f"  round-{r:<2}: +{p} pass (cum: {cumulative_pass}/{total} = {cumulative_pass / total:.1%}), "
            f"+{c} compile (cum: {cumulative_compile}/{total} = {cumulative_compile / total:.1%})"
        )
    print(
        f"  profiling: {profile_calls_total} total "
        f"(nsys={profile_calls_by_tool['nsys']}, ncu={profile_calls_by_tool['ncu']})"
    )

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(all_raw, f, indent=2, default=str)
    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(asdict(score), f, indent=2, default=str)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report_text + "\n")
    with open(os.path.join(run_dir, "tool_aug_stats.json"), "w") as f:
        json.dump({
            "total": total,
            "max_iterations": args.max_iterations,
            "pass_by_round": pass_by_round,
            "compile_by_round": compile_by_round,
            "profile_calls_total": profile_calls_total,
            "profile_calls_by_tool": profile_calls_by_tool,
        }, f, indent=2)

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
