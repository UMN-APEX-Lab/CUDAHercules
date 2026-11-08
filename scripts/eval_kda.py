#!/usr/bin/env python3
"""
CUDA-Hercules evaluation of KDA (Kernel Design Agents, github.com/NVlabs/kda).

KDA is a workflow + prompts. This harness runs it unattended ("lite": KDA
autonomous-flow prompt + KDA budget hook + ncu-report-skill docs; no
KernelWiki, no humanize/Codex) with a minimal tool-calling loop over an
OpenAI-compatible API (scripts/kda_eval/agent_loop.py), one session per task,
under strict isolation:

  * The agent loop and the API key stay outside the sandbox. Every tool call
    (bash / read / write / edit) runs in a fresh bwrap sandbox
    (scripts/kda_eval/sandbox.py) with no network that sees only the task
    workspace, the CUDA toolkit, the CUTLASS include dirs solutions may
    #include, and one GPU. KernelHell, reference code, other runs' results,
    conda envs and the host home directory do not exist inside.
  * The agent evaluates candidates only through `submit`, which talks to a
    judge outside the sandbox (scripts/kda_eval/judge.py). The judge enforces
    the evaluation and wall-clock budget.

After the session, the best timed PASS submission is re-evaluated in a clean
process (agent gone, GPU quiet); that re-evaluation is the task's score.
Output layout matches scripts/eval_agent.py (result.json per task with
best_solution_file), so hacking_audit.py and aggregation scripts work as is.

KDA-lite files: the unattended prompt (kda_lite/autonomous-flow.md, adapted
from kda/prompts/basic-flow.md, CC-BY-4.0) and the budget hook
(kda_lite/budget_guard.py) live in scripts/kda_eval/kda_lite/. KDA itself is
the `kda` submodule (upstream NVlabs/kda, unmodified); only its
ncu-report-skill is used — KernelWiki is deliberately not initialized:
    git submodule update --init kda
    git -C kda submodule update --init skills/ncu-report-skill

Usage:
    python scripts/eval_kda.py --model claude-opus-5 --env-file ~/Dr.Sparse/.env \\
        --gpus 2,5 --filter arch=general --task-list results/kda/pilot_tasks.txt \\
        --time-limit-min 60 --max-evals 10 --run-name kda_lite_pilot
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from kda_eval import sandbox as sbx  # noqa: E402

KH_ROOT = sbx.KH_ROOT
KDA_ROOT = os.path.join(KH_ROOT, "kda")  # submodule, upstream NVlabs/kda
KDA_LITE = os.path.join(HERE, "kda_eval", "kda_lite")
NCU_SKILL = os.path.join(KDA_ROOT, "skills", "ncu-report-skill")
SUPPORTED_BACKENDS = ("class1_make", "class2_defpy")


def load_api_config(args) -> tuple[str, str]:
    """(api_base, api_key) from the environment or --env-file. Never logged."""
    file_env = {}
    if args.env_file and os.path.isfile(os.path.expanduser(args.env_file)):
        for line in open(os.path.expanduser(args.env_file)):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                file_env[k.strip()] = v.strip().strip('"').strip("'")
    key = os.environ.get(args.api_key_env) or file_env.get(args.api_key_env, "")
    base = args.api_base or os.environ.get(args.api_base_env) or file_env.get(args.api_base_env, "")
    if not key or not base:
        raise RuntimeError(f"missing {args.api_key_env} / {args.api_base_env} (env or --env-file)")
    return base.rstrip("/"), key


# ---------------------------------------------------------------------------
# Workspace + prompt
# ---------------------------------------------------------------------------

def _makefile_build_flags(config) -> tuple[str, str]:
    """(NVCC_FLAGS, ARCH) exactly as the task's Makefile builds the solution; ("", "") if unknown.

    Arch-specific tasks (sm_90a / sm_100a / ...) are built with their own -gencode, not with the
    plain native `sm_{sm}` target, and some need e.g. -std=c++20 or --expt-extended-lambda.
    """
    path = os.path.join(KH_ROOT, config.runner.workdir or "", "Makefile")
    if not os.path.isfile(path):
        return "", ""
    found = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^\s*(NVCC_FLAGS|ARCH)\s*[:+?]?=\s*(.*?)\s*$", line)
        if m and m.group(1) not in found:
            found[m.group(1)] = m.group(2)
    return found.get("NVCC_FLAGS", ""), found.get("ARCH", "")


def _sandbox_notes(config, gpu_name: str, sm: int) -> str:
    sol = config.runner.solution_file
    flags = "-O3 --use_fast_math --expt-relaxed-constexpr -std=c++17"
    arch = f"native `sm_{sm}` code plus PTX"
    if config.runner.backend == "class2_defpy":
        flags += (" --expt-extended-lambda -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__"
                  " -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        how = f"`{sol}` is compiled as a PyTorch extension behind an auto-generated pybind11 wrapper."
    else:
        how = f"`{sol}` is included by the judge's private test harness."
        if config.runner.backend == "class1_make" and "/general/" not in config.task_id:
            mk_flags, mk_arch = _makefile_build_flags(config)
            if mk_flags:
                flags = mk_flags
            if mk_arch:
                arch = f"`{mk_arch}`"
                if re.search(r"(?:sm|compute)_\d+a\b", mk_arch):
                    arch += (" (architecture-specific target: instructions such as `tcgen05.*` only assemble "
                             "for the `a` variant; do not build with plain `-arch=sm_%d`)" % sm)
    return f"""### Sandbox notes

- Write your solution to `/workspace/{sol}`. {how}
  TASK.md was written for single-shot generation: ignore its "Output Format"
  section. The deliverable is the file in /workspace, not a code block in a reply.
- The judge's harness, the reference implementation and any `make test` /
  `./test` commands mentioned in TASK.md are NOT available in this workspace.
  The only way to run them is `submit`.
- Judge nvcc flags: `{flags}`, target {arch}, and
  `-I/opt/cutlass/include -I/opt/cutlass/tools/util/include -I/opt/cutlass/examples/common`.
- Tools: `bash` (each call is a fresh shell in /workspace: only files under
  /workspace persist; /tmp and background processes do not survive between
  calls), `read_file`, `write_file`, `edit_file`.
- Local toolchain: CUDA 13.0 (`nvcc`, `ncu`, `nsys`, `compute-sanitizer`) in
  /usr/local/cuda, gcc 11, python3 3.10 with numpy, git. No PyTorch. The
  `ncu_report` Python module is on PYTHONPATH.
- Profiling guide: `/opt/kda/skills/ncu-report-skill/SKILL.md` (ncu collection
  recipes, harness template, `ncu_report` API, diagnosis playbook). It was
  written for B200; metric-name notes may differ on sm_{sm}.
- GPU: one {gpu_name} (sm_{sm}) for your own builds, tests and profiling.
  The judge times submissions on the same GPU.
- No network access.
"""


def build_kda_prompt(config, task_md: str, args, gpu_name: str, sm: int) -> str:
    template = open(os.path.join(KDA_LITE, "autonomous-flow.md")).read()
    sol = config.runner.solution_file
    budget = [f"- Wall-clock time: {args.time_limit_min:.0f} minutes from session start.",
              f"- Evaluations: {args.max_evals} calls of `submit` (with or without `--check`) in total."]
    if args.max_total_tokens > 0:
        budget.append(f"- Model tokens: at most {args.max_total_tokens:,} (input + output) for this session.")
    fields = {
        "task_name": config.task_id,
        "objective": (f"implement the operation specified in `TASK.md` in `/workspace/{sol}`, "
                      "correct and as fast as possible on the target GPU. The reference it is "
                      "compared against is an expert-optimized implementation."),
        "correctness": ("outputs must match the reference within the task's tolerances on every "
                        "tested problem size, as checked by the judge; the library restrictions "
                        "in `TASK.md` are enforced by a static check before anything runs. "
                        "Keep the numerical precision the task specifies: never compute in a "
                        "lower-precision type than the task's data/compute types (e.g. FP16/BF16 "
                        "on a TF32 task), even if the result still passes the tolerance; such "
                        "solutions are scored as failures."),
        "target": "maximize speedup = reference time / solution time, as measured by the judge.",
        "allowed": (f"a single CUDA C++ file `{sol}` (raw CUDA, tensor-core intrinsics, inline "
                    "PTX, CuTe and CUTLASS low-level building blocks); see the library "
                    "restrictions in `TASK.md`."),
        "validation_command": "submit --check",
        "evaluation_command": "submit",
        "promotion_criteria": (f"a timed `submit` result of PASS with speedup >= "
                               f"{args.target_speedup:.2f}x versus the reference."),
        "deliverable": (f"keep your best validated candidate in `/workspace/{sol}`. The final score "
                        "is a fresh re-evaluation of your fastest timed PASS submission; failed "
                        "submissions are never scored."),
        "budget": "\n".join(budget),
        "budget_status_command": "submit --status",
        "task_details": ("The full task specification is in `/workspace/TASK.md` (reproduced "
                         "below).\n\n" + task_md + "\n\n" + _sandbox_notes(config, gpu_name, sm)),
    }
    for k, v in fields.items():
        template = template.replace("{{" + k + "}}", v)
    assert "{{" not in template, "unfilled placeholder in KDA prompt"
    return template


def make_workspace(config, ws: str, task_md: str) -> None:
    os.makedirs(ws, exist_ok=True)
    for d in ("docs", "runs", "outputs", "profile"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    with open(os.path.join(ws, "TASK.md"), "w") as f:
        f.write(task_md)
    if config.runner.backend == "class1_make":
        # The naive solution.h is already shown in the prompt's template block.
        src = os.path.join(KH_ROOT, config.runner.workdir, config.runner.solution_file)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ws, config.runner.solution_file))


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------

def _wait_for(path: str, proc: subprocess.Popen, timeout: float, what: str) -> None:
    t0 = time.time()
    while not os.path.exists(path):
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited early (code {proc.returncode})")
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{what} did not come up in {timeout}s")
        time.sleep(0.2)


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _final_reeval(ea, config, task_out: str, eval_timeout: int) -> dict:
    """Re-evaluate submissions (fastest in-loop timed PASS first, then
    check-only PASSes, newest first); the first that passes again is final."""
    log = os.path.join(task_out, "submissions.jsonl")
    subs = [json.loads(l) for l in open(log)] if os.path.isfile(log) else []
    timed = sorted([s for s in subs if s.get("status") == "PASS" and s["mode"] == "full"],
                   key=lambda s: -s.get("speedup", 0.0))
    checked = sorted([s for s in subs if s.get("status") == "PASS" and s["mode"] == "check"],
                     key=lambda s: -s["index"])
    sol = config.runner.solution_file
    tried = []
    for s in timed + checked:
        path = os.path.join(task_out, "submissions", f"{s['index']:03d}", sol)
        tr = ea._run_task_isolated(config, path, timeout_sec=eval_timeout, measure_perf=True)
        rec = {"index": s["index"], "in_loop_speedup": s.get("speedup", 0.0),
               "status": tr.status.value, "speedup": tr.speedup,
               "latency_mean_ms": tr.latency_mean_ms, "ref_latency_mean_ms": tr.ref_latency_mean_ms,
               "per_size": tr.per_size or [], "error": (tr.error_msg or "")[:1000]}
        tried.append(rec)
        if tr.correct:
            shutil.copy2(path, os.path.join(task_out, f"best_{sol}"))
            return {"final": rec, "tried": tried}
        if len(tried) >= 3:
            break
    return {"final": None, "tried": tried}


def run_one(config, gpu: int, args, ea) -> dict:
    from openai import OpenAI
    from kda_eval.agent_loop import AgentLoop

    run_dir = args.run_dir
    task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
    if os.path.isdir(task_out):
        shutil.rmtree(task_out)  # incomplete earlier attempt (no result.json)
    ws = os.path.join(task_out, "workspace")
    opt_kh = os.path.join(task_out, "opt_kh")
    sock_dir = tempfile.mkdtemp(prefix="kda_sock_")  # short path: AF_UNIX limit
    judge_sock = os.path.join(sock_dir, "judge.sock")

    import torch
    gpu_name = torch.cuda.get_device_name(0)
    sm = int("".join(map(str, torch.cuda.get_device_capability(0))))

    task_md = ea.build_task_prompt(config)
    make_workspace(config, ws, task_md)
    prompt = build_kda_prompt(config, task_md, args, gpu_name, sm)
    with open(os.path.join(task_out, "prompt.md"), "w") as f:
        f.write(prompt)
    sbx.install_opt_kh(opt_kh)

    api_base, api_key = load_api_config(args)
    time_limit_s = args.time_limit_min * 60
    result = {
        "task_id": config.task_id, "task_class": config.task_class, "domain": config.domain,
        "model": args.model, "agent": "kda-lite-api", "gpu": gpu,
        "final_status": "FAIL", "best_speedup": 0.0, "best_solution_file": "",
        "budget": {"time_limit_min": args.time_limit_min, "max_evals": args.max_evals,
                   "max_total_tokens": args.max_total_tokens, "target_speedup": args.target_speedup},
    }
    judge = None
    try:
        judge_env = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(task_out, ".torch_ext"))
        judge = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "kda_eval", "judge.py"),
             "--task-id", config.task_id, "--socket", judge_sock,
             "--run-dir", task_out, "--max-evals", str(args.max_evals),
             "--time-limit", str(time_limit_s), "--eval-timeout", str(args.eval_timeout),
             "--target-speedup", str(args.target_speedup)],
            stdout=open(os.path.join(task_out, "judge.log"), "w"), stderr=subprocess.STDOUT,
            env=judge_env, start_new_session=True)
        _wait_for(judge_sock, judge, 300, "judge")
        deadline = time.time() + time_limit_s  # the judge's clock started slightly earlier

        spec = sbx.SandboxSpec(
            workspace=ws, sock_dir=sock_dir, gpu_index=gpu, opt_kh=opt_kh,
            ro_binds=[(NCU_SKILL, "/opt/kda/skills/ncu-report-skill")],
            env={"KH_SOLUTION_FILE": config.runner.solution_file})
        loop = AgentLoop(
            OpenAI(base_url=api_base, api_key=api_key, timeout=900, max_retries=0),
            args.model, spec, task_out, judge_sock, deadline,
            max_completion_tokens=args.max_completion_tokens, context_limit=args.context_limit,
            max_llm_calls=args.max_llm_calls, max_total_tokens=args.max_total_tokens,
            reasoning_effort=args.reasoning_effort)
        result["session"] = loop.run(prompt)
    except Exception as e:
        result["final_status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        _stop(judge)
        shutil.rmtree(sock_dir, ignore_errors=True)

    state_path = os.path.join(task_out, "judge_state.json")
    if os.path.isfile(state_path):
        st = json.load(open(state_path))
        result.update(evals_used=st["evals_used"], in_loop_best_speedup=st["best_speedup"],
                      in_loop_best_submission=st["best_submission"])
    log = os.path.join(task_out, "submissions.jsonl")
    subs = [json.loads(l) for l in open(log)] if os.path.isfile(log) else []
    result["first_compiled_eval"] = next((s["index"] for s in subs if s.get("compiled")), -1)
    result["first_pass_eval"] = next((s["index"] for s in subs if s.get("status") == "PASS"), -1)

    if result["final_status"] != "ERROR":
        fin = _final_reeval(ea, config, task_out, args.eval_timeout)
        result["final_reeval"] = fin
        if fin["final"]:
            result["final_status"] = "PASS"
            result["best_speedup"] = fin["final"]["speedup"]
            result["best_solution_file"] = f"best_{config.runner.solution_file}"
            sol_dst = os.path.join(run_dir, "solutions", config.task_id)
            os.makedirs(sol_dst, exist_ok=True)
            shutil.copy2(os.path.join(task_out, result["best_solution_file"]),
                         os.path.join(sol_dst, config.runner.solution_file))

    with open(os.path.join(task_out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def _worker(gpu: int, task_ids: "mp.Queue", args) -> None:
    import ctypes
    # Die with the orchestrator; on SIGTERM unwind so run_one's finally stops the judge.
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import eval_agent as ea
    from cuda_hercules.runner import discover_tasks
    from cuda_hercules.utils import get_project_root
    configs = {t.task_id: t for t in discover_tasks(get_project_root())}
    while True:
        tid = task_ids.get()
        if tid is None:
            return
        t0 = time.time()
        print(f"[gpu{gpu}] START {tid}", flush=True)
        try:
            res = run_one(configs[tid], gpu, args, ea)
        except Exception as e:
            res = {"task_id": tid, "final_status": "ERROR", "best_speedup": 0.0,
                   "error": f"{type(e).__name__}: {e}"}
        sess = res.get("session", {})
        print(f"[gpu{gpu}] DONE  {tid}: {res['final_status']} {res.get('best_speedup', 0):.3f}x "
              f"evals={res.get('evals_used', '?')} llm_calls={sess.get('llm_calls', '?')} "
              f"tokens={sess.get('prompt_tokens', 0) + sess.get('completion_tokens', 0):,} "
              f"wall={(time.time() - t0) / 60:.1f}min end={sess.get('end_reason', '')} "
              f"{res.get('error', '')}", flush=True)


def audit(run_dir: str, tasks: list) -> dict:
    """Reporting-layer hack audit; result.json and all artifacts stay untouched.

    A task is flagged when its final solution fails the task's *current* static
    check (so patterns added after a run, e.g. precision floors, apply
    retroactively) or when `<run_dir>/_audit.json` flags it by hand
    ({task_id: {"hack": true, "reason": "..."}}).
    """
    from cuda_hercules.static_checker import validate_cuda_solution
    manual_path = os.path.join(run_dir, "_audit.json")
    manual = json.load(open(manual_path)) if os.path.isfile(manual_path) else {}
    flags = {}
    for t in tasks:
        tdir = os.path.join(run_dir, t.task_id.replace("/", "_"))
        rp = os.path.join(tdir, "result.json")
        res = json.load(open(rp)) if os.path.isfile(rp) else {}
        reasons = []
        sol = os.path.join(tdir, res.get("best_solution_file") or "")
        if res.get("final_status") == "PASS" and os.path.isfile(sol):
            check = validate_cuda_solution(open(sol, errors="replace").read(),
                                           blocked_patterns=t.anti_cheat.blocked_patterns or [],
                                           required_patterns=t.anti_cheat.required_patterns or [])
            if not check.valid:
                reasons.append("static check (current task.yaml): " + "; ".join(check.errors))
        m = manual.get(t.task_id) or {}
        if m.get("hack"):
            reasons.append(m.get("reason", "flagged in _audit.json"))
        if reasons:
            flags[t.task_id] = reasons
    return flags


def summarize(run_dir: str, tasks: list) -> dict:
    flags = audit(run_dir, tasks)
    rows = []
    for t in tasks:
        tid = t.task_id
        p = os.path.join(run_dir, tid.replace("/", "_"), "result.json")
        r = (json.load(open(p)) if os.path.isfile(p)
             else {"task_id": tid, "final_status": "MISSING", "best_speedup": 0.0})
        r["hack"] = tid in flags
        rows.append(r)
    n = len(rows)
    raw_pass = [r for r in rows if r["final_status"] == "PASS"]
    # Hacks count as FAIL in the headline numbers.
    passed = [r for r in raw_pass if not r["hack"]]
    sess = [r.get("session") or {} for r in rows]
    summary = {
        "num_tasks": n,
        "num_pass": len(passed),
        "pass_rate": len(passed) / n if n else 0.0,
        # Arithmetic mean over all tasks, failures count as 0.
        "mean_speedup_all": sum(r["best_speedup"] for r in passed) / n if n else 0.0,
        "num_ge_1x": sum(1 for r in passed if r["best_speedup"] >= 1.0),
        "num_hack": len(flags),
        "num_pass_raw": len(raw_pass),
        "mean_speedup_all_raw": sum(r["best_speedup"] for r in raw_pass) / n if n else 0.0,
        "prompt_tokens": sum(s.get("prompt_tokens", 0) for s in sess),
        "completion_tokens": sum(s.get("completion_tokens", 0) for s in sess),
        "tasks": [dict({k: r.get(k) for k in ("task_id", "final_status", "best_speedup", "evals_used",
                                              "in_loop_best_speedup", "hack")},
                       **({"hack_reasons": flags[r["task_id"]]} if r["hack"] else {}))
                  for r in rows],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model alias on the API endpoint (e.g. claude-opus-5.5)")
    ap.add_argument("--env-file", default=os.path.join(KH_ROOT, ".env"), help="File holding the API key/base")
    ap.add_argument("--api-key-env", default="AWS_OPENAI_API_KEY")
    ap.add_argument("--api-base-env", default="AWS_OPENAI_BASE_URL")
    ap.add_argument("--api-base", default="", help="Overrides --api-base-env")
    ap.add_argument("--reasoning-effort", default="", help="Passed through as reasoning_effort if set")
    ap.add_argument("--max-completion-tokens", type=int, default=32000, help="Per LLM call")
    ap.add_argument("--context-limit", type=int, default=150000,
                    help="Elide old tool outputs once a prompt exceeds this many tokens")
    ap.add_argument("--gpus", required=True, help="Comma-separated nvidia-smi indices, one task per GPU at a time")
    ap.add_argument("--filter", action="append", default=[])
    ap.add_argument("--task-list", default="")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--time-limit-min", type=float, default=60.0, help="Wall-clock budget per task")
    ap.add_argument("--max-evals", type=int, default=10, help="Judge evaluations per task")
    ap.add_argument("--max-llm-calls", type=int, default=1000, help="LLM calls per task")
    ap.add_argument("--max-total-tokens", type=int, default=0, help="Input+output tokens per task (0 = none)")
    ap.add_argument("--target-speedup", type=float, default=1.0, help="KDA promotion threshold")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Re-run the hack audit and rewrite summary.json without running tasks")
    ap.add_argument("--eval-timeout", type=int, default=900, help="Per-evaluation timeout (s)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    from cuda_hercules.runner import discover_tasks, get_gpu_sm, parse_filters
    from cuda_hercules.utils import get_project_root

    if not os.path.isfile(os.path.join(NCU_SKILL, "SKILL.md")):
        sys.exit("kda/skills/ncu-report-skill is missing: run `git submodule update --init kda && "
                 "git -C kda submodule update --init skills/ncu-report-skill`")
    if not args.summarize_only:
        load_api_config(args)  # fail fast

    tasks = discover_tasks(get_project_root(), parse_filters(args.filter))
    gpu_sm = get_gpu_sm()
    skipped = [t.task_id for t in tasks if t.runner.backend not in SUPPORTED_BACKENDS]
    tasks = [t for t in tasks if t.runner.backend in SUPPORTED_BACKENDS and t.hardware.min_sm <= gpu_sm]
    if args.task_list:
        wanted = {l.strip() for l in open(args.task_list) if l.strip() and not l.startswith("#")}
        tasks = [t for t in tasks if t.task_id in wanted or t.task_id.split("/")[-1] in wanted]
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        sys.exit("No tasks selected.")
    if skipped:
        print(f"Skipping {len(skipped)} task(s) with unsupported backends (class3/class4 not wired yet).")

    args.run_dir = os.path.abspath(os.path.join(args.output, args.run_name or f"kda_lite_{args.model}"))
    os.makedirs(args.run_dir, exist_ok=True)
    if args.summarize_only:
        _report(summarize(args.run_dir, tasks))
        return
    kda_rev = subprocess.run(["git", "-C", KDA_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), gpu_sm=gpu_sm, num_tasks=len(tasks), kda_rev=kda_rev,
                       task_ids=[t.task_id for t in tasks]), f, indent=2)

    todo = [t.task_id for t in tasks
            if not os.path.isfile(os.path.join(args.run_dir, t.task_id.replace("/", "_"), "result.json"))]
    print(f"KDA-lite (API) on CUDA-Hercules: {len(tasks)} tasks ({len(tasks) - len(todo)} already done), "
          f"model={args.model}, GPUs={gpus}, budget={args.time_limit_min:.0f}min/{args.max_evals} evals, "
          f"out={args.run_dir}", flush=True)

    ctx = mp.get_context("spawn")
    q_in = ctx.Queue()
    for tid in todo:
        q_in.put(tid)
    for _ in gpus:
        q_in.put(None)
    procs = [ctx.Process(target=_worker, args=(g, q_in, args)) for g in gpus]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    _report(summarize(args.run_dir, tasks))


def _report(s: dict) -> None:
    print(f"\nPASS {s['num_pass']}/{s['num_tasks']} ({s['pass_rate']:.1%}, {s['num_hack']} hack(s) as FAIL), "
          f"mean speedup (all, fail=0) {s['mean_speedup_all']:.3f}x, >=1x: {s['num_ge_1x']}, "
          f"tokens in/out {s['prompt_tokens']:,}/{s['completion_tokens']:,}")


if __name__ == "__main__":
    main()
