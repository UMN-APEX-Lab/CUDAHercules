#!/usr/bin/env python3
"""Re-time every earlier run's best solution under the current task harness.

For each task and each experiment directory results/<run>/<task_id with '/'→'_'>/,
pick that run's best correct solution (by the speedup the run recorded), re-check it
with the current static rules, and re-run it with the current harness (problem sizes,
verification, per-task geomean over sizes). No LLM calls. One JSON per task is written
to --out, raw run directories are never modified.

  python scripts/retime_best_solutions.py --gpus 4,6 --out results/retime_realistic_sizes \\
      --tasks class1/general/05_batched_gemm class2/general/fft_c2c_1d_1024_fp32
  python scripts/retime_best_solutions.py --gpus 4 --task-list tasks.txt --out ... --dry-run
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RESULTS = os.path.join(ROOT, "results")
SKIP_RUN_PREFIXES = ("retime_",)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _first_existing(*paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def find_best(task_dir: str, sol_name: str):
    """Return (solution_path, recorded_speedup, source) of the run's best correct solution."""
    cands = []
    res = _load(os.path.join(task_dir, "result_reeval.json")) or _load(os.path.join(task_dir, "result.json")) or {}
    rep = _load(os.path.join(task_dir, "final_report.json")) or {}

    for rd in res.get("rounds") or []:
        if rd.get("correct"):
            f = _first_existing(os.path.join(task_dir, rd["solution_file"]) if rd.get("solution_file") else None,
                                os.path.join(task_dir, f"r{rd['round']}_{sol_name}"))
            cands.append((f, rd.get("speedup") or 0.0, f"round {rd['round']}"))
    for h in rep.get("history") or []:
        if h.get("correct") and h.get("solution_file"):
            cands.append((_first_existing(os.path.join(task_dir, h["solution_file"])),
                          h.get("speedup") or 0.0, f"round {h.get('round')}"))
    for s in res.get("sample_details") or []:
        if s.get("correct"):
            i = s["sample"]
            f = _first_existing(os.path.join(task_dir, f"sample_{i}", sol_name),
                                os.path.join(task_dir, f"sample_{i}_{sol_name}"))
            cands.append((f, s.get("speedup") or 0.0, f"sample {i}"))
    for r in (res, rep):  # KDA / tool-aug best_solution_file
        if r.get("best_solution_file") and (r.get("best_speedup") or 0) > 0 and r.get("final_status", "PASS") == "PASS":
            cands.append((_first_existing(os.path.join(task_dir, r["best_solution_file"])),
                          r["best_speedup"], "best_solution_file"))
    if not cands and ((res.get("num_correct") or 0) > 0 or res.get("correct") is True):  # single-file runs
        cands.append((_first_existing(os.path.join(task_dir, sol_name)),
                      res.get("best_speedup") or res.get("speedup") or 0.0, "solution"))

    cands = [c for c in cands if c[0]]
    return max(cands, key=lambda c: c[1]) if cands else None


def collect(task_ids, configs):
    jobs = []
    for tid in task_ids:
        cfg = configs[tid]
        sol_name = cfg.runner.solution_file
        for d in sorted(glob.glob(os.path.join(RESULTS, "**", tid.replace("/", "_")), recursive=True)):
            run = os.path.relpath(os.path.dirname(d), RESULTS)
            if run.startswith(SKIP_RUN_PREFIXES) or "/solutions/" in d or not os.path.isdir(d):
                continue
            best = find_best(d, sol_name)
            jobs.append({"task_id": tid, "run": run, "solution": os.path.relpath(best[0], ROOT) if best else None,
                         "recorded_speedup": best[1] if best else 0.0, "source": best[2] if best else "no correct solution"})
    return jobs


def _worker(gpu, q_in, q_out, torch_ext_root):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(torch_ext_root, f"gpu{gpu}")
    from cuda_hercules.runner import discover_tasks
    from cuda_hercules.static_checker import validate_cuda_solution
    from cuda_hercules.utils import get_project_root
    import eval_agent as ea
    configs = {t.task_id: t for t in discover_tasks(get_project_root(), [])}
    while True:
        job = q_in.get()
        if job is None:
            return
        cfg = configs[job["task_id"]]
        path = os.path.join(ROOT, job["solution"])
        code = open(path, errors="replace").read()
        chk = validate_cuda_solution(code, blocked_patterns=cfg.anti_cheat.blocked_patterns or [],
                                     required_patterns=cfg.anti_cheat.required_patterns or [])
        t0 = time.time()
        tr = ea._run_task_isolated(cfg, path, timeout_sec=900, measure_perf=True)
        status = tr.status.value if tr.status else "ERROR"
        q_out.put(dict(job, gpu=gpu, static_ok=chk.valid, static_errors=chk.errors, run_status=status,
                       final_status="PASS" if status == "PASS" and chk.valid else "FAIL",
                       speedup=tr.speedup if status == "PASS" and chk.valid else 0.0,
                       raw_speedup=tr.speedup, per_size=tr.per_size or [],
                       latency_ms=tr.latency_mean_ms, ref_latency_ms=tr.ref_latency_mean_ms,
                       error=(tr.error_msg or "")[:1500], wall_s=round(time.time() - t0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--task-list", default="")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Only list the selected best solutions")
    args = ap.parse_args()

    from cuda_hercules.runner import discover_tasks
    from cuda_hercules.utils import get_project_root
    task_ids = list(args.tasks)
    if args.task_list:
        task_ids += [l.strip() for l in open(args.task_list) if l.strip() and not l.startswith("#")]
    configs = {t.task_id: t for t in discover_tasks(get_project_root(), [])}
    missing = [t for t in task_ids if t not in configs]
    if missing:
        sys.exit(f"unknown task ids: {missing}")

    jobs = collect(task_ids, configs)
    runnable = [j for j in jobs if j["solution"]]
    print(f"{len(task_ids)} tasks, {len(jobs)} run dirs, {len(runnable)} with a correct best solution", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['task_id']:55s} {j['run']:45s} {j['recorded_speedup']:.3f} {j['source']}")
        return

    os.makedirs(args.out, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    # Slow (recorded speedup < 0.05) solutions last so a few stragglers don't block the rest.
    for j in sorted(runnable, key=lambda j: j["recorded_speedup"] < 0.05):
        q_in.put(j)
    for _ in gpus:
        q_in.put(None)
    torch_ext_root = os.path.join(os.path.abspath(args.out), ".torch_ext")
    procs = [ctx.Process(target=_worker, args=(g, q_in, q_out, torch_ext_root)) for g in gpus]
    for p in procs:
        p.start()

    by_task = {t: [dict(j, final_status="FAIL", speedup=0.0) for j in jobs if j["task_id"] == t and not j["solution"]]
               for t in task_ids}
    for i in range(len(runnable)):
        r = q_out.get()
        by_task[r["task_id"]].append(r)
        print(f"[{i + 1}/{len(runnable)}] {r['task_id']:50s} {r['run']:42s} {r['final_status']} "
              f"{r['speedup']:.3f} (recorded {r['recorded_speedup']:.3f}) {r['error'][:80]!r}", flush=True)
        with open(os.path.join(args.out, r["task_id"].replace("/", "_") + ".json"), "w") as f:
            json.dump({"task_id": r["task_id"], "task_speedup": "geometric mean over sizes",
                       "solutions": by_task[r["task_id"]]}, f, indent=1)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
