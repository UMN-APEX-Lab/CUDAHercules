#!/usr/bin/env python3
"""Probe candidate formulas from the Zenodo 5138008 benchmark and pick a 10-instance subset.

Strategy:
  1. Stratify by compressed file size into 4 bins
  2. Sample a fixed number from each bin
  3. For each sampled formula, decompress to /tmp, run ParaFROST with a timeout, parse
     verdict + Simplifier/Solver time
  4. Classify by total reference time into {small, medium, large} and select
     6 small (5-30s) + 2 medium (100-500s) + 2 large (500-1800s)
  5. Emit subset_j2021_v1.json with final selection; this script must be reproducible

Parallelism: probe runs on 4 GPUs in parallel (CUDA_VISIBLE_DEVICES=2,3,6,7 by default).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAFROST_BIN = os.path.join(ROOT, "src", "parafrost")
XZ_DIR_DEFAULT = "/mnt/data2/li004074/parafrost_benchmarks/xz_files"
OUT_DIR_DEFAULT = os.path.join(ROOT, "subset_j2021_v1")


@dataclass
class ProbeResult:
    name: str
    compressed_size: int
    ok: bool = False
    verdict: str = ""
    simp_sec: float = -1.0
    solve_sec: float = -1.0
    total_sec: float = -1.0
    error: str = ""


def _parse_pf_output(text: str) -> dict:
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi.sub("", text)
    sat = "SATISFIABLE" in clean and "UNSATISFIABLE" not in clean
    unsat = "UNSATISFIABLE" in clean
    verdict = "SAT" if sat else ("UNSAT" if unsat else "UNKNOWN")
    m = re.search(r"Simplifier time\s*:\s*([0-9.]+)", clean)
    simp_sec = float(m.group(1)) if m else 0.0
    m = re.search(r"Solver time\s*:\s*([0-9.]+)", clean)
    solve_sec = float(m.group(1)) if m else 0.0
    return {"verdict": verdict, "simp_sec": simp_sec, "solve_sec": solve_sec}


def _probe_one(args: tuple) -> ProbeResult:
    xz_path, gpu_id, timeout_sec = args
    name = os.path.basename(xz_path).replace(".cnf.xz", "")
    size = os.path.getsize(xz_path)
    res = ProbeResult(name=name, compressed_size=size)

    tmp_dir = tempfile.mkdtemp(prefix="pf_probe_")
    try:
        cnf_path = os.path.join(tmp_dir, f"{name}.cnf")
        with open(cnf_path, "wb") as out:
            p = subprocess.Popen(
                ["xz", "-dc", xz_path], stdout=out, stderr=subprocess.PIPE
            )
            _, err = p.communicate(timeout=600)
            if p.returncode != 0:
                res.error = f"xz failed: {err.decode(errors='replace')[:200]}"
                return res

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [PARAFROST_BIN, cnf_path, "--timeout=" + str(timeout_sec)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_sec + 30,
            )
            wall = time.time() - t0
            parsed = _parse_pf_output(proc.stdout + proc.stderr)
            res.verdict = parsed["verdict"]
            res.simp_sec = parsed["simp_sec"]
            res.solve_sec = parsed["solve_sec"]
            res.total_sec = parsed["simp_sec"] + parsed["solve_sec"]
            if res.total_sec <= 0 and res.verdict != "UNKNOWN":
                res.total_sec = wall  # fall back to wall if parafrost didn't print times
            res.ok = res.verdict in ("SAT", "UNSAT")
        except subprocess.TimeoutExpired:
            res.verdict = "TIMEOUT"
            res.total_sec = float(timeout_sec)
            res.error = f"ParaFROST timeout after {timeout_sec}s"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return res


def _stratify(files_with_sizes: list[tuple[str, int]], seed: int) -> dict[str, list]:
    bins = {"xs": [], "s": [], "m": [], "l": [], "xl": []}
    for name, sz in files_with_sizes:
        if sz < 1_000_000:
            bins["xs"].append((name, sz))
        elif sz < 5_000_000:
            bins["s"].append((name, sz))
        elif sz < 20_000_000:
            bins["m"].append((name, sz))
        elif sz < 50_000_000:
            bins["l"].append((name, sz))
        else:
            bins["xl"].append((name, sz))
    return bins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xz-dir", default=XZ_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument(
        "--gpus", default="2,3,6,7", help="Comma-separated CUDA_VISIBLE_DEVICES to rotate across"
    )
    ap.add_argument("--timeout", type=int, default=1800, help="Per-probe ParaFROST timeout in seconds")
    ap.add_argument(
        "--per-bin", type=int, default=8,
        help="Formulas to probe per size bin (xs=skip, s/m/l/xl)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-results", default="", help="Reuse a JSON of probe results if present")
    args = ap.parse_args()

    random.seed(args.seed)

    xz_files = []
    for f in sorted(os.listdir(args.xz_dir)):
        if f.endswith(".cnf.xz"):
            p = os.path.join(args.xz_dir, f)
            xz_files.append((p, os.path.getsize(p)))
    print(f"Found {len(xz_files)} .cnf.xz files under {args.xz_dir}")

    bins = _stratify(xz_files, args.seed)
    for k, v in bins.items():
        print(f"  bin {k}: {len(v)} files")

    # Probe: sample per bin, skip xs (paper excludes formulas < 5MB uncompressed ≈ 1.7MB comp)
    sampled = []
    for bin_key in ("s", "m", "l", "xl"):
        pool = bins[bin_key]
        if not pool:
            continue
        take = min(args.per_bin, len(pool))
        picked = random.sample(pool, take)
        sampled.extend([(p, sz, bin_key) for p, sz in picked])
    print(f"Probing {len(sampled)} formulas (per-bin={args.per_bin}, timeout={args.timeout}s)")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    probe_results: list[ProbeResult] = []

    # Parallel over GPUs using ProcessPoolExecutor
    jobs = [(p, gpus[i % len(gpus)], args.timeout) for i, (p, _, _) in enumerate(sampled)]
    os.makedirs(args.out_dir, exist_ok=True)
    probe_cache = os.path.join(args.out_dir, "probe_results.json")

    if args.probe_results and os.path.isfile(args.probe_results):
        probe_cache = args.probe_results
    if os.path.isfile(probe_cache):
        print(f"Loading cached probe results from {probe_cache}")
        probe_results = [ProbeResult(**d) for d in json.load(open(probe_cache))]
    else:
        with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(_probe_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                probe_results.append(r)
                print(
                    f"[{i+1}/{len(futures)}] {r.name[:60]:60s}  "
                    f"size={r.compressed_size/1024/1024:5.1f}MB  verdict={r.verdict:7s}  "
                    f"simp={r.simp_sec:7.2f}s  solve={r.solve_sec:7.2f}s  total={r.total_sec:7.2f}s"
                    f"{'  '+r.error if r.error else ''}",
                    flush=True,
                )
                # Incremental cache write — if the process is killed mid-run,
                # we don't lose work.
                with open(probe_cache, "w") as f:
                    json.dump([asdict(x) for x in probe_results], f, indent=2)
        print(f"Wrote {probe_cache}")

    # Classify
    def _tier(r: ProbeResult) -> str:
        if not r.ok or r.total_sec <= 0:
            return "fail"
        if 5 <= r.total_sec < 30:
            return "small"
        if 30 <= r.total_sec < 100:
            return "mid_low"
        if 100 <= r.total_sec < 500:
            return "medium"
        if 500 <= r.total_sec <= 1800:
            return "large"
        return "out_of_range"

    by_tier: dict[str, list[ProbeResult]] = {}
    for r in probe_results:
        by_tier.setdefault(_tier(r), []).append(r)
    print()
    for t, rs in by_tier.items():
        print(f"Tier {t}: {len(rs)} candidates")
        for r in sorted(rs, key=lambda x: x.total_sec)[:20]:
            print(
                f"  {r.name[:60]:60s}  size={r.compressed_size/1024/1024:5.1f}MB  "
                f"v={r.verdict:7s}  total={r.total_sec:7.2f}s"
            )

    # Target: 6 small + 2 medium + 2 large, with SAT/UNSAT mix in medium + large
    def _pick(tier: str, n: int, prefer_mixed: bool) -> list[ProbeResult]:
        pool = sorted(by_tier.get(tier, []), key=lambda x: x.total_sec)
        if not prefer_mixed:
            return pool[:n]
        sats = [r for r in pool if r.verdict == "SAT"]
        unsats = [r for r in pool if r.verdict == "UNSAT"]
        out: list[ProbeResult] = []
        si = ui = 0
        while len(out) < n and (si < len(sats) or ui < len(unsats)):
            if si < len(sats) and (len(out) % 2 == 0 or ui >= len(unsats)):
                out.append(sats[si]); si += 1
            elif ui < len(unsats):
                out.append(unsats[ui]); ui += 1
        return out[:n]

    picked_small = _pick("small", 6, prefer_mixed=True)
    picked_medium = _pick("medium", 2, prefer_mixed=True)
    picked_large = _pick("large", 2, prefer_mixed=True)

    # Fallback: if a tier is undershot, widen range (pull from mid_low / out_of_range)
    if len(picked_small) < 6:
        extras = sorted(by_tier.get("mid_low", []), key=lambda x: x.total_sec)
        picked_small += extras[: 6 - len(picked_small)]
    if len(picked_large) < 2:
        extras = [r for r in by_tier.get("out_of_range", []) if 1800 < r.total_sec < 3600]
        extras.sort(key=lambda x: x.total_sec)
        picked_large += extras[: 2 - len(picked_large)]

    final = picked_small + picked_medium + picked_large

    print(f"\nFinal subset ({len(final)}):")
    total_ref = 0.0
    manifest = {
        "version": 1,
        "format": "dimacs-cnf",
        "source": "Zenodo 5138008 - Large SAT Benchmark Suite (Osama/Wijs/Biere FMSD'24)",
        "timeout_sec_per_instance": 1800,
        "instances": [],
    }
    for r in final:
        tier = _tier(r)
        print(
            f"  [{tier:7s}] {r.name[:60]:60s}  size={r.compressed_size/1024/1024:5.1f}MB  "
            f"v={r.verdict:7s}  total={r.total_sec:7.2f}s"
        )
        total_ref += r.total_sec
        manifest["instances"].append(
            {
                "name": r.name,
                "file": os.path.join("instances", r.name + ".cnf"),
                "xz_source": os.path.relpath(
                    os.path.join(args.xz_dir, r.name + ".cnf.xz"),
                    start=os.path.dirname(OUT_DIR_DEFAULT),
                ),
                "compressed_size": r.compressed_size,
                "reference_verdict": r.verdict,
                "reference_time_sec": r.total_sec,
                "tier": tier,
            }
        )
    print(f"\nTotal reference time: {total_ref:.1f} s ({total_ref/60:.1f} min)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "subset_j2021_v1.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {os.path.join(args.out_dir, 'subset_j2021_v1.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
