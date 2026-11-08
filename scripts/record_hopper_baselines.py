#!/usr/bin/env python3
"""Publish source-matched Hopper validation evidence and H200 timing caches.

Run the three Hopper validation scripts first. This command refuses incomplete,
failed, or stale results. It does not run kernels or modify task definitions.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from cuda_hercules.utils import task_source_hash

ROOT = Path(__file__).resolve().parents[1]


def read_results(path):
    return json.loads((ROOT / path).read_text())


def require_current(record, task):
    if not record.get("passed"):
        raise ValueError(f"Failed validation: {task}")
    if record.get("source_hash") != task_source_hash(str(task)):
        raise ValueError(f"Stale validation: {task}; rerun with --resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make", default="results/hopper_make_verified.json")
    parser.add_argument("--fa3", default="results/hopper_fa3_verified.json")
    parser.add_argument("--tk", default="results/hopper_tk_oracles.json")
    parser.add_argument("--splits", default="results/hopper_fa3_splits_long.json")
    args = parser.parse_args()
    make = read_results(args.make)
    fa3 = read_results(args.fa3)
    tk = read_results(args.tk)
    splits = read_results(args.splits)
    if (len(make), len(fa3), len(tk), len(splits)) != (25, 60, 4, 20):
        raise ValueError("Expected 25 Make, 60 FA3, 4 independent TK, 20 split results")

    benchmarks = {}
    for record in make:
        task_id = record["task"]
        require_current(record, ROOT / "tasks" / task_id)
        if not (record["reference_passed"] and record["empty_rejected"] and record["positive_passed"]):
            raise ValueError(f"Missing positive/negative control: {task_id}")
        benchmarks[task_id] = dict(source_hash=record["source_hash"],
                                   per_size=record["reference_cases"])

    fa3_summary = []
    for record in fa3:
        task_id = "class2/hopper/" + record["task"]
        require_current(record, ROOT / "tasks" / task_id)
        if [t["seed"] for t in record["trials"]] != [42, 43, 44]:
            raise ValueError(f"Missing seeds: {task_id}")
        if not all(t["correct"] and all(t["negative_controls"].values()) for t in record["trials"]):
            raise ValueError(f"Failed control: {task_id}")
        benchmarks[task_id] = {k: record[k] for k in ["source_hash", "ref_min_ms", "ref_mean_ms"]}
        margins = {}
        controls = {}
        for trial in record["trials"]:
            controls.update(trial["negative_controls"])
            for output, values in trial["margin"].items():
                dest = margins.setdefault(output, {})
                for key, value in values.items():
                    if value is not None:
                        dest[key] = max(dest.get(key, 0), value)
        fa3_summary.append(dict(task=task_id, passed=True, **benchmarks[task_id],
                                seeds=[42, 43, 44], scalar_args=record["scalar_args"],
                                worst_margin=margins, negative_controls=controls,
                                peak_memory_bytes=record["peak_memory_bytes"]))

    for record in tk:
        require_current(record, ROOT / "tasks/class2/hopper" / record["task"])
        if not all(record["negative_controls"].values()):
            raise ValueError(f"Failed TK control: {record['task']}")

    expected = {str(path.parent.relative_to(ROOT / "tasks"))
                for cls in ["class1", "class2"]
                for path in (ROOT / "tasks" / cls / "hopper").glob("*/task.yaml")}
    if set(benchmarks) != expected:
        raise ValueError("Validation records do not cover the exact Hopper task set")
    frozen_splits = read_results("scripts/templates/fa3/h200_splits.json")
    if {r["task"]: r["best_num_splits"] for r in splits if r["passed"]} != frozen_splits:
        raise ValueError("Split calibration does not match frozen generator settings")

    gpu_names = {r["gpu"] for r in make + fa3}
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    if gpu_names != {gpu} or "H200" not in gpu:
        raise ValueError(f"Expected one validated H200 model, found {gpu_names} / {gpu}")
    import torch
    metadata = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), gpu=gpu,
        device_capability=list(torch.cuda.get_device_capability()),
        torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        timing_statistic="minimum / minimum", l2_eviction_bytes={"make":128 << 20,"fa3":256 << 20},
        gpu_execution="serialized with .cache/hopper_gpu.lock",
        coverage="active Makefile targets plus 60 generated FA3 tasks; Blackwell excluded",
    )
    # cuBLASLt algorithm IDs and heuristics depend on the installed library.
    # Also record vendored source fingerprints so a reproduced experiment can
    # distinguish a task change from a dependency/toolchain change.
    blas = ctypes.CDLL("libcublasLt.so")
    blas.cublasLtGetVersion.restype = ctypes.c_size_t
    metadata["reference_libraries"] = dict(
        cublaslt_version=int(blas.cublasLtGetVersion()),
        source_hashes={name: task_source_hash(ROOT / path) for name, path in {
            "cutlass": "reference_sources/cutlass/include",
            "fa3": "reference_sources/flash_attn_v3",
            "fa3_cutlass": "reference_sources/flash-attention/csrc/cutlass/include",
            "thunderkittens": "reference_sources/ThunderKittens/include",
            "deepgemm": "reference_sources/deep_gemm/deep_gemm",
        }.items()},
    )
    metadata["release_flags"] = dict(make=["-DNDEBUG"],
        fa3=["-DNDEBUG", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED"])
    cache = ROOT / "reference_benchmarks" / (gpu.replace(" ", "_") + ".json")
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(dict(metadata=metadata, benchmarks=benchmarks), indent=2) + "\n")
    evidence = dict(metadata=metadata, counts=dict(make=25, fa3=60, tk_independent_oracles=4),
                    make=make, fa3=fa3_summary, tk_independent_oracles=tk,
                    split_tuning=[{k: r[k] for k in ["task", "passed", "split_tuning_ms", "best_num_splits"]} for r in splits])
    report = ROOT / "docs/validation/hopper_h200.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Recorded {len(benchmarks)} current baselines: {cache}")
    print(f"Validation evidence: {report}")


if __name__ == "__main__":
    main()
