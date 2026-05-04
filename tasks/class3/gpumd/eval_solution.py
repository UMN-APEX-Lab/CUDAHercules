#!/usr/bin/env python3
"""
Evaluate a black-box gpumd project candidate.

Expected input:
    python eval_solution.py <path/to/generated_project>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from array import array


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "ref_forces")
RUN_PY = os.path.join(TASK_DIR, "run.py")
PREP_SCRIPT = os.path.join(TASK_DIR, "prepare_blackbox_inputs.py")

REQUIRED_OPS = [
    "neighbor_list",
    "integration",
    "kinetic_energy",
    "lennard_jones",
    "tersoff",
    "coulomb_real",
    "coulomb_kspace",
]

FORBIDDEN_PATTERNS = [
    r"tasks/class3/gpumd/src",
    r"tasks/class3/gpumd/data",
    r"\bmd_bench\b",
    r"\bref_forces\b",
    r"\.blackbox_inputs_v1\b",
    r"prepare_blackbox_inputs\.py",
    r"model\.xyz",
]


def _fail(message: str, compiled: bool = False, ref_time_ms: float = -1.0) -> int:
    print(
        json.dumps(
            {
                "compiled": compiled,
                "correct": False,
                "kernel_time_ms": -1,
                "ref_time_ms": ref_time_ms,
                "speedup": -1,
                "output": message,
                "error": message,
            }
        )
    )
    return 1


def _is_safe_relpath(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return bool(
        normalized
        and not normalized.startswith("/")
        and normalized not in (".", "..")
        and not normalized.startswith("../")
        and "/../" not in normalized
    )


def _ensure_reference_force_files(env: dict) -> None:
    expected = [
        "lj_Ar_256000.bin",
        "lj_Ar_500000.bin",
        "tersoff_Si_27000.bin",
        "tersoff_Si_64000.bin",
        "coulomb_NaCl_8000.bin",
        "coulomb_NaCl_32768.bin",
    ]
    if all(os.path.isfile(os.path.join(REF_DIR, name)) for name in expected):
        return

    src_dir = os.path.join(TASK_DIR, "src")
    executable = os.path.join(src_dir, "md_bench")
    if not os.path.isfile(executable):
        subprocess.check_call(["make", "md_bench"], cwd=src_dir, env=env)
    subprocess.check_call([executable, DATA_DIR, "--save-ref"], cwd=src_dir, env=env)


def _prepare_standardized_inputs(env: dict) -> tuple[str, dict]:
    _ensure_reference_force_files(env)
    input_dir = (
        subprocess.check_output([sys.executable, PREP_SCRIPT], text=True, env=env).strip()
    )
    manifest_path = os.path.join(input_dir, "workloads.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"workloads.json not found in {input_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return input_dir, manifest


def _parse_reference_output(output: str) -> dict:
    m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
    kernel_time_ms = float(m.group(1)) if m else -1.0
    passed = "Passed" in output
    return {"kernel_time_ms": kernel_time_ms, "passed": passed}


def _run_reference(timeout: int, env: dict) -> tuple[bool, dict, str]:
    proc = subprocess.run(
        [sys.executable, RUN_PY],
        cwd=TASK_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    parsed = _parse_reference_output(output)
    correct = proc.returncode == 0 and parsed["passed"] and parsed["kernel_time_ms"] > 0
    return correct, parsed, output


def _load_manifest(project_dir: str) -> tuple[dict | None, str]:
    manifest_path = os.path.join(project_dir, "project_manifest.json")
    if not os.path.isfile(manifest_path):
        return None, "project_manifest.json not found"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return None, f"failed to parse project_manifest.json: {e}"

    if not isinstance(manifest, dict):
        return None, "project_manifest.json must be a JSON object"

    build_script = str(manifest.get("build_script", "")).strip()
    run_script = str(manifest.get("run_script", "")).strip()
    files = manifest.get("files", [])
    cuda_sources = manifest.get("cuda_sources", [])
    ops = manifest.get("ops", {})

    if not build_script or not run_script:
        return None, "manifest must define build_script and run_script"
    if not isinstance(files, list) or not files:
        return None, "manifest must define a non-empty files list"
    if not isinstance(cuda_sources, list) or not cuda_sources:
        return None, "manifest must define a non-empty cuda_sources list"
    if not isinstance(ops, dict):
        return None, "manifest ops must be a JSON object"

    seen = set()
    for rel in files:
        rel = str(rel).strip()
        if not _is_safe_relpath(rel):
            return None, f"unsafe file path in manifest: {rel!r}"
        seen.add(rel)

    if build_script not in seen or run_script not in seen:
        return None, "build_script and run_script must both appear in files"

    for rel in cuda_sources:
        rel = str(rel).strip()
        if not _is_safe_relpath(rel):
            return None, f"unsafe cuda_sources path: {rel!r}"
        if rel not in seen:
            return None, f"cuda_sources entry missing from files: {rel}"

    missing_ops = [op for op in REQUIRED_OPS if op not in ops]
    if missing_ops:
        return None, f"manifest ops missing required keys: {', '.join(missing_ops)}"

    for op, rel in ops.items():
        rel = str(rel).strip()
        if not _is_safe_relpath(rel):
            return None, f"unsafe ops path for {op}: {rel!r}"
        if rel not in seen:
            return None, f"ops entry for {op} missing from files: {rel}"

    return manifest, ""


def _scan_candidate_sources(project_dir: str, manifest: dict) -> tuple[bool, str]:
    for rel in manifest["files"]:
        path = os.path.join(project_dir, rel)
        if not os.path.isfile(path):
            return False, f"manifest-listed file not found: {rel}"
        if os.path.getsize(path) == 0:
            return False, f"empty manifest-listed file: {rel}"
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"forbidden reference dependency found in {rel}: {pattern}"

    combined_cuda = []
    for rel in manifest["cuda_sources"]:
        path = os.path.join(project_dir, rel)
        with open(path, "r", errors="ignore") as f:
            combined_cuda.append(f"// FILE: {rel}\n{f.read()}")
    combined_text = "\n\n".join(combined_cuda)
    if "__global__" not in combined_text or "<<<" not in combined_text:
        return False, "candidate project does not appear to contain custom CUDA kernels"
    return True, ""


def _stage_candidate_project(solution_dir: str, scratch_root: str) -> tuple[str | None, dict | None, str]:
    if not os.path.isdir(solution_dir):
        return None, None, f"solution directory not found: {solution_dir}"

    candidate_dir = os.path.join(scratch_root, "generated_project")
    shutil.copytree(solution_dir, candidate_dir)

    manifest, error = _load_manifest(candidate_dir)
    if manifest is None:
        return None, None, error

    ok, scan_error = _scan_candidate_sources(candidate_dir, manifest)
    if not ok:
        return None, None, scan_error

    return candidate_dir, manifest, ""


def _run_candidate_build(candidate_dir: str, manifest: dict, env: dict, timeout: int) -> tuple[bool, str]:
    proc = subprocess.run(
        ["bash", manifest["build_script"]],
        cwd=candidate_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output


def _load_force_file(path: str) -> tuple[int, array, array, array, array]:
    with open(path, "rb") as f:
        raw_n = f.read(4)
        if len(raw_n) != 4:
            raise ValueError(f"invalid force file header: {path}")
        n = struct.unpack("<i", raw_n)[0]
        arrays = []
        for _ in range(4):
            buf = array("d")
            buf.fromfile(f, n)
            arrays.append(buf)
    return n, arrays[0], arrays[1], arrays[2], arrays[3]


def _compare_forces(candidate_path: str, reference_path: str, force_tol: float) -> tuple[bool, dict]:
    cn, cfx, cfy, cfz, cpe = _load_force_file(candidate_path)
    rn, rfx, rfy, rfz, rpe = _load_force_file(reference_path)
    if cn != rn:
        return False, {"error": f"force file atom_count mismatch: {cn} vs {rn}"}

    sum_fmag = 0.0
    for i in range(rn):
        sum_fmag += math.sqrt(rfx[i] * rfx[i] + rfy[i] * rfy[i] + rfz[i] * rfz[i])
    avg_fmag = sum_fmag / rn if rn else 0.0
    fmag_floor = avg_fmag * 1e-6

    max_abs_err = 0.0
    max_rel_err = 0.0
    max_pe_err = 0.0
    sum_sq = 0.0

    for i in range(rn):
        efx = abs(cfx[i] - rfx[i])
        efy = abs(cfy[i] - rfy[i])
        efz = abs(cfz[i] - rfz[i])
        epe = abs(cpe[i] - rpe[i])
        e = max(efx, efy, efz)
        fmag = math.sqrt(rfx[i] * rfx[i] + rfy[i] * rfy[i] + rfz[i] * rfz[i])

        max_abs_err = max(max_abs_err, e)
        max_pe_err = max(max_pe_err, epe)
        sum_sq += efx * efx + efy * efy + efz * efz

        if fmag > fmag_floor:
            max_rel_err = max(max_rel_err, e / fmag)

    rms_err = math.sqrt(sum_sq / (3.0 * rn)) if rn else 0.0
    correct = max_rel_err < force_tol
    return correct, {
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "rms_err": rms_err,
        "max_pe_err": max_pe_err,
    }


def _validate_candidate_results(output_dir: str, manifest: dict) -> tuple[bool, float, dict, str]:
    results_path = os.path.join(output_dir, "benchmark_results.json")
    if not os.path.isfile(results_path):
        return False, -1.0, {}, "benchmark_results.json not found"

    try:
        with open(results_path) as f:
            payload = json.load(f)
    except Exception as e:
        return False, -1.0, {}, f"failed to parse benchmark_results.json: {e}"

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False, -1.0, {}, "benchmark_results.json must contain a results list"

    by_name = {}
    for entry in payload["results"]:
        if not isinstance(entry, dict):
            return False, -1.0, {}, "invalid result entry in benchmark_results.json"
        name = str(entry.get("name", "")).strip()
        force_file = str(entry.get("force_file", "")).strip()
        if not name or not force_file:
            return False, -1.0, {}, "each result entry must define name and force_file"
        by_name[name] = entry

    total_time = 0.0
    details = {}
    all_correct = True
    for workload in manifest.get("workloads", []):
        name = workload["name"]
        if name not in by_name:
            return False, -1.0, {}, f"missing workload result: {name}"
        entry = by_name[name]
        force_file = str(entry.get("force_file", "")).strip()
        if not _is_safe_relpath(force_file):
            return False, -1.0, {}, f"unsafe force_file path for {name}: {force_file!r}"
        candidate_force_path = os.path.join(output_dir, force_file)
        reference_path = os.path.join(REF_DIR, workload["ref_file"])
        if not os.path.isfile(candidate_force_path):
            return False, -1.0, {}, f"force output missing for {name}: {force_file}"
        if not os.path.isfile(reference_path):
            return False, -1.0, {}, f"reference force file missing for {name}: {reference_path}"

        force_correct, force_detail = _compare_forces(
            candidate_force_path,
            reference_path,
            float(workload["force_rel_tol"]),
        )
        energy_drift = float(entry.get("energy_drift", 1e30))
        time_ms = float(entry.get("time_ms", -1.0))
        total_time += max(time_ms, 0.0)

        energy_correct = energy_drift < float(workload["energy_drift_tol"])
        workload_correct = force_correct and energy_correct and time_ms > 0
        if not workload_correct:
            all_correct = False

        details[name] = {
            "force_correct": force_correct,
            "energy_correct": energy_correct,
            "energy_drift": energy_drift,
            "time_ms": time_ms,
            **force_detail,
        }

    return all_correct, total_time, details, ""


def _run_candidate(
    candidate_dir: str,
    candidate_manifest: dict,
    workload_manifest: dict,
    env: dict,
    timeout: int,
    input_dir: str,
    ref_time_ms: float,
) -> tuple[bool, float, dict, str]:
    output_dir = os.path.join(candidate_dir, "candidate_outputs")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "bash",
        candidate_manifest["run_script"],
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
    ]
    if ref_time_ms > 0:
        cmd.extend(["--ref-time-ms", f"{ref_time_ms:.4f}"])

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=candidate_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed_ms = (time.time() - start) * 1000.0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, -1.0, {}, output

    correct, reported_total_ms, details, error = _validate_candidate_results(
        output_dir, workload_manifest
    )
    total_ms = reported_total_ms if reported_total_ms > 0 else elapsed_ms
    return correct, total_ms, details, output if not error else f"{output}\n{error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a black-box gpumd project")
    parser.add_argument("solution", help="Path to generated_project")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    args = parser.parse_args()

    solution_dir = os.path.abspath(args.solution)
    env = os.environ.copy()

    try:
        input_dir, manifest = _prepare_standardized_inputs(env)
    except Exception as e:
        return _fail(f"Failed to prepare standardized inputs: {e}")

    try:
        ref_ok, ref_parsed, ref_output = _run_reference(args.timeout, env)
    except subprocess.TimeoutExpired:
        return _fail(f"Reference run timed out ({args.timeout}s)")
    if not ref_ok:
        return _fail(f"Reference run failed:\n{ref_output}")

    scratch_root = tempfile.mkdtemp(prefix="kh_gpumd_blackbox_")
    try:
        candidate_dir, candidate_manifest, stage_error = _stage_candidate_project(solution_dir, scratch_root)
        if candidate_dir is None or candidate_manifest is None:
            return _fail(stage_error, compiled=False, ref_time_ms=ref_parsed["kernel_time_ms"])

        try:
            compiled, build_output = _run_candidate_build(
                candidate_dir, candidate_manifest, env, args.timeout
            )
        except subprocess.TimeoutExpired:
            return _fail(
                f"Build timed out ({args.timeout}s)",
                compiled=False,
                ref_time_ms=ref_parsed["kernel_time_ms"],
            )
        if not compiled:
            return _fail(
                f"Build failed:\n{build_output}",
                compiled=False,
                ref_time_ms=ref_parsed["kernel_time_ms"],
            )

        try:
            correct, kernel_time_ms, details, output = _run_candidate(
                candidate_dir,
                candidate_manifest,
                manifest,
                env,
                args.timeout,
                input_dir,
                ref_parsed["kernel_time_ms"],
            )
        except subprocess.TimeoutExpired:
            return _fail(
                f"Candidate run timed out ({args.timeout}s)",
                compiled=True,
                ref_time_ms=ref_parsed["kernel_time_ms"],
            )

        speedup = (
            ref_parsed["kernel_time_ms"] / kernel_time_ms
            if correct and kernel_time_ms > 0 and ref_parsed["kernel_time_ms"] > 0
            else -1
        )

        result = {
            "compiled": True,
            "correct": correct,
            "kernel_time_ms": kernel_time_ms,
            "ref_time_ms": ref_parsed["kernel_time_ms"],
            "speedup": speedup,
            "workloads": details,
            "output": output[-4000:],
            "error": "" if correct else output[-4000:],
        }
        print(json.dumps(result))
        return 0 if correct else 1
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
