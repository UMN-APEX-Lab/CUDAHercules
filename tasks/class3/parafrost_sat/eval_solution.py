#!/usr/bin/env python3
"""
Evaluate a black-box parafrost_sat project candidate.

Expected input:
    python eval_solution.py <path/to/generated_project>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, "src")
EXECUTABLE = os.path.join(SRC_DIR, "parafrost")
PREPARE_DATA_SH = os.path.join(TASK_DIR, "prepare_data.sh")
PREP_SCRIPT = os.path.join(TASK_DIR, "prepare_blackbox_inputs.py")

REQUIRED_OPS = [
    "occurrence_table",
    "variable_scoring",
    "elimination",
    "subsumption",
    "blocked_clause_elimination",
    "equivalence_reasoning",
    "redundancy_elimination",
    "memory_recycling",
]

FORBIDDEN_PATTERNS = [
    r"tasks/class3/parafrost_sat/src",
    r"tasks/class3/parafrost_sat/dep",
    r"tasks/class3/parafrost_sat/data",
    r"\bparafrost\b",
    r"\.blackbox_inputs_v1\b",
    r"prepare_blackbox_inputs\.py",
]

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_dimacs_cnf(path: str) -> tuple[int, list[list[int]]]:
    """Parse a DIMACS CNF file. Returns (num_vars, clauses).

    Each clause is a list of non-zero signed ints (positive = pos literal,
    negative = neg literal). The terminating 0 is stripped.
    """
    num_vars = 0
    clauses: list[list[int]] = []
    current: list[int] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] == "c":
                continue
            if line[0] == "p":
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    try:
                        num_vars = int(parts[2])
                    except ValueError:
                        num_vars = 0
                continue
            for tok in line.split():
                try:
                    v = int(tok)
                except ValueError:
                    continue
                if v == 0:
                    if current:
                        clauses.append(current)
                        current = []
                else:
                    current.append(v)
    if current:
        clauses.append(current)
    return num_vars, clauses


def _verify_sat_assignment(
    cnf_path: str, assignment: list[int]
) -> tuple[bool, str]:
    """Check that `assignment` (DIMACS signed-int literals, positive = True)
    satisfies every clause in the given CNF file.

    Returns (ok, reason).
    """
    try:
        num_vars, clauses = _parse_dimacs_cnf(cnf_path)
    except OSError as e:
        return False, f"cannot read CNF {cnf_path}: {e}"

    # Build a variable → truth-value map.
    values: dict[int, bool] = {}
    for lit in assignment:
        if not isinstance(lit, (int, bool)):
            return False, f"assignment entry {lit!r} is not an integer"
        if isinstance(lit, bool):
            return False, f"assignment entry {lit!r} is boolean, expected signed int"
        if lit == 0:
            return False, "assignment must not contain 0"
        var = abs(lit)
        if var > num_vars:
            return False, f"assignment references var {var} but CNF declares only {num_vars} vars"
        truth = (lit > 0)
        if var in values and values[var] != truth:
            return False, f"assignment contains both {var} and -{var} (contradiction)"
        values[var] = truth

    unsatisfied: list[int] = []  # clause indices
    for idx, clause in enumerate(clauses):
        satisfied = False
        for lit in clause:
            var = abs(lit)
            truth = values.get(var)
            if truth is None:
                continue  # unassigned variable does not help satisfy this literal
            if (lit > 0 and truth) or (lit < 0 and not truth):
                satisfied = True
                break
        if not satisfied:
            unsatisfied.append(idx)
            if len(unsatisfied) >= 3:
                break
    if unsatisfied:
        return False, (f"assignment does not satisfy clauses "
                       f"{unsatisfied} (showing up to 3 of {len(clauses)} total clauses)")
    return True, ""


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


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _is_safe_relpath(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return bool(
        normalized
        and not normalized.startswith("/")
        and normalized not in (".", "..")
        and not normalized.startswith("../")
        and "/../" not in normalized
    )


def _prepare_data(env: dict) -> None:
    """Ensure the curated subset CNFs are staged under data/.

    The released repo keeps only scripts and an embedded subset manifest.
    `prepare_data.sh` downloads the Zenodo archive if any selected CNF is
    missing; it is idempotent when data is already present locally.
    """
    subprocess.check_call(["bash", PREPARE_DATA_SH], env=env)


def _prepare_standardized_inputs(env: dict) -> tuple[str, dict]:
    _prepare_data(env)
    input_dir = subprocess.check_output([sys.executable, PREP_SCRIPT], text=True, env=env).strip()
    manifest_path = os.path.join(input_dir, "instances.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"instances.json not found in {input_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return input_dir, manifest


def _parse_solver_output(output: str) -> dict:
    clean = _strip_ansi(output)
    sat = "SATISFIABLE" in clean and "UNSATISFIABLE" not in clean
    unsat = "UNSATISFIABLE" in clean
    verdict = "SAT" if sat else ("UNSAT" if unsat else "UNKNOWN")
    simp_sec = float(m.group(1)) if (m := re.search(r"Simplifier time\s*:\s*([0-9.]+)", clean)) else 0.0
    solve_sec = float(m.group(1)) if (m := re.search(r"Solver time\s*:\s*([0-9.]+)", clean)) else 0.0
    kernel_time_ms = (simp_sec + solve_sec) * 1000.0
    return {
        "verdict": verdict,
        "kernel_time_ms": kernel_time_ms,
        "output": clean,
    }


def _build_reference(env: dict) -> None:
    if os.path.isfile(EXECUTABLE):
        return
    subprocess.check_call(["make", f"-j{os.cpu_count()}"], cwd=SRC_DIR, env=env)


def _run_reference(timeout: int, env: dict, input_manifest: dict, input_dir: str) -> tuple[bool, dict, str]:
    _build_reference(env)
    if not isinstance(input_manifest, dict) or not isinstance(input_manifest.get("instances"), list):
        return False, {}, "invalid benchmark-owned instances.json"

    results: dict[str, dict] = {}
    logs: list[str] = []
    total_time_ms = 0.0

    for entry in input_manifest["instances"]:
        name = str(entry.get("name", "")).strip()
        rel = str(entry.get("file", "")).strip()
        if not name or not rel:
            return False, {}, "invalid instance entry in instances.json"
        cnf_path = os.path.join(input_dir, rel)
        proc = subprocess.run(
            [EXECUTABLE, cnf_path],
            cwd=SRC_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parsed = _parse_solver_output(proc.stdout + proc.stderr)
        results[name] = {
            "verdict": parsed["verdict"],
            "time_ms": parsed["kernel_time_ms"],
        }
        logs.append(f"== {name} ==\n{parsed['output'][-1000:]}")
        if parsed["verdict"] == "UNKNOWN":
            return False, {}, "\n\n".join(logs)[-4000:]
        total_time_ms += parsed["kernel_time_ms"]

    return True, {"kernel_time_ms": total_time_ms, "results": results}, "\n\n".join(logs)


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
    code = "\n\n".join(combined_cuda)
    if "__global__" not in code or "<<<" not in code:
        return False, "candidate project does not appear to contain custom CUDA kernels"
    return True, ""


def _stage_candidate_project(solution_dir: str, scratch_root: str) -> tuple[str | None, dict | None, str]:
    if not os.path.isdir(solution_dir):
        return None, None, f"solution directory not found: {solution_dir}"
    candidate_dir = os.path.join(scratch_root, "generated_project")
    shutil.copytree(solution_dir, candidate_dir)
    manifest, err = _load_manifest(candidate_dir)
    if err:
        return None, None, err
    ok, scan_err = _scan_candidate_sources(candidate_dir, manifest)
    if not ok:
        return None, None, scan_err
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
    if proc.returncode != 0:
        return False, f"Build failed:\n{(proc.stdout + proc.stderr)[-4000:]}"
    return True, ""


def _run_candidate(
    candidate_dir: str,
    manifest: dict,
    env: dict,
    timeout: int,
    input_dir: str,
    ref_time_ms: float,
) -> tuple[bool, str]:
    cmd = [
        "bash",
        manifest["run_script"],
        "--input-dir",
        input_dir,
        "--output-dir",
        os.path.join(candidate_dir, "benchmark_output"),
    ]
    if ref_time_ms > 0:
        cmd += ["--ref-time-ms", str(ref_time_ms)]
    proc = subprocess.run(
        cmd,
        cwd=candidate_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _validate_candidate_results(
    candidate_dir: str,
    ref_results: dict[str, dict],
    input_dir: str | None = None,
    input_manifest: dict | None = None,
) -> tuple[bool, float, dict, str]:
    output_dir = os.path.join(candidate_dir, "benchmark_output")
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

    by_name: dict[str, dict] = {}
    for entry in payload["results"]:
        if not isinstance(entry, dict):
            return False, -1.0, {}, "each benchmark_results entry must be an object"
        name = str(entry.get("name", "")).strip()
        verdict = str(entry.get("verdict", "")).strip().upper()
        time_ms = entry.get("time_ms", -1)
        assignment = entry.get("assignment", None)
        if not name:
            return False, -1.0, {}, "benchmark_results entry missing name"
        if verdict not in {"SAT", "UNSAT", "UNKNOWN"}:
            return False, -1.0, {}, f"invalid verdict for {name}: {verdict!r}"
        try:
            time_ms = float(time_ms)
        except Exception:
            return False, -1.0, {}, f"invalid time_ms for {name}"
        by_name[name] = {"verdict": verdict, "time_ms": time_ms, "assignment": assignment}

    missing = [name for name in ref_results if name not in by_name]
    if missing:
        return False, -1.0, {}, f"benchmark_results missing instances: {', '.join(missing)}"

    # Build a name→cnf_path lookup from the benchmark-owned manifest so we can
    # verify SAT assignments against the actual CNF.
    cnf_by_name: dict[str, str] = {}
    if input_dir and input_manifest:
        for inst in input_manifest.get("instances", []):
            iname = str(inst.get("name", "")).strip()
            rel = str(inst.get("file", "")).strip()
            if iname and rel:
                cnf_by_name[iname] = os.path.join(input_dir, rel)

    total_time_ms = 0.0
    details: dict[str, dict] = {}
    for name, ref in ref_results.items():
        cand = by_name[name]
        details[name] = {
            "reference_verdict": ref["verdict"],
            "candidate_verdict": cand["verdict"],
            "time_ms": cand["time_ms"],
        }
        if cand["verdict"] != ref["verdict"]:
            return False, -1.0, details, f"verdict mismatch for {name}: {cand['verdict']} vs {ref['verdict']}"
        if cand["time_ms"] < 0:
            return False, -1.0, details, f"invalid negative time_ms for {name}"
        # Anti-cheat: SAT verdicts MUST ship a model assignment we can verify.
        if cand["verdict"] == "SAT":
            assignment = cand.get("assignment")
            if not isinstance(assignment, list) or not assignment:
                return (False, -1.0, details,
                        f"SAT verdict for {name} missing required 'assignment' field "
                        f"(a list of signed DIMACS literals)")
            cnf_path = cnf_by_name.get(name)
            if not cnf_path or not os.path.isfile(cnf_path):
                return (False, -1.0, details,
                        f"cannot locate CNF file for {name} to verify assignment")
            ok, reason = _verify_sat_assignment(cnf_path, assignment)
            details[name]["assignment_verified"] = ok
            if not ok:
                return False, -1.0, details, f"SAT assignment failed for {name}: {reason}"
        total_time_ms += cand["time_ms"]

    return True, total_time_ms, details, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a black-box parafrost_sat project")
    parser.add_argument("solutions", nargs="+", help="Directory containing the generated_project files")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-phase timeout in seconds (matches paper §12.1; was 300 for synthetic-random-SAT data)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    if len(args.solutions) != 1:
        return _fail("Provide exactly one directory containing the generated project")
    solution_dir = os.path.abspath(args.solutions[0])
    if not os.path.isdir(solution_dir):
        return _fail(f"Solution directory not found: {solution_dir}")

    env = os.environ.copy()

    try:
        input_dir, input_manifest = _prepare_standardized_inputs(env)
    except Exception as e:
        return _fail(f"Failed to prepare benchmark-owned inputs: {e}")

    try:
        ref_ok, ref_metrics, ref_output = _run_reference(args.timeout, env, input_manifest, input_dir)
    except subprocess.TimeoutExpired:
        return _fail(f"Reference run timed out ({args.timeout}s)")
    if not ref_ok:
        return _fail(f"Reference run failed:\n{ref_output[-4000:]}")

    ref_time_ms = ref_metrics["kernel_time_ms"]
    ref_results = ref_metrics["results"]

    scratch_root = tempfile.mkdtemp(prefix="kh_parafrost_blackbox_")
    try:
        candidate_dir, manifest, err = _stage_candidate_project(solution_dir, scratch_root)
        if err:
            return _fail(err, compiled=False, ref_time_ms=ref_time_ms)

        try:
            compiled, build_err = _run_candidate_build(candidate_dir, manifest, env, args.timeout)
        except subprocess.TimeoutExpired:
            return _fail(f"Candidate build timed out ({args.timeout}s)", compiled=False, ref_time_ms=ref_time_ms)
        if not compiled:
            return _fail(build_err, compiled=False, ref_time_ms=ref_time_ms)

        try:
            ran_ok, sol_output = _run_candidate(candidate_dir, manifest, env, args.timeout, input_dir, ref_time_ms)
        except subprocess.TimeoutExpired:
            return _fail(f"Candidate run timed out ({args.timeout}s)", compiled=True, ref_time_ms=ref_time_ms)
        if not ran_ok:
            return _fail(sol_output[-4000:], compiled=True, ref_time_ms=ref_time_ms)

        correct, kernel_time_ms, details, err = _validate_candidate_results(
            candidate_dir, ref_results,
            input_dir=input_dir, input_manifest=input_manifest,
        )
        speedup = ref_time_ms / kernel_time_ms if correct and ref_time_ms > 0 and kernel_time_ms > 0 else -1.0

        print(
            json.dumps(
                {
                    "compiled": True,
                    "correct": correct,
                    "kernel_time_ms": kernel_time_ms,
                    "ref_time_ms": ref_time_ms,
                    "speedup": round(speedup, 4) if speedup > 0 else -1,
                    "reference_results": ref_results,
                    "candidate_results": details,
                    "output": sol_output[-4000:],
                    "error": "" if correct else err or sol_output[-4000:],
                },
                indent=2,
            )
        )
        return 0 if correct else 1
    finally:
        if args.keep_tmp:
            print(f"Temp directory kept at: {scratch_root}", file=sys.stderr)
        else:
            shutil.rmtree(scratch_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
