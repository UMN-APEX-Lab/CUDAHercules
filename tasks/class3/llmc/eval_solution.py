#!/usr/bin/env python3
"""
Evaluate a black-box llmc training project candidate.

Expected input:
    python eval_solution.py <path/to/generated_project>

The candidate supplies a complete project with:
  - build.sh
  - run.sh
  - project_manifest.json

The benchmark owns:
  - reference build / timing
  - dataset paths
  - fixed training configuration
  - correctness / speedup parsing
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "tiny_shakespeare_train.bin")
VAL_FILE = os.path.join(DATA_DIR, "tiny_shakespeare_val.bin")
RUN_PY = os.path.join(TASK_DIR, "run.py")
PREPARE_DATA_SH = os.path.join(TASK_DIR, "prepare_data.sh")

NUM_STEPS = 20
BATCH_SIZE = 16
SEQ_LEN = 256
LEARNING_RATE = 3e-4
VAL_EVERY = 10
SEED = 1337
DTYPE = "bf16"
OVERFIT_SINGLE_BATCH = 1

REQUIRED_OPS = [
    "attention",
    "layernorm",
    "gelu",
    "encoder",
    "fused_classifier",
    "adamw",
    "global_norm",
]

FORBIDDEN_PATTERNS = [
    r"#include\s*[<\"].*llmc/",
    r"\btrain_gpt2cu\b",
    r"\btrain_gpt2\.cu\b",
    r"tasks/class3/llmc/src",
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


def _prepare_data() -> None:
    if os.path.isfile(TRAIN_FILE) and os.path.isfile(VAL_FILE):
        return
    subprocess.check_call(["bash", PREPARE_DATA_SH, DATA_DIR])


def _parse_training_output(output: str) -> dict:
    def _find_float(pattern: str, default: float = -1.0) -> float:
        m = re.search(pattern, output, re.IGNORECASE)
        return float(m.group(1)) if m else default

    m = re.search(r"Solution loss:\s*([0-9.]+)\s*->\s*([0-9.]+)", output)
    first_loss = float(m.group(1)) if m else -1.0
    last_loss = float(m.group(2)) if m else -1.0

    # Full per-step trajectory (used for anti-cheat)
    per_step: list[float] = []
    m_traj = re.search(r"Loss per step:\s*([0-9.,eE+\-\s]+)", output)
    if m_traj:
        for tok in m_traj.group(1).strip().split(","):
            tok = tok.strip()
            try:
                per_step.append(float(tok))
            except ValueError:
                pass

    # Model parameter count (used for anti-cheat: prevents training a smaller model)
    model_params: int = -1
    m_params = re.search(r"Model params\s*:\s*([0-9]+)", output)
    if m_params:
        try:
            model_params = int(m_params.group(1))
        except ValueError:
            pass

    m = re.search(r"val loss:\s*([0-9.]+)\s*->\s*([0-9.]+)", output, re.IGNORECASE)
    first_val = float(m.group(1)) if m else -1.0
    last_val = float(m.group(2)) if m else -1.0

    checkpoints = {}
    m = re.search(
        r"Solution loss checkpoints:\s*.*?25%=([0-9.]+).*?50%=([0-9.]+).*?75%=([0-9.]+).*?100%=([0-9.]+)",
        output,
        re.IGNORECASE,
    )
    if m:
        checkpoints = {
            "25%": float(m.group(1)),
            "50%": float(m.group(2)),
            "75%": float(m.group(3)),
            "100%": float(m.group(4)),
        }

    kernel_time_ms = _find_float(r"Kernel time:\s*([0-9.]+)\s*ms")
    ref_time_ms = _find_float(r"Ref time:\s*([0-9.]+)\s*ms")
    speedup = _find_float(r"Speedup:\s*([0-9.]+)x")
    tok_per_sec = _find_float(r"tok/s:\s*([0-9.]+)")

    return {
        "kernel_time_ms": kernel_time_ms,
        "ref_time_ms": ref_time_ms,
        "speedup": speedup,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_val": first_val,
        "last_val": last_val,
        "loss_checkpoints": checkpoints,
        "per_step_losses": per_step,
        "model_params": model_params,
        "tok_per_sec": tok_per_sec,
    }


TRAJECTORY_DELTA_STD_MIN = 0.3          # reject smooth synthetic loss curves
TRAJECTORY_TOL_FRAC_MIN = 0.70          # fraction of steps that must be within tol of reference
TRAJECTORY_ABS_TOL = 3.0                # absolute tolerance floor (loss units)
TRAJECTORY_REL_TOL = 0.40               # relative tolerance (40% of ref loss at each step)
FIRST_LOSS_MIN = 10.0                   # ln(50257) = 10.82, so ~10.0 floor for random-init GPT-2
MODEL_PARAMS_MIN = 700_000_000          # must be >= 700M (GPT-2 774M target with 10% tolerance)
MODEL_PARAMS_MAX = 850_000_000          # must be <= 850M
CHECKPOINT_MIN_BYTES = 1_400_000_000    # >= ~1.4 GB (774M weights × 2 bytes bf16 = 1.55 GB)


def _trajectory_anti_cheat(sol_losses: list[float], ref_losses: list[float]) -> tuple[bool, str]:
    """Reject hand-crafted / synthetic loss trajectories.

    Check 1 — noise floor: the step-to-step deltas must have a reasonable
    standard deviation. A perfectly linear fake like `a - b*step` has zero std.

    Check 2 — reference alignment: at least TRAJECTORY_TOL_FRAC_MIN of the
    per-step losses must be within max(TRAJECTORY_ABS_TOL, TRAJECTORY_REL_TOL*ref)
    of the reference's loss at the same step. A completely fabricated trajectory
    won't track reality.
    """
    if len(sol_losses) < 4:
        return False, f"need at least 4 per-step losses for anti-cheat (got {len(sol_losses)})"
    deltas = [sol_losses[i + 1] - sol_losses[i] for i in range(len(sol_losses) - 1)]
    try:
        delta_std = statistics.pstdev(deltas)
    except statistics.StatisticsError:
        delta_std = 0.0
    if delta_std < TRAJECTORY_DELTA_STD_MIN:
        return False, (f"suspicious loss trajectory: step-to-step delta std={delta_std:.3f} "
                       f"(< {TRAJECTORY_DELTA_STD_MIN}); real training has noisy per-step losses.")
    if ref_losses:
        n = min(len(sol_losses), len(ref_losses))
        n_close = sum(
            1 for i in range(n)
            if abs(sol_losses[i] - ref_losses[i]) <= max(TRAJECTORY_ABS_TOL,
                                                         TRAJECTORY_REL_TOL * abs(ref_losses[i]))
        )
        frac = n_close / n if n else 0.0
        if frac < TRAJECTORY_TOL_FRAC_MIN:
            return False, (f"loss trajectory diverges from reference: only "
                           f"{n_close}/{n} = {frac:.1%} of steps within tolerance "
                           f"(need >= {TRAJECTORY_TOL_FRAC_MIN:.0%}). "
                           f"Your training is not producing the same loss as the reference.")
    return True, ""


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
    parsed = _parse_training_output(output)
    correct = proc.returncode == 0 and parsed["last_loss"] > 0 and parsed["last_loss"] < parsed["first_loss"] and parsed["last_loss"] < 8.0
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
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return False, f"failed to read {rel}: {e}"
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"forbidden reference dependency found in {rel}: {pattern}"

    combined_cuda = []
    for rel in manifest["cuda_sources"]:
        path = os.path.join(project_dir, rel)
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return False, f"failed to read CUDA source {rel}: {e}"
        combined_cuda.append(f"// FILE: {rel}\n{text}")

    code = "\n\n".join(combined_cuda)
    if "__global__" not in code or "<<<" not in code:
        return False, "candidate project does not appear to contain real CUDA kernels"
    return True, ""


def _stage_candidate_project(solution_dir: str, scratch_root: str) -> tuple[str, dict | None, str]:
    candidate_dir = os.path.join(scratch_root, "generated_project")
    shutil.copytree(solution_dir, candidate_dir)
    manifest, err = _load_manifest(candidate_dir)
    if err:
        return "", None, err
    ok, scan_err = _scan_candidate_sources(candidate_dir, manifest)
    if not ok:
        return "", None, scan_err
    return candidate_dir, manifest, ""


def _run_candidate_build(candidate_dir: str, manifest: dict, env: dict, timeout: int) -> tuple[bool, str]:
    build_script = os.path.join(candidate_dir, manifest["build_script"])
    if not os.path.isfile(build_script):
        return False, f"build script not found: {manifest['build_script']}"
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


def _run_candidate(candidate_dir: str, manifest: dict, env: dict, timeout: int,
                   ref_time_ms: float, ref_per_step: list[float] | None = None) -> tuple[bool, dict, str]:
    run_script = os.path.join(candidate_dir, manifest["run_script"])
    if not os.path.isfile(run_script):
        return False, {}, f"run script not found: {manifest['run_script']}"

    cmd = [
        "bash",
        manifest["run_script"],
        "--train-bin",
        TRAIN_FILE,
        "--val-bin",
        VAL_FILE,
        "--steps",
        str(NUM_STEPS),
        "--batch-size",
        str(BATCH_SIZE),
        "--seq-len",
        str(SEQ_LEN),
        "--learning-rate",
        str(LEARNING_RATE),
        "--val-every",
        str(VAL_EVERY),
        "--seed",
        str(SEED),
        "--dtype",
        DTYPE,
        "--overfit-single-batch",
        str(OVERFIT_SINGLE_BATCH),
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
    output = proc.stdout + proc.stderr
    parsed = _parse_training_output(output)
    basic_correct = (
        proc.returncode == 0
        and parsed["last_loss"] > 0
        and parsed["last_loss"] < parsed["first_loss"]
        and parsed["last_loss"] < 8.0
    )
    if not basic_correct:
        parsed["anti_cheat_reason"] = ""
        return False, parsed, output

    # Hidden gate: first_loss must be consistent with random-init GPT-2
    # over vocab=50257 (CE ≈ ln(50257) ≈ 10.82). Lower starts imply wrong
    # vocab / smaller model / pre-trained weights. NOT exposed in the prompt.
    if parsed["first_loss"] < FIRST_LOSS_MIN:
        parsed["anti_cheat_reason"] = (
            f"anti-cheat: suspicious first_loss={parsed['first_loss']:.3f} "
            f"(expected >= {FIRST_LOSS_MIN} for random-init GPT-2)."
        )
        return False, parsed, output

    # Hidden gate: final_weights.bin size + entropy must be consistent with a
    # real GPT-2 774M checkpoint (bf16 weights ≈ 1.55 GB). This catches
    # solutions that silently train a smaller model OR fake a large file by
    # filling with zeros. The contract requires the file; the SIZE threshold
    # and entropy check are NOT disclosed.
    ckpt_path = os.path.join(candidate_dir, "final_weights.bin")
    ckpt_size = os.path.getsize(ckpt_path) if os.path.isfile(ckpt_path) else 0
    parsed["ckpt_size_bytes"] = ckpt_size
    if ckpt_size < CHECKPOINT_MIN_BYTES:
        if ckpt_size == 0:
            parsed["anti_cheat_reason"] = (
                "anti-cheat: final_weights.bin is missing or empty. "
                "Your run.sh must save the final trained weights before exiting."
            )
        else:
            parsed["anti_cheat_reason"] = (
                f"anti-cheat: final_weights.bin is {ckpt_size:,} bytes — too small."
            )
        return False, parsed, output
    # Entropy spot-check: sample four 64 KB windows at 10/30/60/90% of the
    # file. Real weight arrays (random init + trained) are almost never mostly
    # zero, so reject files where any sampled window is >90% zero bytes.
    try:
        with open(ckpt_path, "rb") as f:
            bad_windows = 0
            for frac in (0.10, 0.30, 0.60, 0.90):
                f.seek(int(ckpt_size * frac))
                sample = f.read(65536)
                if not sample:
                    continue
                nonzero = sum(1 for b in sample if b != 0)
                if nonzero / len(sample) < 0.10:
                    bad_windows += 1
        if bad_windows >= 2:
            parsed["anti_cheat_reason"] = (
                f"anti-cheat: final_weights.bin has low-entropy regions "
                f"({bad_windows}/4 sampled windows are >90% zero bytes)."
            )
            return False, parsed, output
    except OSError as e:
        parsed["anti_cheat_reason"] = f"anti-cheat: cannot read final_weights.bin ({e})"
        return False, parsed, output

    # Hidden gate: if the solution happens to print a parameter count that
    # clearly signals a shrunken model, reject. Silent when the field is absent
    # to avoid leaking the check into the prompt.
    n_params = parsed.get("model_params", -1)
    if n_params > 0 and not (MODEL_PARAMS_MIN <= n_params <= MODEL_PARAMS_MAX):
        parsed["anti_cheat_reason"] = (
            f"anti-cheat: reported parameter count {n_params:,} outside expected range."
        )
        return False, parsed, output

    # Trajectory-based anti-cheat: catches fabricated loss curves.
    ok, reason = _trajectory_anti_cheat(parsed.get("per_step_losses", []),
                                        ref_per_step or [])
    parsed["anti_cheat_reason"] = (f"anti-cheat: {reason}" if not ok else "")
    return ok, parsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a black-box llmc project")
    parser.add_argument("solutions", nargs="+", help="Directory containing the generated_project files")
    parser.add_argument("--timeout", type=int, default=300, help="Per-phase timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    if len(args.solutions) != 1:
        return _fail("Provide exactly one directory containing the generated project")
    solution_dir = os.path.abspath(args.solutions[0])
    if not os.path.isdir(solution_dir):
        return _fail(f"Solution directory not found: {solution_dir}")

    _prepare_data()

    env = os.environ.copy()
    env["NO_MULTI_GPU"] = "1"

    try:
        ref_ok, ref_metrics, ref_output = _run_reference(args.timeout, env)
    except subprocess.TimeoutExpired:
        return _fail(f"Reference run timed out ({args.timeout}s)")
    if not ref_ok:
        return _fail(f"Reference run failed:\n{ref_output[-4000:]}")

    ref_time_ms = ref_metrics["kernel_time_ms"]

    scratch_root = tempfile.mkdtemp(prefix="kh_llmc_blackbox_")
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
            correct, sol_metrics, sol_output = _run_candidate(
                candidate_dir, manifest, env, args.timeout, ref_time_ms,
                ref_per_step=ref_metrics.get("per_step_losses", []),
            )
        except subprocess.TimeoutExpired:
            return _fail(f"Candidate run timed out ({args.timeout}s)", compiled=True, ref_time_ms=ref_time_ms)

        kernel_time_ms = sol_metrics["kernel_time_ms"]
        speedup = ref_time_ms / kernel_time_ms if ref_time_ms > 0 and kernel_time_ms > 0 else -1
        ref_last = ref_metrics["last_loss"]
        sol_last = sol_metrics["last_loss"]
        loss_ratio = sol_last / ref_last if ref_last > 0 and sol_last > 0 else -1

        print(
            json.dumps(
                {
                    "compiled": True,
                    "correct": correct,
                    "kernel_time_ms": kernel_time_ms,
                    "ref_time_ms": ref_time_ms,
                    "speedup": round(speedup, 4) if speedup > 0 else -1,
                    "loss_ratio": round(loss_ratio, 4) if loss_ratio > 0 else -1,
                    "ref_loss": {
                        "first": ref_metrics["first_loss"],
                        "last": ref_metrics["last_loss"],
                        "first_val": ref_metrics["first_val"],
                        "last_val": ref_metrics["last_val"],
                        "checkpoints": ref_metrics.get("loss_checkpoints", {}),
                    },
                    "solution_loss": {
                        "first": sol_metrics["first_loss"],
                        "last": sol_metrics["last_loss"],
                        "first_val": sol_metrics["first_val"],
                        "last_val": sol_metrics["last_val"],
                        "checkpoints": sol_metrics.get("loss_checkpoints", {}),
                    },
                    "anti_cheat_reason": sol_metrics.get("anti_cheat_reason", ""),
                    "sol_params": sol_metrics.get("model_params", -1),
                    "ref_params": ref_metrics.get("model_params", -1),
                    "ckpt_size_bytes": sol_metrics.get("ckpt_size_bytes", 0),
                    "tok_per_sec": sol_metrics["tok_per_sec"],
                    "output": sol_output[-4000:],
                    "error": (sol_metrics.get("anti_cheat_reason", "") if not correct else "")
                             or ("" if correct else sol_output[-4000:]),
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
