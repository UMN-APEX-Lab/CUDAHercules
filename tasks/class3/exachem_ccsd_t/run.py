#!/usr/bin/env python3
"""
Run harness for ExaChem CCSD(T) benchmark task.

Builds reference and solution kernels, runs both, compares energies,
and reports performance.
"""
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, "src")

# Tolerance for energy comparison (FP64, relative)
ENERGY_REL_TOL = 1e-6
PLACEHOLDER_MARKERS = {
    "solution.cu": "CUDA_HERCULES_PLACEHOLDER_SOLUTION",
    "ccsd_t_g2s_device_functions.cu": "CUDA_HERCULES_PLACEHOLDER_G2S",
    "tensor_core_helper.cuh": "CUDA_HERCULES_PLACEHOLDER_TENSOR_CORE_HELPER",
}


def run_cmd(cmd, cwd=None, timeout=300):
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"


def build(target):
    """Build a target (ref or test). Returns True on success."""
    print(f"Building {target}...")
    rc, stdout, stderr = run_cmd(f"make {target}", cwd=SRC_DIR, timeout=120)
    if rc != 0:
        print(f"Build failed for {target}:")
        print(stderr)
        print(stdout)
        return False
    return True


def run_benchmark(binary_name):
    """Run a benchmark binary. Returns (success, stdout)."""
    binary = os.path.join(TASK_DIR, binary_name)
    if not os.path.exists(binary):
        print(f"Binary not found: {binary}")
        return False, ""

    print(f"Running {binary_name}...")
    rc, stdout, stderr = run_cmd(binary, cwd=TASK_DIR, timeout=300)
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return rc == 0, stdout


def parse_energy(stdout, label):
    """Parse energy value from stdout."""
    pattern = rf"{label}:\s+([-+]?\d+\.?\d*[eE][-+]?\d+)"
    match = re.search(pattern, stdout)
    if match:
        return float(match.group(1))
    return None


def parse_kernel_time(stdout):
    """Parse kernel time from stdout."""
    match = re.search(r"Kernel time:\s+([\d.]+)\s*ms", stdout)
    if match:
        return float(match.group(1))
    return None


def compare_energies(ref_e1, ref_e2, sol_e1, sol_e2):
    """Compare reference and solution energies."""
    for name, ref, sol in [("Energy_T", ref_e1, sol_e1), ("Energy_T5", ref_e2, sol_e2)]:
        if ref is None or sol is None:
            print(f"FAIL: Could not parse {name}")
            return False

        if ref == 0.0:
            if sol != 0.0:
                print(f"FAIL: {name}: ref=0 but sol={sol:.15e}")
                return False
            continue

        rel_err = abs(sol - ref) / abs(ref)
        print(f"{name}: ref={ref:.15e}, sol={sol:.15e}, rel_err={rel_err:.2e}")
        if rel_err > ENERGY_REL_TOL:
            print(f"FAIL: {name} relative error {rel_err:.2e} > tolerance {ENERGY_REL_TOL:.2e}")
            return False

    return True


def main():
    # Clean previous builds
    run_cmd("make clean", cwd=SRC_DIR)

    # Build reference
    if not build("ref"):
        print("FAILED: Reference build failed")
        return 1

    missing_files = []
    for relpath, marker in PLACEHOLDER_MARKERS.items():
        path = os.path.join(TASK_DIR, relpath)
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            text = ""
        if marker in text:
            missing_files.append(relpath)

    if missing_files:
        print("FAILED: Candidate implementation is incomplete")
        print("Replace these placeholder files before running run.py:")
        for relpath in missing_files:
            print(f"  - tasks/class3/exachem_ccsd_t/{relpath}")
        return 1

    # Build solution
    if not build("test"):
        print("FAILED: Solution build failed")
        return 1

    # Run reference
    ref_ok, ref_stdout = run_benchmark("ref_benchmark")
    if not ref_ok:
        print("FAILED: Reference run failed")
        return 1

    # Run solution
    sol_ok, sol_stdout = run_benchmark("sol_benchmark")
    if not sol_ok:
        print("FAILED: Solution run failed")
        return 1

    # Parse reference energies
    ref_e1 = parse_energy(ref_stdout, "Energy_T")
    ref_e2 = parse_energy(ref_stdout, "Energy_T5")
    ref_time = parse_kernel_time(ref_stdout)

    # Parse solution energies
    sol_e1 = parse_energy(sol_stdout, "Energy_T")
    sol_e2 = parse_energy(sol_stdout, "Energy_T5")
    sol_time = parse_kernel_time(sol_stdout)

    # Compare energies
    if not compare_energies(ref_e1, ref_e2, sol_e1, sol_e2):
        print("FAILED: Energy mismatch")
        return 1

    # Report performance
    if ref_time is not None and sol_time is not None:
        speedup = ref_time / sol_time if sol_time > 0 else 0
        print(f"\nRef time: {ref_time:.3f} ms")
        print(f"Kernel time: {sol_time:.3f} ms")
        print(f"Speedup: {speedup:.2f} x")
    elif sol_time is not None:
        print(f"\nKernel time: {sol_time:.3f} ms")

    print("Passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
