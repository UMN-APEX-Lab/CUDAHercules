#!/usr/bin/env python3
"""
Icicle ZK Benchmark -- CUDA-Hercules Class 3

Runs NTT (finite-field FFT) and MSM (multi-scalar multiplication) on BN254
at multiple sizes. Compares the custom CUDA backend against:
  1. CPU baseline (correctness reference)
  2. Icicle's optimized CUDA backend (performance target)

Reference: Icicle (Ingonyama) — GPU-accelerated ZK proof library.
"""

import os
import re
import subprocess
import sys


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, "src")
BUILD_DIR = os.path.join(SRC_DIR, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "bench")

ICICLE_SRC = os.path.join(TASK_DIR, "..", "..", "..", "reference_sources", "icicle")
ICICLE_LIB = os.path.join(ICICLE_SRC, "build")
BASELINE_CUDA_BACKEND = os.path.join(TASK_DIR, "cuda_backend")
CUSTOM_CUDA_BACKEND = os.path.join(TASK_DIR, "custom_cuda_backend")

REF_DIR = os.path.join(TASK_DIR, "ref_data")
CUDA_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "")


def backend_available(path):
    """Return True if a backend directory contains at least one shared library."""
    if not os.path.isdir(path):
        return False
    for root, _, files in os.walk(path):
        if any(name.endswith(".so") for name in files):
            return True
    return False


def find_custom_backend_so(path):
    """Return the preferred custom backend shared library path, if any."""
    if not os.path.isdir(path):
        return ""

    preferred = []
    fallback = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith(".so"):
                continue
            full = os.path.join(root, name)
            if name == "libkh_custom_backend.so":
                preferred.append(full)
            else:
                fallback.append(full)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def build_icicle():
    """Build Icicle core library with CPU backend if not already built."""
    if os.path.isfile(os.path.join(ICICLE_LIB, "libicicle_curve_bn254.so")):
        return
    print("Building Icicle...", flush=True)
    os.makedirs(ICICLE_LIB, exist_ok=True)
    subprocess.check_call(
        [
            "cmake",
            "..",
            "-DCPU_BACKEND=ON",
            "-DCURVE=bn254",
            "-DMSM=ON",
            "-DNTT=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTS=OFF",
        ],
        cwd=ICICLE_LIB,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["make", f"-j{os.cpu_count()}"],
        cwd=ICICLE_LIB,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print("Icicle built.", flush=True)


def build_bench():
    """Configure and build the benchmark harness."""
    print("Building benchmark...", flush=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.check_call(
        ["cmake", ".."],
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["make", f"-j{os.cpu_count()}"],
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print("Benchmark built.", flush=True)


def run_bench(device, backend_dir=None, custom_so=None, save_ref=False):
    """Run the benchmark with the requested execution path."""
    cmd = [EXECUTABLE, "--device", device, "--ref-dir", REF_DIR]
    if backend_dir:
        cmd += ["--backend", backend_dir]
    if custom_so:
        cmd += ["--custom-so", custom_so]
    if save_ref:
        cmd += ["--save-ref"]

    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        ICICLE_LIB if not existing_ld else f"{ICICLE_LIB}:{existing_ld}"
    )
    if backend_dir:
        env["ICICLE_BACKEND_INSTALL_DIR"] = backend_dir
    if CUDA_DEVICE:
        env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + result.stderr

    ops = re.findall(r"((?:NTT|MSM) 2\^\d+):\s+([0-9.]+)\s+ms\s+\[(PASS|FAIL)\]", output)
    ntt_total = float(m.group(1)) if (m := re.search(r"NTT total:\s+([0-9.]+)", output)) else -1
    msm_total = float(m.group(1)) if (m := re.search(r"MSM total:\s+([0-9.]+)", output)) else -1
    kernel_time = float(m.group(1)) if (m := re.search(r"Kernel time:\s+([0-9.]+)", output)) else -1
    passed = "Passed" in output and result.returncode == 0

    return {
        "ops": ops,
        "ntt_total": ntt_total,
        "msm_total": msm_total,
        "kernel_time": kernel_time,
        "passed": passed,
        "returncode": result.returncode,
        "output": output,
    }


def main():
    build_icicle()
    build_bench()
    os.makedirs(REF_DIR, exist_ok=True)

    print("\n=== Icicle ZK Benchmark: NTT + MSM on BN254 ===\n", flush=True)

    print("--- CPU Reference ---", flush=True)
    cpu = run_bench("CPU", save_ref=True)
    print(
        f"  NTT: {cpu['ntt_total']:.2f} ms  MSM: {cpu['msm_total']:.2f} ms  "
        f"E2E: {cpu['kernel_time']:.2f} ms"
    )

    baseline = None
    if backend_available(BASELINE_CUDA_BACKEND):
        print("\n--- CUDA Baseline (Icicle) ---", flush=True)
        baseline = run_bench("CUDA", backend_dir=BASELINE_CUDA_BACKEND)
        if baseline["passed"]:
            print(
                f"  NTT: {baseline['ntt_total']:.2f} ms  MSM: {baseline['msm_total']:.2f} ms  "
                f"E2E: {baseline['kernel_time']:.2f} ms"
            )
            print(f"  Speedup vs CPU: {cpu['kernel_time'] / baseline['kernel_time']:.1f}x")
        else:
            print("  CUDA baseline FAILED correctness check")
            baseline = None
    else:
        print("\n--- CUDA baseline not available (no cuda_backend/) ---")

    solution = None
    custom_so = find_custom_backend_so(CUSTOM_CUDA_BACKEND)
    if custom_so:
        print("\n--- Custom CUDA Backend (Solution) ---", flush=True)
        solution = run_bench("CUSTOM", custom_so=custom_so)
        if solution["passed"]:
            print(
                f"  NTT: {solution['ntt_total']:.2f} ms  MSM: {solution['msm_total']:.2f} ms  "
                f"E2E: {solution['kernel_time']:.2f} ms"
            )
            print(f"  Speedup vs CPU: {cpu['kernel_time'] / solution['kernel_time']:.1f}x")
            if baseline and baseline["passed"]:
                print(f"  Speedup vs baseline: {baseline['kernel_time'] / solution['kernel_time']:.1f}x")
        else:
            print("  Custom CUDA backend FAILED correctness check")
            solution = None
    else:
        print("\n--- Custom CUDA backend not available (no custom backend .so found) ---")

    primary = solution if solution and solution["passed"] else baseline if baseline and baseline["passed"] else cpu

    print("\n=== Summary ===")
    print("Passed" if primary["passed"] else "FAILED")
    print(f"Kernel time: {primary['kernel_time']:.4f} ms")
    print(f"  NTT total: {primary['ntt_total']:.2f} ms")
    print(f"  MSM total: {primary['msm_total']:.2f} ms")

    if solution and solution["passed"]:
        print("\n  Per-op breakdown (Solution CUDA):")
        for name, time_ms, _status in solution["ops"]:
            cpu_time = next((float(t) for n, t, _ in cpu["ops"] if n == name), 0.0)
            sol_time = float(time_ms)
            speedup = cpu_time / sol_time if sol_time > 0 else 0.0
            print(f"    {name}: {sol_time:.2f} ms (CPU: {cpu_time:.2f} ms, {speedup:.1f}x)")
    elif baseline and baseline["passed"]:
        print("\n  Per-op breakdown (CUDA baseline):")
        for name, time_ms, _status in baseline["ops"]:
            cpu_time = next((float(t) for n, t, _ in cpu["ops"] if n == name), 0.0)
            base_time = float(time_ms)
            speedup = cpu_time / base_time if base_time > 0 else 0.0
            print(f"    {name}: {base_time:.2f} ms (CPU: {cpu_time:.2f} ms, {speedup:.1f}x)")

    if not primary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
