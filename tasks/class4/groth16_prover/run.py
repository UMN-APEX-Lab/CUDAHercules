#!/usr/bin/env python3
"""
Groth16 ZK Prover Benchmark -- CUDA-Hercules Class 4

Runs the full Groth16 proving pipeline (NTT + MSM) on BN254 at multiple
circuit sizes. Compares against Icicle CPU baseline and CUDA backend.

Pipeline per circuit:
  Phase 1: 3 IFFT + 3 coset FFT + element-wise ops + 1 coset IFFT → H(x)
  Phase 2: 5 MSMs (3×G1 + 1×G2 + 1×G1) → proof (π_A, π_B, π_C)
"""

import os
import sys
import re
import subprocess

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, 'src')
BUILD_DIR = os.path.join(SRC_DIR, 'build')
EXECUTABLE = os.path.join(BUILD_DIR, 'groth16_bench')

ICICLE_SRC = os.path.join(TASK_DIR, '..', '..', '..', 'reference_sources', 'icicle')
ICICLE_LIB = os.path.join(ICICLE_SRC, 'build')
CUDA_BACKEND = os.path.join(TASK_DIR, '..', '..', 'class3', 'icicle_zk', 'cuda_backend')

CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '')

# ── Build ──────────────────────────────────────────────────────────────

def build_icicle():
    if os.path.isfile(os.path.join(ICICLE_LIB, 'libicicle_curve_bn254.so')):
        return
    print("Building Icicle...", flush=True)
    os.makedirs(ICICLE_LIB, exist_ok=True)
    subprocess.check_call(
        ['cmake', '..', '-DCPU_BACKEND=ON', '-DCURVE=bn254', '-DMSM=ON', '-DNTT=ON',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTS=OFF'],
        cwd=ICICLE_LIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(
        ['make', f'-j{os.cpu_count()}'],
        cwd=ICICLE_LIB, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def build_bench():
    if os.path.isfile(EXECUTABLE):
        return
    print("Building benchmark...", flush=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.check_call(['cmake', '..'], cwd=BUILD_DIR,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(['make', f'-j{os.cpu_count()}'], cwd=BUILD_DIR,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# ── Run ────────────────────────────────────────────────────────────────

def run_bench(device, backend_dir=None):
    cmd = [EXECUTABLE, '--device', device]
    if backend_dir:
        cmd += ['--backend', backend_dir]

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ICICLE_LIB
    if backend_dir:
        env['ICICLE_BACKEND_INSTALL_DIR'] = backend_dir
    if CUDA_DEVICE:
        env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + result.stderr

    kernel_time = float(m.group(1)) if (m := re.search(r'Kernel time:\s+([0-9.]+)', output)) else -1
    passed = 'Passed' in output and result.returncode == 0

    return {
        'kernel_time': kernel_time,
        'passed': passed,
        'output': output,
    }

# ── Main ──────────────────────────────────────────────────────────────

def main():
    build_icicle()
    build_bench()

    print("\n=== Groth16 Prover Benchmark ===\n", flush=True)

    # CPU baseline
    print("--- CPU Baseline ---", flush=True)
    cpu = run_bench('CPU')
    print(cpu['output'])

    # CUDA (if available)
    cuda_backend = os.path.realpath(CUDA_BACKEND)
    cuda = None
    if os.path.isdir(cuda_backend):
        print("\n--- CUDA Baseline (Icicle) ---", flush=True)
        cuda = run_bench('CUDA', backend_dir=cuda_backend)
        # Filter debug lines
        for line in cuda['output'].split('\n'):
            if not line.startswith('[DEBUG]'):
                print(line)

    # Summary
    primary = cuda if cuda and cuda['passed'] else cpu
    print(f"\n=== Summary ===")
    if primary['passed']:
        print("Passed")
    else:
        print("FAILED")

    print(f"Kernel time: {primary['kernel_time']:.4f} ms")

    if cuda and cuda['passed'] and cpu['passed']:
        speedup = cpu['kernel_time'] / cuda['kernel_time']
        print(f"CPU: {cpu['kernel_time']:.2f} ms | CUDA: {cuda['kernel_time']:.2f} ms | Speedup: {speedup:.1f}x")

    if not primary['passed']:
        sys.exit(1)

if __name__ == '__main__':
    main()
