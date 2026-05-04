#!/usr/bin/env python3
"""
cuSZp GPU Lossy Compression Benchmark -- CUDA-Hercules Class 3

Builds cuSZp from source and runs 1D float32 compression/decompression
across 3 encoding modes (fixed, plain, outlier) on 2GB synthetic data.

Reference: cuSZp (SC'23 / SC'25) — error-bounded lossy compression.
"""

import os
import sys
import re
import subprocess

# ── Paths ──────────────────────────────────────────────────────────────

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, 'src')
BUILD_DIR = os.path.join(SRC_DIR, 'cmake-build-release')
EXECUTABLE = os.path.join(BUILD_DIR, 'cuSZp_bench')

CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '')

# ── Build ──────────────────────────────────────────────────────────────

def _check_no_placeholders():
    """The 6 src/cuSZp_kernels_*.cu files start as `#error` placeholders so
    candidate solutions cannot silently inherit the reference. `python run.py`
    is meant to run against a candidate that has REPLACED every placeholder.
    The SC'23 reference baseline is measured separately via
    `scripts/eval_cuszp_toolaug.py` / `scripts/replay_cuszp_planc.py`, which
    work in an isolated tmp dir — they NEVER mutate this src/ tree. If you
    want to baseline from the command line, use those tools (or copy
    reference/*.cu to a tmp src/ yourself); do NOT auto-overlay here, as it
    would leave the reference impl in src/ on disk and let any later candidate
    test silently inherit it.
    """
    for f in os.listdir(SRC_DIR):
        if f.startswith('cuSZp_kernels_') and f.endswith('.cu'):
            path = os.path.join(SRC_DIR, f)
            with open(path) as fh:
                if 'CUDA_HERCULES_PLACEHOLDER' in fh.read():
                    sys.exit(
                        f"ERROR: src/{f} still contains a #error placeholder.\n"
                        f"Candidate solutions must rewrite all six "
                        f"src/cuSZp_kernels_*.cu files before `python run.py` "
                        f"can build. To measure the SC'23 reference baseline, "
                        f"use scripts/eval_cuszp_toolaug.py or "
                        f"scripts/replay_cuszp_planc.py — they overlay "
                        f"reference/*.cu in an isolated tmp dir.")


def build():
    """Build cuSZp benchmark from source."""
    _check_no_placeholders()
    if os.path.isfile(EXECUTABLE):
        print(f"Binary exists: {EXECUTABLE}")
        return

    print("Building cuSZp...", flush=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.check_call(
        ['cmake', '..', '-DCMAKE_BUILD_TYPE=Release'],
        cwd=BUILD_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(
        ['make', f'-j{os.cpu_count()}'],
        cwd=BUILD_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print("Build complete.", flush=True)

# ── Run ────────────────────────────────────────────────────────────────

def run():
    """Run the benchmark and parse results."""
    env = os.environ.copy()
    if CUDA_DEVICE:
        env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE

    result = subprocess.run(
        [EXECUTABLE], capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + result.stderr

    # Print raw output for debugging
    print(output)

    # Parse results
    passed = 'Passed' in output and result.returncode == 0

    # Parse kernel time
    m = re.search(r'Kernel time:\s*([0-9.]+)\s*ms', output)
    kernel_time = float(m.group(1)) if m else -1

    # Parse per-mode results
    modes = re.findall(
        r'(\w+):\s+cmp\s+([0-9.]+)\s+GB/s,\s+dec\s+([0-9.]+)\s+GB/s,\s+'
        r'ratio\s+([0-9.]+)x,\s+(PASS|FAIL)',
        output)

    return passed, kernel_time, modes, result.returncode

# ── Main ──────────────────────────────────────────────────────────────

def main():
    build()

    print("\n=== cuSZp Benchmark: 1D/2D/3D x f32/f64, 2GB, REL 1E-2 ===\n", flush=True)

    passed, kernel_time, modes, rc = run()

    print(f"\n=== Summary ===")
    if passed:
        print("Passed")
    else:
        print("FAILED")

    print(f"Kernel time: {kernel_time:.4f} ms")

    for name, cmp_gbs, dec_gbs, ratio, status in modes:
        print(f"  {name}: cmp {cmp_gbs} GB/s, dec {dec_gbs} GB/s, ratio {ratio}x, {status}")

    if not passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
