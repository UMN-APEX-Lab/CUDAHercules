#!/usr/bin/env python3
"""
MGG Multi-GPU GCN Benchmark -- CUDA-Hercules Class 3

Builds MGG from source and runs 4-GPU GCN inference on ogbn-papers100M,
measuring end-to-end throughput and per-layer SpMM kernel time.

Reference: MGG (OSDI'23) — NVSHMEM-based multi-GPU GNN with
fine-grained intra-kernel communication-computation pipelining.
"""

import os
import sys
import re
import subprocess
import shutil

# ── Configuration ──────────────────────────────────────────────────────

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, 'src')
BUILD_DIR = os.path.join(SRC_DIR, 'build')
EXECUTABLE = os.path.join(BUILD_DIR, 'mgg_gcn')

DATA_DIR = os.environ.get('MGG_DATA_DIR', os.path.join(TASK_DIR, 'data'))
BEG_FILE = os.path.join(DATA_DIR, 'bin', 'paper100M_beg_pos.bin')
CSR_FILE = os.path.join(DATA_DIR, 'bin', 'paper100M_csr.bin')
WEIGHT_FILE = os.path.join(DATA_DIR, 'bin', 'paper100M_weight.bin')

# Model parameters (from MGG bench_MGG.py for paper100M)
NUM_GPUS = 4
PART_SIZE = 16
WARP_PER_BLOCK = 4
INTERLEAVED_DIST = 16
INPUT_DIM = 128
HIDDEN_DIM = 16
OUTPUT_DIM = 64

# GPU selection: use first 4 GPUs
CUDA_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')

# NVSHMEM symmetric heap (20GB per PE for paper100M)
NVSHMEM_SYMMETRIC_SIZE = os.environ.get('NVSHMEM_SYMMETRIC_SIZE', '21474836480')

# ── Build ──────────────────────────────────────────────────────────────

def build():
    """Build MGG via CMake."""
    if os.path.isfile(EXECUTABLE):
        print(f"Binary exists: {EXECUTABLE}")
        return True

    print("Building MGG...", flush=True)
    os.makedirs(BUILD_DIR, exist_ok=True)

    env = os.environ.copy()
    try:
        result = subprocess.run(
            ['cmake', '..'],
            cwd=BUILD_DIR, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"CMake failed:\n{result.stderr[-2000:]}")
            return False

        result = subprocess.run(
            ['make', '-j'],
            cwd=BUILD_DIR, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr[-2000:]}")
            return False
    except FileNotFoundError as e:
        print(f"Build error: {e}")
        return False

    print("Build complete.", flush=True)
    return True

# ── Data check ────────────────────────────────────────────────────────

def check_data():
    """Verify binary data files exist."""
    for f in [BEG_FILE, CSR_FILE, WEIGHT_FILE]:
        if not os.path.isfile(f):
            print(f"Data file missing: {f}")
            print(f"Run: bash prepare_data.sh {DATA_DIR}")
            return False
    return True

# ── Run ────────────────────────────────────────────────────────────────

def run():
    """Run MGG with mpirun and parse results."""
    cmd = [
        'mpirun', '--allow-run-as-root', '-np', str(NUM_GPUS),
        EXECUTABLE,
        BEG_FILE, CSR_FILE, WEIGHT_FILE,
        str(NUM_GPUS),
        str(PART_SIZE),
        str(WARP_PER_BLOCK),
        str(INTERLEAVED_DIST),
        str(INPUT_DIM),
        str(HIDDEN_DIM),
        str(OUTPUT_DIM),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICES
    env['NVSHMEM_SYMMETRIC_SIZE'] = NVSHMEM_SYMMETRIC_SIZE
    env['OMPI_MCA_plm_rsh_agent'] = 'sh'

    print(f"Running: {' '.join(cmd[:6])} ...", flush=True)
    print(f"  GPUs: {CUDA_DEVICES}, NVSHMEM heap: {int(NVSHMEM_SYMMETRIC_SIZE)/1e9:.1f} GB", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + result.stderr
    print(output)

    # Parse per-PE times
    pe_times = re.findall(r'PE-(\d+),\s*Total\s*\(ms\):\s*([0-9.]+)', output)
    if pe_times:
        times = {int(pe): float(ms) for pe, ms in pe_times}
        max_pe_time = max(times.values())
    else:
        max_pe_time = -1

    # Parse MPI time (overall)
    m = re.search(r'MPI time\s*\(ms\)\s*([0-9.]+)', output)
    mpi_time = float(m.group(1)) if m else -1

    # Parse preprocessing time
    m = re.search(r'Preproc\s*\(ms\):\s*([0-9.]+)', output)
    preproc_time = float(m.group(1)) if m else -1

    return {
        'returncode': result.returncode,
        'pe_times': times if pe_times else {},
        'max_pe_time_ms': max_pe_time,
        'mpi_time_ms': mpi_time,
        'preproc_time_ms': preproc_time,
        'output': output,
    }

# ── Main ──────────────────────────────────────────────────────────────

def main():
    if not check_data():
        sys.exit(1)

    if not build():
        sys.exit(1)

    print(f"\n=== MGG 4-GPU GCN: ogbn-papers100M, dim={INPUT_DIM}, hidden={HIDDEN_DIM}, out={OUTPUT_DIM} ===\n",
          flush=True)

    r = run()

    passed = r['returncode'] == 0 and r['mpi_time_ms'] > 0

    print(f"\n=== Summary ===")
    if passed:
        print("Passed")
    else:
        print("FAILED")

    # Kernel time = MPI time (end-to-end per iteration, 100 iterations averaged in MGG)
    kernel_time = r['mpi_time_ms']
    print(f"Kernel time: {kernel_time:.4f} ms")
    print(f"  max PE time: {r['max_pe_time_ms']:.2f} ms")
    print(f"  preproc: {r['preproc_time_ms']:.2f} ms")
    for pe, t in sorted(r.get('pe_times', {}).items()):
        print(f"  PE-{pe}: {t:.2f} ms")

    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
