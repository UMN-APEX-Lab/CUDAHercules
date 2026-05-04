#!/usr/bin/env python3
"""
FlashAttention-4 Forward Pass Benchmark -- CUDA-Hercules Class 4

Evaluates a hand-written CUDA implementation of FA4 forward pass against:
  1. PyTorch scaled_dot_product_attention (correctness reference)
  2. FA4 CuTe DSL implementation (performance baseline)

Requires: NVIDIA B200/GB200 GPU (SM100), CUDA 13.0+, PyTorch, flash-attn package.
"""

import os
import sys
import time
import subprocess
import argparse

import torch
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────────────

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
PROJECT_ROOT = os.environ.get(
    'CUDA_HERCULES_ROOT',
    os.path.abspath(os.path.join(TASK_DIR, '..', '..', '..')),
)
EXT_BUILD_DIR = os.path.join(TASK_DIR, '.torch_extensions', 'fa4_fwd_solution')

# Test configurations: (batch, seqlen_q, seqlen_k, num_heads, num_heads_k, head_dim, causal)
TEST_CONFIGS = [
    # Small (warmup / correctness)
    (2, 1024, 1024, 16, 16, 128, False),
    (2, 1024, 1024, 16, 16, 128, True),
    # Medium
    (1, 4096, 4096, 32, 32, 128, False),
    (1, 4096, 4096, 32, 8, 128, True),   # GQA: 32 heads, 8 KV heads
    # Large
    (1, 8192, 8192, 32, 32, 128, True),
    (1, 16384, 16384, 16, 16, 128, True),
]

WARMUP_ITERS = 3
BENCH_ITERS = 10
ATOL = 0.01
RTOL = 0.01

# ── Solution compilation ──────────────────────────────────────────────

def compile_solution():
    """Compile solution.cu into a PyTorch extension."""
    from torch.utils.cpp_extension import load
    solution_cu = os.path.join(TASK_DIR, 'solution.cu')
    if not os.path.isfile(solution_cu):
        print("ERROR: solution.cu not found")
        sys.exit(1)

    os.makedirs(EXT_BUILD_DIR, exist_ok=True)
    include_dirs = [
        os.path.join(PROJECT_ROOT, 'reference_sources', 'cutlass', 'include'),
        os.path.join(PROJECT_ROOT, 'reference_sources', 'cutlass', 'tools', 'util', 'include'),
    ]
    include_dirs = [p for p in include_dirs if os.path.isdir(p)]

    print("Compiling solution.cu...", flush=True)
    mod = load(
        name='fa4_fwd_solution',
        sources=[solution_cu],
        build_directory=EXT_BUILD_DIR,
        extra_include_paths=include_dirs,
        extra_cuda_cflags=[
            '-O3', '--use_fast_math',
            '-gencode', 'arch=compute_100,code=sm_100',
            '-gencode', 'arch=compute_100,code=compute_100',
        ],
        verbose=False,
    )
    return mod

# ── PyTorch reference ─────────────────────────────────────────────────

def pytorch_reference(Q, K, V, causal):
    """PyTorch SDPA as correctness reference."""
    # Q: [B, N_q, H, D] -> [B, H, N_q, D]
    q = Q.transpose(1, 2)
    k = K.transpose(1, 2)
    v = V.transpose(1, 2)
    # Handle GQA: repeat KV heads
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # [B, N_q, H, D]

# ── FA4 CuTe DSL baseline ────────────────────────────────────────────

def fa4_baseline(Q, K, V, causal):
    """FA4 CuTe DSL implementation via flash-attn package."""
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func(Q, K, V, causal=causal)
    except ImportError:
        return None

# ── Benchmark ─────────────────────────────────────────────────────────

def benchmark_fn(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a function with CUDA event timing."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only', action='store_true',
                        help='Compile solution.cu and exit without running benchmarks.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    device = torch.device('cuda')

    # Check SM version
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"GPU: {torch.cuda.get_device_name()}, SM {cap[0]}.{cap[1]}")
    if sm < 100:
        print(f"WARNING: This task requires SM100 (B200/GB200). Current GPU is SM{sm}.")
        print("Correctness testing may work but performance numbers won't be meaningful.")

    # Compile solution
    solution = compile_solution()
    if args.build_only:
        print("Build passed")
        return

    all_passed = True
    total_solution_ms = 0
    total_baseline_ms = 0

    print(f"\n=== FlashAttention-4 Forward Benchmark ===\n")

    for cfg in TEST_CONFIGS:
        B, Nq, Nk, H, Hk, D, causal = cfg
        tag = f"B={B} Nq={Nq} Nk={Nk} H={H} Hk={Hk} D={D} {'causal' if causal else 'full'}"
        print(f"--- {tag} ---")

        # Generate inputs
        Q = torch.randn(B, Nq, H, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(B, Nk, Hk, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(B, Nk, Hk, D, dtype=torch.bfloat16, device=device)

        # Reference
        O_ref = pytorch_reference(Q, K, V, causal)

        # Solution
        try:
            O_sol = solution.flash_attn4_fwd(Q, K, V, causal)
            max_diff = (O_sol.float() - O_ref.float()).abs().max().item()
            correct = torch.allclose(O_sol.float(), O_ref.float(), atol=ATOL, rtol=RTOL)
        except Exception as e:
            print(f"  Solution ERROR: {e}")
            correct = False
            max_diff = float('inf')

        if not correct:
            all_passed = False

        # Timing
        sol_ms = benchmark_fn(solution.flash_attn4_fwd, Q, K, V, causal)
        total_solution_ms += sol_ms

        # Baseline
        O_base = fa4_baseline(Q, K, V, causal)
        if O_base is not None:
            base_ms = benchmark_fn(fa4_baseline, Q, K, V, causal)
            total_baseline_ms += base_ms
            # FLOPS
            flops = 4 * B * H * Nq * Nk * D  # 2 matmuls (QK, PV) × 2 (mul+add)
            if causal:
                flops //= 2
            sol_tflops = flops / (sol_ms * 1e-3) / 1e12
            base_tflops = flops / (base_ms * 1e-3) / 1e12
            ratio = base_ms / sol_ms if sol_ms > 0 else 0
            print(f"  Solution: {sol_ms:.2f} ms ({sol_tflops:.0f} TFLOPS) | "
                  f"FA4-CuTe: {base_ms:.2f} ms ({base_tflops:.0f} TFLOPS) | "
                  f"Ratio: {ratio:.2f}x | maxdiff: {max_diff:.4f} | "
                  f"{'PASS' if correct else 'FAIL'}")
        else:
            print(f"  Solution: {sol_ms:.2f} ms | FA4-CuTe: N/A | "
                  f"maxdiff: {max_diff:.4f} | {'PASS' if correct else 'FAIL'}")

    # Summary
    print(f"\n=== Summary ===")
    if all_passed:
        print("Passed")
    else:
        print("FAILED")

    print(f"Kernel time: {total_solution_ms:.4f} ms")
    if total_baseline_ms > 0:
        print(f"FA4-CuTe baseline: {total_baseline_ms:.4f} ms")
        print(f"Solution / baseline ratio: {total_baseline_ms / total_solution_ms:.2f}x")

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
