#!/usr/bin/env python3
"""
FlashAttention-4 Backward Pass Benchmark -- CUDA-Hercules Class 4

Evaluates a hand-written CUDA implementation of FA4 backward pass against
PyTorch scaled_dot_product_attention autograd.

Requires: NVIDIA B200/GB200 GPU (SM100), CUDA 13.0+, PyTorch.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    'CUDA_HERCULES_ROOT',
    os.path.abspath(os.path.join(TASK_DIR, '..', '..', '..')),
)
EXT_BUILD_DIR = os.path.join(TASK_DIR, '.torch_extensions', 'fa4_bwd_solution')

# Test configurations: (batch, seqlen_q, seqlen_k, num_heads, num_heads_k, head_dim, causal)
TEST_CONFIGS = [
    (2, 1024, 1024, 16, 16, 128, False),
    (2, 1024, 1024, 16, 16, 128, True),
    (1, 2048, 2048, 32, 32, 128, False),
    (1, 4096, 4096, 32, 8, 128, True),
]

WARMUP_ITERS = 2
BENCH_ITERS = 5
ATOL = 0.05
RTOL = 0.05


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
    return load(
        name='fa4_bwd_solution',
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


def _repeat_kv_for_gqa(K, V, num_heads):
    if K.shape[2] == num_heads:
        return K, V
    n_rep = num_heads // K.shape[2]
    return K.repeat_interleave(n_rep, dim=2), V.repeat_interleave(n_rep, dim=2)


def pytorch_reference_forward(Q, K, V, causal):
    """PyTorch SDPA forward as saved-state reference."""
    K_full, V_full = _repeat_kv_for_gqa(K, V, Q.shape[2])
    q = Q.transpose(1, 2)
    k = K_full.transpose(1, 2)
    v = V_full.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


@torch.no_grad()
def compute_lse(Q, K, causal, block_q=128):
    """Compute [B, H, Nq] log-sum-exp without materializing the full score matrix."""
    B, Nq, H, D = Q.shape
    _, Nk, Hk, _ = K.shape
    scale = D ** -0.5
    K_full, _ = _repeat_kv_for_gqa(K, K, H)
    q = Q.transpose(1, 2).float()
    k = K_full.transpose(1, 2).float()
    parts = []
    key_idx = torch.arange(Nk, device=Q.device)

    for start in range(0, Nq, block_q):
        end = min(start + block_q, Nq)
        scores = torch.einsum('bhqd,bhkd->bhqk', q[:, :, start:end], k) * scale
        if causal:
            query_idx = torch.arange(start, end, device=Q.device)[:, None]
            mask = key_idx[None, :] > query_idx
            scores = scores.masked_fill(mask[None, None], float('-inf'))
        parts.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(parts, dim=-1).contiguous()


def pytorch_reference_backward(Q, K, V, dO, causal):
    """Return dQ, dK, dV from PyTorch SDPA autograd."""
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    O_ref = pytorch_reference_forward(Q_ref, K_ref, V_ref, causal)
    O_ref.backward(dO)
    return Q_ref.grad, K_ref.grad, V_ref.grad


def benchmark_fn(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
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


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _allclose(a, b):
    return torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only', action='store_true',
                        help='Compile solution.cu and exit without running benchmarks.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"GPU: {torch.cuda.get_device_name()}, SM {cap[0]}.{cap[1]}")
    if sm < 100:
        print(f"WARNING: This task requires SM100 (B200/GB200). Current GPU is SM{sm}.")
        print("Correctness testing may work but performance numbers won't be meaningful.")

    solution = compile_solution()
    if args.build_only:
        print("Build passed")
        return

    device = torch.device('cuda')
    all_passed = True
    total_solution_ms = 0.0
    total_baseline_ms = 0.0

    print("\n=== FlashAttention-4 Backward Benchmark ===\n")

    for cfg in TEST_CONFIGS:
        B, Nq, Nk, H, Hk, D, causal = cfg
        tag = f"B={B} Nq={Nq} Nk={Nk} H={H} Hk={Hk} D={D} {'causal' if causal else 'full'}"
        print(f"--- {tag} ---")

        Q = torch.randn(B, Nq, H, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(B, Nk, Hk, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(B, Nk, Hk, D, dtype=torch.bfloat16, device=device)
        dO = torch.randn(B, Nq, H, D, dtype=torch.bfloat16, device=device)

        O = pytorch_reference_forward(Q, K, V, causal).detach()
        LSE = compute_lse(Q, K, causal)
        dQ_ref, dK_ref, dV_ref = pytorch_reference_backward(Q, K, V, dO, causal)

        try:
            grads = solution.flash_attn4_bwd(Q, K, V, O, dO, LSE, causal)
            if len(grads) != 3:
                raise RuntimeError(f"Expected 3 gradients, got {len(grads)}")
            dQ_sol, dK_sol, dV_sol = grads
            correct = _allclose(dQ_sol, dQ_ref) and _allclose(dK_sol, dK_ref) and _allclose(dV_sol, dV_ref)
            max_diff = max(_max_diff(dQ_sol, dQ_ref), _max_diff(dK_sol, dK_ref), _max_diff(dV_sol, dV_ref))
        except Exception as e:
            print(f"  Solution ERROR: {e}")
            correct = False
            max_diff = float('inf')

        if not correct:
            all_passed = False

        sol_ms = benchmark_fn(solution.flash_attn4_bwd, Q, K, V, O, dO, LSE, causal)
        total_solution_ms += sol_ms

        base_ms = benchmark_fn(pytorch_reference_backward, Q, K, V, dO, causal, warmup=1, iters=3)
        total_baseline_ms += base_ms
        ratio = base_ms / sol_ms if sol_ms > 0 else 0
        print(f"  Solution: {sol_ms:.2f} ms | PyTorch: {base_ms:.2f} ms | "
              f"Ratio: {ratio:.2f}x | maxdiff: {max_diff:.4f} | "
              f"{'PASS' if correct else 'FAIL'}")

    print("\n=== Summary ===")
    if all_passed:
        print("Passed")
    else:
        print("FAILED")

    print(f"Kernel time: {total_solution_ms:.4f} ms")
    print(f"PyTorch baseline: {total_baseline_ms:.4f} ms")
    if total_solution_ms > 0:
        print(f"Solution / baseline ratio: {total_baseline_ms / total_solution_ms:.2f}x")

    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
