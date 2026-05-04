#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Naive causal attention: Q@K^T @ V (simple baseline, ignoring the linear attention / feature map aspect).
// Q,K: [B,H,N,D_QK=16], V: [B,H,N,D_VO=64], O: [B,H,N,D_VO=64]
// All bf16.

#define BASED_D_QK 16
#define BASED_D_VO 64

__global__ void based_naive_causal_attention_kernel(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K,
    const __nv_bfloat16 *V, __nv_bfloat16 *O,
    int B, int H, int N)
{
    // Each thread handles one output element: (b, h, n, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * BASED_D_VO;
    if (idx >= total) return;

    int d  = idx % BASED_D_VO;
    int n  = (idx / BASED_D_VO) % N;
    int h  = (idx / (BASED_D_VO * N)) % H;
    int b  = idx / (BASED_D_VO * N * H);

    // Base pointers for this (b,h)
    const __nv_bfloat16 *q_bh = Q + ((b * H + h) * N) * BASED_D_QK;
    const __nv_bfloat16 *k_bh = K + ((b * H + h) * N) * BASED_D_QK;
    const __nv_bfloat16 *v_bh = V + ((b * H + h) * N) * BASED_D_VO;

    float acc = 0.0f;
    float score_sum = 0.0f;

    // Causal: attend to positions 0..n
    for (int m = 0; m <= n; m++) {
        // Compute Q[n] . K[m]
        float dot = 0.0f;
        for (int dk = 0; dk < BASED_D_QK; dk++) {
            dot += __bfloat162float(q_bh[n * BASED_D_QK + dk]) *
                   __bfloat162float(k_bh[m * BASED_D_QK + dk]);
        }
        float score = dot / sqrtf((float)BASED_D_QK);
        acc += score * __bfloat162float(v_bh[m * BASED_D_VO + d]);
        score_sum += score;
    }

    O[((b * H + h) * N + n) * BASED_D_VO + d] = __float2bfloat16(acc);
}

void BasedLinearAttention(
    __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
    int B, int H, int N)
{
    int total = B * H * N * BASED_D_VO;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    based_naive_causal_attention_kernel<<<blocks, threads>>>(Q, K, V, O, B, H, N);
    cudaDeviceSynchronize();
}
