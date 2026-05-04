#pragma once
#include <cuda_bf16.h>
#include <cmath>

// Naive linear attention with exponential decay
// q: [B, H, N, D] bf16
// k: [B, H, N, D] bf16
// v: [B, H, N, D] bf16
// o: [B, H, N, D] bf16
// slopes: [H] float
// Computes: o[i] = sum_{j<=i} exp(-slope * (i-j)) * (q[i] . k[j]) * v[j]

__global__ void linear_attention_naive_kernel(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
    __nv_bfloat16 *O, const float *slopes,
    int B, int H, int N, int D)
{
    // Each block handles one (batch, head), each thread handles one output row
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    if (b >= B) return;

    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float slope = slopes[h];

    long long base_offset = ((long long)b * H + h) * N * D;
    const __nv_bfloat16 *q_row = Q + base_offset + (long long)row * D;
    const __nv_bfloat16 *k_base = K + base_offset;
    const __nv_bfloat16 *v_base = V + base_offset;
    __nv_bfloat16 *o_row = O + base_offset + (long long)row * D;

    // Accumulate in float
    float acc[128]; // D <= 128
    for (int d = 0; d < D; d++) acc[d] = 0.0f;

    for (int j = 0; j <= row; j++) {
        float decay = expf(-slope * (float)(row - j));

        // dot product q[row] . k[j]
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_base[j * D + d]);
        }
        dot *= decay;

        for (int d = 0; d < D; d++) {
            acc[d] += dot * __bfloat162float(v_base[j * D + d]);
        }
    }

    for (int d = 0; d < D; d++) {
        o_row[d] = __float2bfloat16(acc[d]);
    }
}

void LinearAttention(
    __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
    float *slopes,
    int B, int H, int N, int D)
{
    int threads_per_block = 64;
    int rows_per_grid = (N + threads_per_block - 1) / threads_per_block;
    dim3 grid(B * H, rows_per_grid);
    linear_attention_naive_kernel<<<grid, threads_per_block>>>(Q, K, V, O, slopes, B, H, N, D);
}
