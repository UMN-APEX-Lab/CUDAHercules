#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Naive Rotary Position Embeddings (RoPE) kernel
// For each position, splits head dim in half and applies rotation:
//   out[..., :D/2]  = x[..., :D/2] * cos - x[..., D/2:] * sin
//   out[..., D/2:]  = x[..., D/2:] * cos + x[..., :D/2] * sin
// sin_table/cos_table shape: [N, D/2]
// x shape: [B, H, N, D]

__global__ void naive_fused_rotary_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *sin_table,
    const __nv_bfloat16 *cos_table,
    int B, int H, int N, int D)
{
    int D2 = D / 2;
    // Each thread handles one element in the head dimension
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * D;
    if (idx >= total) return;

    // Decode indices: x is [B, H, N, D]
    int d = idx % D;
    int n = (idx / D) % N;
    // b*H + h is encoded in the upper bits
    int bh = idx / (N * D);

    // Lookup sin/cos for this position and dimension
    // sin_table and cos_table are [N, D/2]
    float x_val = __bfloat162float(x[idx]);
    float result;

    if (d < D2) {
        // First half: out = x1 * cos - x2 * sin
        float cos_val = __bfloat162float(cos_table[n * D2 + d]);
        float sin_val = __bfloat162float(sin_table[n * D2 + d]);
        float x2_val  = __bfloat162float(x[bh * N * D + n * D + d + D2]);
        result = x_val * cos_val - x2_val * sin_val;
    } else {
        // Second half: out = x2 * cos + x1 * sin
        int d2 = d - D2;
        float cos_val = __bfloat162float(cos_table[n * D2 + d2]);
        float sin_val = __bfloat162float(sin_table[n * D2 + d2]);
        float x1_val  = __bfloat162float(x[bh * N * D + n * D + d2]);
        result = x_val * cos_val + x1_val * sin_val;
    }

    out[idx] = __float2bfloat16(result);
}

void FusedRotary(
    __nv_bfloat16 *out, __nv_bfloat16 *x,
    __nv_bfloat16 *sin_table, __nv_bfloat16 *cos_table,
    int B, int H, int N, int D)
{
    int total = B * H * N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    naive_fused_rotary_kernel<<<blocks, threads>>>(
        out, x, sin_table, cos_table, B, H, N, D);
    cudaDeviceSynchronize();
}
