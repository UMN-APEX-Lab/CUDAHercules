#pragma once
#include <cuda_bf16.h>

// Naive Mamba2 forward: causal Q@K^T * decay @ V per chunk
// q: [B, 1, N, 64] bf16 (shared across heads)
// k: [B, 1, N, 64] bf16 (shared across heads)
// v: [B, H, N, 64] bf16
// o: [B, H, N, 64] bf16
// A: [B, H, N] float (log-space decay factors per token)
// D=64 hardcoded, chunk_size=64

__global__ void mamba2_naive_kernel(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
    __nv_bfloat16 *O, const float *A,
    int B, int H, int N)
{
    constexpr int D = 64;
    constexpr int CHUNK = 64;

    // Each block handles one (batch, head, chunk)
    int idx = blockIdx.x;
    int num_chunks = N / CHUNK;
    int b = idx / (H * num_chunks);
    int rem = idx % (H * num_chunks);
    int h = rem / num_chunks;
    int c = rem % num_chunks;

    if (b >= B) return;

    int chunk_start = c * CHUNK;
    int tid = threadIdx.x; // covers output row within chunk

    if (tid >= CHUNK) return;

    int out_row = chunk_start + tid;

    // Pointers
    // Q: [B, 1, N, D] -> q_base = B*1*N*D
    const __nv_bfloat16 *q_row = Q + ((long long)b * N + out_row) * D;
    // K: [B, 1, N, D]
    const __nv_bfloat16 *k_base = K + (long long)b * N * D;
    // V: [B, H, N, D]
    const __nv_bfloat16 *v_base = V + ((long long)b * H + h) * N * D;
    // A: [B, H, N]
    const float *a_base = A + ((long long)b * H + h) * N;
    // O: [B, H, N, D]
    __nv_bfloat16 *o_row = O + (((long long)b * H + h) * N + out_row) * D;

    // Compute cumulative sum of A for this chunk
    // a_cumsum[i] = sum(A[chunk_start..chunk_start+i])
    float a_cumsum_row = 0.0f;
    for (int i = 0; i <= tid; i++) {
        a_cumsum_row += a_base[chunk_start + i];
    }

    // Accumulate output: o[row, d] = sum over col in [chunk_start..out_row] of
    //   exp(a_cumsum[row] - a_cumsum[col]) * (q[row] dot k[col]) * v[col, d]
    // This is the intra-chunk (diagonal block) computation only.
    // For a naive baseline, we skip the inter-chunk recurrence (KV state).
    float acc[D];
    for (int d = 0; d < D; d++) acc[d] = 0.0f;

    float a_cumsum_col = 0.0f;
    for (int j = 0; j <= tid; j++) {
        int col = chunk_start + j;
        if (j > 0) a_cumsum_col += a_base[chunk_start + j];
        else a_cumsum_col = a_base[chunk_start];

        float decay = expf(a_cumsum_row - a_cumsum_col);

        // dot product q[row] . k[col]
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_base[col * D + d]);
        }
        dot *= decay;

        for (int d = 0; d < D; d++) {
            acc[d] += dot * __bfloat162float(v_base[col * D + d]);
        }
    }

    for (int d = 0; d < D; d++) {
        o_row[d] = __float2bfloat16(acc[d]);
    }
}

void Mamba2Forward(
    __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V,
    __nv_bfloat16 *O, float *A,
    int B, int H, int N)
{
    constexpr int CHUNK = 64;
    int num_chunks = N / CHUNK;
    int num_blocks = B * H * num_chunks;
    mamba2_naive_kernel<<<num_blocks, CHUNK>>>(Q, K, V, O, A, B, H, N);
}
