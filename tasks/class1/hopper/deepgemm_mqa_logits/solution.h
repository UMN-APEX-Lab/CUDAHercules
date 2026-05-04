#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Naive MQA logits kernel.
// Each thread handles one (q, kv) output position.
__global__ void naive_mqa_logits_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ kv,
    const float* __restrict__ kv_scales,
    const float* __restrict__ weights,
    const int* __restrict__ ks,
    const int* __restrict__ ke,
    float* __restrict__ logits,
    int seq_len, int seq_len_kv,
    int num_heads, int head_dim) {

    int q_idx  = blockIdx.y;
    int kv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len || kv_idx >= seq_len_kv) return;

    int k_start = ks[q_idx];
    int k_end   = ke[q_idx];

    if (kv_idx < k_start || kv_idx >= k_end) {
        logits[q_idx * seq_len_kv + kv_idx] = -__FLT_MAX__;
        return;
    }

    float result = 0.0f;
    for (int h = 0; h < num_heads; h++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float qv = float(q[(q_idx * num_heads + h) * head_dim + d]);
            float kk = float(kv[kv_idx * head_dim + d]);
            dot += qv * kk;
        }
        float score = dot > 0.0f ? dot : 0.0f;  // ReLU
        result += score * weights[q_idx * num_heads + h];
    }
    logits[q_idx * seq_len_kv + kv_idx] = result * kv_scales[kv_idx];
}

void solution_mqa_logits(
    const __nv_fp8_e4m3* q, const __nv_fp8_e4m3* kv,
    const float* kv_scales, const float* weights,
    const int* ks, const int* ke,
    float* logits,
    int seq_len, int seq_len_kv,
    int num_heads, int head_dim,
    cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((seq_len_kv + 255) / 256, seq_len);
    naive_mqa_logits_kernel<<<grid, block, 0, stream>>>(
        q, kv, kv_scales, weights, ks, ke, logits,
        seq_len, seq_len_kv, num_heads, head_dim);
}
