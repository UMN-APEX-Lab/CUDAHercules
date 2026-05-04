#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Naive FMHA: O = softmax(Q * K^T / sqrt(head_dim)) * V
// Q: [batch, num_heads, seq_q, head_dim]
// K: [batch, num_heads, seq_k, head_dim]
// V: [batch, num_heads, seq_k, head_dim]
// O: [batch, num_heads, seq_q, head_dim]
__global__ void naive_fmha_kernel(
    int batch, int num_heads, int seq_q, int seq_k, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = batch * num_heads * seq_q;
  if (idx >= total) return;

  int tmp = idx;
  int q_pos = tmp % seq_q; tmp /= seq_q;
  int h = tmp % num_heads; tmp /= num_heads;
  int b = tmp;

  int q_head_stride = seq_q * head_dim;
  int k_head_stride = seq_k * head_dim;
  int q_batch_stride = num_heads * q_head_stride;
  int k_batch_stride = num_heads * k_head_stride;

  int q_base = b * q_batch_stride + h * q_head_stride + q_pos * head_dim;
  int k_base = b * k_batch_stride + h * k_head_stride;

  float scale = 1.0f / sqrtf((float)head_dim);

  // Compute S = Q[q_pos] * K^T, scaled
  // Use register-based approach for small seq_k
  float max_val = -1e30f;

  // First pass: compute scores and find max
  // We'll do this in two passes to avoid allocating dynamic memory
  // Pass 1: find max
  for (int j = 0; j < seq_k; ++j) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + j * head_dim + d]);
    }
    dot *= scale;
    if (dot > max_val) max_val = dot;
  }

  // Pass 2: compute exp and sum
  float sum_exp = 0.0f;
  for (int j = 0; j < seq_k; ++j) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + j * head_dim + d]);
    }
    dot *= scale;
    sum_exp += expf(dot - max_val);
  }

  float inv_sum = 1.0f / sum_exp;

  // Pass 3: compute output O = softmax * V
  for (int d = 0; d < head_dim; ++d) {
    float acc = 0.0f;
    for (int j = 0; j < seq_k; ++j) {
      float dot = 0.0f;
      for (int dd = 0; dd < head_dim; ++dd) {
        dot += __half2float(Q[q_base + dd]) * __half2float(K[k_base + j * head_dim + dd]);
      }
      dot *= scale;
      float attn = expf(dot - max_val) * inv_sum;
      acc += attn * __half2float(V[k_base + j * head_dim + d]);
    }
    int o_idx = b * q_batch_stride + h * q_head_stride + q_pos * head_dim + d;
    O[o_idx] = __float2half(acc);
  }
}

cudaError_t BlackwellFmha(
    int batch, int num_heads, int seq_q, int seq_k, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  int total = batch * num_heads * seq_q;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  naive_fmha_kernel<<<blocks, threads>>>(
    batch, num_heads, seq_q, seq_k, head_dim, Q, K, V, O);
  return cudaGetLastError();
}
