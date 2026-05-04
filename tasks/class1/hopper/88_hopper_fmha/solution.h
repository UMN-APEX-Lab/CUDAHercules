#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Naive FMHA kernel for testing the harness
// Q: [batch, num_heads, seq_q, head_dim] row-major
// K: [batch, num_heads, seq_k, head_dim] row-major
// V: [batch, num_heads, seq_k, head_dim] row-major
// O: [batch, num_heads, seq_q, head_dim] row-major
// O = softmax(Q * K^T / sqrt(head_dim)) * V

__global__ void naive_fmha_kernel(
    int batch, int num_heads, int seq_q, int seq_k, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  // Each thread handles one (batch, head, query_row)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * num_heads * seq_q;
  if (idx >= total) return;

  int b = idx / (num_heads * seq_q);
  int rem = idx % (num_heads * seq_q);
  int h = rem / seq_q;
  int q = rem % seq_q;

  int q_head_stride = seq_q * head_dim;
  int k_head_stride = seq_k * head_dim;
  int q_batch_stride = num_heads * q_head_stride;
  int k_batch_stride = num_heads * k_head_stride;

  int q_base = b * q_batch_stride + h * q_head_stride + q * head_dim;
  int k_base = b * k_batch_stride + h * k_head_stride;

  float scale = 1.0f / sqrtf((float)head_dim);

  // Compute S[q, :] = Q[q, :] * K^T * scale (dynamic allocation on stack for small seq)
  // Use external memory since seq_k can be large
  extern __shared__ float shared[];
  // Not using shared memory for simplicity; use register-based approach
  // For large seq_k this is slow but correct

  // Pass 1: compute max
  float max_val = -1e30f;
  for (int j = 0; j < seq_k; ++j) {
    float dot = 0;
    for (int d = 0; d < head_dim; ++d)
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + j * head_dim + d]);
    dot *= scale;
    if (dot > max_val) max_val = dot;
  }

  // Pass 2: compute sum of exp
  float sum_exp = 0;
  for (int j = 0; j < seq_k; ++j) {
    float dot = 0;
    for (int d = 0; d < head_dim; ++d)
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + j * head_dim + d]);
    dot *= scale;
    sum_exp += expf(dot - max_val);
  }

  // Pass 3: compute O = softmax(S) * V
  int o_base = b * q_batch_stride + h * q_head_stride + q * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    float acc = 0;
    for (int j = 0; j < seq_k; ++j) {
      float dot = 0;
      for (int dd = 0; dd < head_dim; ++dd)
        dot += __half2float(Q[q_base + dd]) * __half2float(K[k_base + j * head_dim + dd]);
      dot *= scale;
      float w = expf(dot - max_val) / sum_exp;
      acc += w * __half2float(V[k_base + j * head_dim + d]);
    }
    O[o_base + d] = __float2half(acc);
  }
}

cudaError_t HopperFmha(
    int batch, int num_heads, int seq_q, int seq_k, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  int total = batch * num_heads * seq_q;
  int threads = 128;
  int blocks = (total + threads - 1) / threads;
  naive_fmha_kernel<<<blocks, threads>>>(
    batch, num_heads, seq_q, seq_k, head_dim, Q, K, V, O);
  return cudaGetLastError();
}
