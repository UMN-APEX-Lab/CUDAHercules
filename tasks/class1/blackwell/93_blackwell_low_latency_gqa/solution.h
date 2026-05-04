#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Naive GQA decoding kernel
// Q: [batch, q_heads, 1, head_dim]
// K: [batch, kv_heads, seq_len, head_dim]
// V: [batch, kv_heads, seq_len, head_dim]
// O: [batch, q_heads, 1, head_dim]
__global__ void naive_gqa_kernel(
    int batch, int kv_heads, int q_heads,
    int seq_len, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_qheads = batch * q_heads;
  if (idx >= total_qheads) return;

  int b = idx / q_heads;
  int qh = idx % q_heads;
  int q_per_kv = q_heads / kv_heads;
  int kvh = qh / q_per_kv;
  float scale = 1.0f / sqrtf((float)head_dim);

  int q_off = ((b * q_heads + qh) * 1) * head_dim;

  // Compute attention scores
  float max_score = -1e30f;
  for (int t = 0; t < seq_len; ++t) {
    int k_off = ((b * kv_heads + kvh) * seq_len + t) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_off + d]) * __half2float(K[k_off + d]);
    }
    float score = dot * scale;
    if (score > max_score) max_score = score;
  }

  // Softmax and weighted sum
  float sum_exp = 0.0f;
  for (int t = 0; t < seq_len; ++t) {
    int k_off = ((b * kv_heads + kvh) * seq_len + t) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_off + d]) * __half2float(K[k_off + d]);
    }
    sum_exp += expf(dot * scale - max_score);
  }

  int o_off = ((b * q_heads + qh) * 1) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      int k_off = ((b * kv_heads + kvh) * seq_len + t) * head_dim;
      float dot = 0.0f;
      for (int dd = 0; dd < head_dim; ++dd) {
        dot += __half2float(Q[q_off + dd]) * __half2float(K[k_off + dd]);
      }
      float weight = expf(dot * scale - max_score) / sum_exp;
      int v_off = ((b * kv_heads + kvh) * seq_len + t) * head_dim;
      acc += weight * __half2float(V[v_off + d]);
    }
    O[o_off + d] = __float2half(acc);
  }
}

cudaError_t BlackwellGQA(
    int batch, int kv_heads, int q_heads,
    int seq_len, int head_dim,
    __half const *Q, __half const *K, __half const *V, __half *O) {

  int total = batch * q_heads;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  naive_gqa_kernel<<<blocks, threads>>>(batch, kv_heads, q_heads, seq_len, head_dim, Q, K, V, O);
  return cudaGetLastError();
}
