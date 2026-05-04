#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive Fused Multi-Head Attention (half-precision) for testing the harness.
//
// Memory layout: BMHK format [batch, seq_len, num_heads, head_dim]
//   Q[b, m, h, k] at Q_base + b * q_strideB + m * q_strideM + h * q_strideH + k
//   K[b, m, h, k] at K_base + b * k_strideB + m * k_strideM + h * k_strideH + k
//   V[b, m, h, k] at V_base + b * v_strideB + m * v_strideM + h * v_strideH + k
//   O[b, m, h, k] at O_base + b * (seq_q * o_strideM) + m * o_strideM + h * head_dim_v + k
//
// Operation per (batch, head):
//   S[i,j] = sum_k(Q[i,k] * K[j,k]) * scale     (scale = 1/sqrt(head_dim))
//   P[i,j] = softmax(S[i,:])_j                   (row-wise softmax)
//   O[i,k] = sum_j(P[i,j] * V[j,k])

__global__ void naive_fmha_kernel(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {

  // Each thread handles one (batch, head, query_row)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_q;
  if (idx >= total) return;

  int b = idx / (num_heads * seq_q);
  int rem = idx % (num_heads * seq_q);
  int h = rem / seq_q;
  int i = rem % seq_q;

  float scale = 1.0f / sqrtf((float)head_dim);

  // Compute Q base offset for this (b, i, h)
  int q_base = b * q_strideB + i * q_strideM + h * q_strideH;
  int k_base_batch = b * k_strideB;
  int v_base_batch = b * v_strideB;

  int n_keys = causal ? min(i + 1, seq_kv) : seq_kv;

  // Compute S[j] = dot(Q[i,:], K[j,:]) * scale and find max
  float max_val = -1e30f;
  // We need to iterate over all keys, so we use a loop
  // First pass: compute scores and max
  // We'll need a temporary array - use registers for small head_dim
  // For simplicity, do three passes

  // Pass 1: find max
  for (int j = 0; j < n_keys; ++j) {
    int k_base = k_base_batch + j * k_strideM + h * k_strideH;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
    }
    float s = dot * scale;
    if (s > max_val) max_val = s;
  }

  // Pass 2: compute sum of exp(S - max)
  float sum_exp = 0.0f;
  for (int j = 0; j < n_keys; ++j) {
    int k_base = k_base_batch + j * k_strideM + h * k_strideH;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
    }
    float s = dot * scale;
    sum_exp += expf(s - max_val);
  }

  float inv_sum = 1.0f / sum_exp;

  // Pass 3: compute O = sum(softmax * V)
  int o_base = b * (seq_q * o_strideM) + i * o_strideM + h * head_dim_v;

  for (int d = 0; d < head_dim_v; ++d) {
    float acc = 0.0f;
    for (int j = 0; j < n_keys; ++j) {
      int k_base = k_base_batch + j * k_strideM + h * k_strideH;
      float dot = 0.0f;
      for (int dk = 0; dk < head_dim; ++dk) {
        dot += __half2float(Q[q_base + dk]) * __half2float(K[k_base + dk]);
      }
      float s = dot * scale;
      float p = expf(s - max_val) * inv_sum;

      int v_base = v_base_batch + j * v_strideM + h * v_strideH;
      acc += p * __half2float(V[v_base + d]);
    }
    O[o_base + d] = __float2half(acc);
  }
}

cudaError_t fmha_solution(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {

  int total = batch_size * num_heads * seq_q;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  naive_fmha_kernel<<<blocks, threads>>>(
    batch_size, seq_q, seq_kv,
    num_heads, head_dim, head_dim_v,
    Q, K, V, O,
    q_strideB, q_strideM, q_strideH,
    k_strideB, k_strideM, k_strideH,
    v_strideB, v_strideM, v_strideH,
    o_strideM,
    causal);

  return cudaGetLastError();
}
