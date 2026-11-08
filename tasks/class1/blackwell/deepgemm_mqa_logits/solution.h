#pragma once
// Reference-free "naive but not slow" MQA-logits kernel for the harness. One CTA of 128 threads per
// (query row, tile of 128 KV positions): the query row (64 heads x 128) is dequantized to FP32 in
// shared memory, every thread dequantizes one KV row to registers and runs the 64 heads' dot products
// in FP32, applies ReLU and the weights, and writes one BF16 logit. Exact up to summation order, so
// it is also the comparator used to calibrate the tolerances. CUDA cores only; slow but finishes the
// whole test in a few minutes.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace naive_mqa {

constexpr int kHeads = 64, kDim = 128, kTile = 128;

__device__ __forceinline__ float pow2_from_ue8m0(uint32_t e) { return __uint_as_float(e << 23); }

__device__ __forceinline__ void dequant8(uint32_t packed, float scale, float *out) {
  // packed: 4 E4M3 bytes
  __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(packed & 0xffff), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(packed >> 16), __NV_E4M3);
  float2 a = __half22float2(__half2(lo)), b = __half22float2(__half2(hi));
  out[0] = a.x * scale; out[1] = a.y * scale; out[2] = b.x * scale; out[3] = b.y * scale;
}

__global__ void mqa_logits_kernel(const uint8_t *__restrict__ q, const uint32_t *__restrict__ q_sf,
                                  const uint8_t *__restrict__ kv, const uint32_t *__restrict__ kv_sf,
                                  const __nv_bfloat16 *__restrict__ weights, int w_stride,
                                  const int *__restrict__ ks, const int *__restrict__ ke,
                                  __nv_bfloat16 *__restrict__ logits, int logits_stride) {
  int i = blockIdx.y;
  int start = ks[i], end = ke[i];
  int tile0 = blockIdx.x * kTile;
  if (start + tile0 >= end) return;
  __shared__ __align__(16) float sq[kHeads][kDim];
  __shared__ float sw[kHeads];
  // Query row -> FP32 with its block scales.
  for (int idx = threadIdx.x; idx < kHeads * (kDim / 4); idx += blockDim.x) {
    int h = idx / (kDim / 4), d4 = idx % (kDim / 4);
    uint32_t sf = q_sf[i * kHeads + h];
    float scale = pow2_from_ue8m0((sf >> (8 * (d4 / 8))) & 0xff);   // 32 elements = 8 groups of 4
    uint32_t packed = *reinterpret_cast<const uint32_t *>(q + ((size_t)i * kHeads + h) * kDim + d4 * 4);
    dequant8(packed, scale, &sq[h][d4 * 4]);
  }
  if (threadIdx.x < kHeads) sw[threadIdx.x] = __bfloat162float(weights[(size_t)i * w_stride + threadIdx.x]);
  __syncthreads();

  int j = tile0 + threadIdx.x;       // compressed column
  int p = start + j;                 // KV position
  if (p >= end) return;
  float k[kDim];
  uint32_t ksf = kv_sf[p];
  const uint4 *krow = reinterpret_cast<const uint4 *>(kv + (size_t)p * kDim);
#pragma unroll
  for (int c = 0; c < kDim / 16; ++c) {
    uint4 v = krow[c];
    float scale = pow2_from_ue8m0((ksf >> (8 * (c / 2))) & 0xff);   // 16 elements per uint4
    dequant8(v.x, scale, &k[c * 16 + 0]);
    dequant8(v.y, scale, &k[c * 16 + 4]);
    dequant8(v.z, scale, &k[c * 16 + 8]);
    dequant8(v.w, scale, &k[c * 16 + 12]);
  }
  float acc = 0.f;
#pragma unroll 2
  for (int h = 0; h < kHeads; ++h) {
    float dot = 0.f;
    const float4 *qh = reinterpret_cast<const float4 *>(sq[h]);
#pragma unroll
    for (int d4 = 0; d4 < kDim / 4; ++d4) {
      float4 qq = qh[d4];
      dot += qq.x * k[d4 * 4] + qq.y * k[d4 * 4 + 1] + qq.z * k[d4 * 4 + 2] + qq.w * k[d4 * 4 + 3];
    }
    acc += fmaxf(dot, 0.f) * sw[h];
  }
  logits[(size_t)i * logits_stride + j] = __float2bfloat16_rn(acc);
}

}  // namespace naive_mqa

cudaError_t solution_mqa_logits(const __nv_fp8_e4m3 *q, const uint32_t *q_sf, const __nv_fp8_e4m3 *kv,
                                const uint32_t *kv_sf, const __nv_bfloat16 *weights, int weights_stride,
                                const int *cu_seq_len_k_start, const int *cu_seq_len_k_end,
                                __nv_bfloat16 *logits, int logits_stride, int seq_len, int seq_len_kv,
                                int num_heads, int head_dim, cudaStream_t stream) {
  if (num_heads != naive_mqa::kHeads || head_dim != naive_mqa::kDim) return cudaErrorInvalidValue;
  dim3 grid((seq_len_kv + naive_mqa::kTile - 1) / naive_mqa::kTile, seq_len);
  naive_mqa::mqa_logits_kernel<<<grid, naive_mqa::kTile, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(q), q_sf, reinterpret_cast<const uint8_t *>(kv), kv_sf, weights,
      weights_stride, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_stride);
  return cudaGetLastError();
}
