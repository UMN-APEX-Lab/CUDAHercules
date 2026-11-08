#pragma once
// Reference-free "naive but not slow" GQA decode kernel for the harness: one CTA of 8 warps per
// (batch, KV head); every warp walks a strided subset of the cached keys with an FP32 online
// softmax for all query heads of the group, then the warps are merged through shared memory. FP32
// math throughout (exact up to summation order), so it is also the comparator used to calibrate
// the tolerances. Far from a good kernel (CUDA cores, no split-KV), but it finishes the whole test
// in a few minutes.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <stdint.h>

namespace naive_gqa {

constexpr int kWarps = 8;
constexpr int kMaxGroup = 8;   // query heads per KV head handled by this naive kernel

__device__ __forceinline__ float2 ld2(__nv_bfloat16 const *p) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(p));
}

// D = 32 * DPL: every lane owns DPL consecutive head-dim elements.
template <int DPL>
__global__ void gqa_decode_kernel(int kvH, int qH, int L, int window, float scale,
                                  __nv_bfloat16 const *Q, __nv_bfloat16 const *K,
                                  __nv_bfloat16 const *V, int const *seq_lens,
                                  float const *sinks, __nv_bfloat16 *O) {
  constexpr int D = 32 * DPL;
  int b = blockIdx.x / kvH, hk = blockIdx.x % kvH;
  int group = qH / kvH;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int len = min(seq_lens[b], L);
  int kstart = window > 0 ? max(0, len - window) : 0;

  float qv[kMaxGroup][DPL], acc[kMaxGroup][DPL], m[kMaxGroup], l[kMaxGroup];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
    m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[g][i] = 0.f;
    if (g < group) {
      __nv_bfloat16 const *qp = Q + ((long long)b * qH + hk * group + g) * D + lane * DPL;
#pragma unroll
      for (int i = 0; i < DPL; i += 2) { float2 f = ld2(qp + i); qv[g][i] = f.x; qv[g][i + 1] = f.y; }
    }
  }
  for (int j = kstart + warp; j < len; j += kWarps) {
    long long off = (((long long)b * L + j) * kvH + hk) * D + lane * DPL;
    float kv[DPL], vv[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i += 2) {
      float2 f = ld2(K + off + i); kv[i] = f.x; kv[i + 1] = f.y;
      float2 h = ld2(V + off + i); vv[i] = h.x; vv[i + 1] = h.y;
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
      if (g < group) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) s += qv[g][i] * kv[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= scale;
        float mn = fmaxf(m[g], s);
        float alpha = __expf(m[g] - mn), p = __expf(s - mn);
        l[g] = l[g] * alpha + p;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[g][i] = acc[g][i] * alpha + p * vv[i];
        m[g] = mn;
      }
    }
  }
  // Merge the 8 warps (and the attention sink) through shared memory.
  __shared__ float sm_m[kWarps][kMaxGroup], sm_l[kWarps][kMaxGroup];
  __shared__ float sm_acc[kWarps][kMaxGroup][D];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
    if (g < group) {
      if (lane == 0) { sm_m[warp][g] = m[g]; sm_l[warp][g] = l[g]; }
#pragma unroll
      for (int i = 0; i < DPL; ++i) sm_acc[warp][g][lane * DPL + i] = acc[g][i];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < group * D; idx += blockDim.x) {
    int g = idx / D, d = idx % D;
    float M = sinks ? sinks[hk * group + g] : -INFINITY;
    for (int w = 0; w < kWarps; ++w) M = fmaxf(M, sm_m[w][g]);
    float num = 0.f, den = sinks ? __expf(sinks[hk * group + g] - M) : 0.f;
    for (int w = 0; w < kWarps; ++w) {
      float a = sm_m[w][g] == -INFINITY ? 0.f : __expf(sm_m[w][g] - M);
      num += a * sm_acc[w][g][d];
      den += a * sm_l[w][g];
    }
    O[((long long)b * qH + hk * group + g) * D + d] = __float2bfloat16_rn(num / den);
  }
}

}  // namespace naive_gqa

// Q, O: [batch, num_q_heads, head_dim]; K, V: [batch, max_seq_len, num_kv_heads, head_dim];
// seq_lens: [batch]; sinks: [num_q_heads] or nullptr. Query head h uses KV head h / (Hq / Hkv).
cudaError_t BlackwellGQA(int batch, int num_kv_heads, int num_q_heads, int max_seq_len,
                         int head_dim, int sliding_window, __nv_bfloat16 const *Q,
                         __nv_bfloat16 const *K, __nv_bfloat16 const *V, int const *seq_lens,
                         float const *sinks, __nv_bfloat16 *O) {
  if (num_q_heads / num_kv_heads > naive_gqa::kMaxGroup) return cudaErrorInvalidValue;
  float scale = 1.0f / sqrtf((float)head_dim);
  dim3 grid(batch * num_kv_heads), block(32 * naive_gqa::kWarps);
  if (head_dim == 64)
    naive_gqa::gqa_decode_kernel<2><<<grid, block>>>(num_kv_heads, num_q_heads, max_seq_len, sliding_window, scale, Q, K, V, seq_lens, sinks, O);
  else if (head_dim == 128)
    naive_gqa::gqa_decode_kernel<4><<<grid, block>>>(num_kv_heads, num_q_heads, max_seq_len, sliding_window, scale, Q, K, V, seq_lens, sinks, O);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
