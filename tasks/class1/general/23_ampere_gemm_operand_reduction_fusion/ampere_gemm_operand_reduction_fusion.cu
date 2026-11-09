/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Based on CUTLASS example 23 (GEMM with an operand reduction). The task computes the backward
data-gradient GEMM of a linear layer together with the layer's bias gradient:
  D[M=tokens, N=in_features] = alpha * A[M, K=out_features] * B[K, N] + beta * C
  row_reduction[k] = sum_m A[m, k]
The reduction runs over the GEMM's M dimension (not K), so it cannot use CUTLASS's fused
K-reduction; the reference is a CUTLASS BF16 tensor-core GEMM (per-size configuration from an
offline sweep on the benchmark GPU, RTX PRO 6000 Blackwell sm_120) followed by a
bandwidth-bound column-sum kernel over A.
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_bf16.h>

#include <cstdio>   // std::fprintf (KH_TRIAL_TIMES)
#include <cstdlib>  // std::getenv / std::atoi (KH_WARMUP / KH_ITERS overrides)

/// Optional environment overrides for the timing loops (reference and solution):
///   KH_WARMUP = warmup launches before timing, KH_ITERS = number of timed trials.
/// Unset (or empty) -> the built-in defaults are used unchanged.
static int kh_env_int(const char *name, int dflt) {
  const char *v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : dflt;
}

/// KH_TRIAL_TIMES=1: additionally log every timed trial to stderr as
/// "KH_TRIAL <label> <trial> <ms>" (stdout output is unchanged).
static void kh_trial_log(const char *label, int trial, float ms) {
  if (kh_env_int("KH_TRIAL_TIMES", 0)) std::fprintf(stderr, "KH_TRIAL %s %d %.6f\n", label, trial, ms);
}

// ===================================== KH_TIMING_BEGIN ======================================
// Timing protocol of the Class 1 harnesses (this block is identical in every harness).
//
// kh_bench(label, warmup, iters, prep, call, check) measures one implementation; the reference
// and the solution are measured the same way:
//  * prep(s) runs before every warmup and timed call, outside the timed region. It copies input
//    set s into the fixed input buffers (the call always sees the same pointers) and re-poisons
//    the outputs. s alternates 0, 1, 0, 1, ... over all calls and the two sets hold different
//    data, so a call can never reuse the result of an earlier call.
//  * Before every timed call the L2 is flushed by READING a max(256 MB, 2 x L2) buffer (L2-cached
//    loads, no stores), then cudaDeviceSynchronize(). This also writes back the dirty lines left by
//    prep, and the L2 then holds only clean, unrelated lines, so the timed call pays no write-back
//    of flush data (a write-based flush would leave up to a full L2 of dirty lines behind).
//  * Every trial is timed with CUDA events on the legacy default stream and with the host clock,
//    from just before the start event to after the cudaDeviceSynchronize() that follows the stop
//    event. Trial time = max(event time, wall time - fixed overhead), so work the call leaves
//    running on other streams (private non-blocking streams, graphs, host callbacks) when it
//    returns is counted in full. The fixed overhead (clock reads, event records, synchronization
//    wake-up) is calibrated once per process as the 90th percentile of (wall - event) over 64
//    empty-kernel calls measured the same way, clamped to [2, 50] us.
//  * check(s) runs after every timed call, outside the timed region: a device-side comparison of
//    the outputs with the golden result of input set s. Any failure (or CUDA error) prints
//    "Incorrect" and exits with status 1.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void kh_empty_kernel() {}

__global__ void kh_l2_flush_kernel(int4 const *p, size_t n, int *sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    int4 const v = __ldcg(p + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x13579bdf) *sink = 1;  // keeps the loads alive
}

/// Aborts the run as incorrect.
[[noreturn]] static void kh_fail(char const *what) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s\n", what);
  std::cout << "Incorrect" << std::endl;
  std::exit(1);
}

[[maybe_unused]] static void kh_cuda_ok(cudaError_t e, char const *what) {
  if (e != cudaSuccess) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "%s: CUDA error: %s", what, cudaGetErrorString(e));
    kh_fail(msg);
  }
}

/// Asynchronous device-to-device copy on the legacy default stream (input-set restore).
[[maybe_unused]] static void kh_copy(void *dst, void const *src, size_t bytes) {
  kh_cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice), "input restore");
}

static double kh_host_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct KhTimer {
  cudaEvent_t start = nullptr, stop = nullptr;
  int4 *flush = nullptr;
  size_t flush_n = 0;
  int *sink = nullptr;
  double overhead_ms = 0.0;
};

static void kh_l2_flush(KhTimer &t) {
  kh_l2_flush_kernel<<<1024, 256>>>(t.flush, t.flush_n, t.sink);
  kh_cuda_ok(cudaDeviceSynchronize(), "L2 flush");
}

/// One measured call: L2 flush, then event and host-clock timing up to device-wide completion.
template <typename Call>
static void kh_timed_call(KhTimer &t, Call &&call, char const *label, float *event_ms,
                          double *wall_ms) {
  kh_l2_flush(t);
  double const t0 = kh_host_ms();
  cudaEventRecord(t.start);
  call();
  cudaEventRecord(t.stop);
  cudaError_t const e1 = cudaEventSynchronize(t.stop);
  cudaError_t const e2 = cudaDeviceSynchronize();
  double const t1 = kh_host_ms();
  kh_cuda_ok(e1 != cudaSuccess ? e1 : e2, label);
  kh_cuda_ok(cudaEventElapsedTime(event_ms, t.start, t.stop), label);
  *wall_ms = t1 - t0;
}

static KhTimer &kh_timer() {
  static KhTimer t;
  static bool ready = false;
  if (ready) return t;
  int dev = 0, l2 = 0;
  kh_cuda_ok(cudaGetDevice(&dev), "timer init");
  kh_cuda_ok(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev), "timer init");
  size_t const bytes = std::max<size_t>(size_t(256) << 20, 2 * size_t(l2));
  kh_cuda_ok(cudaMalloc(&t.flush, bytes), "L2 flush buffer");
  kh_cuda_ok(cudaMemset(t.flush, 0x5a, bytes), "L2 flush buffer");
  kh_cuda_ok(cudaMalloc(&t.sink, sizeof(int)), "L2 flush buffer");
  t.flush_n = bytes / sizeof(int4);
  kh_cuda_ok(cudaEventCreate(&t.start), "timer init");
  kh_cuda_ok(cudaEventCreate(&t.stop), "timer init");
  std::vector<double> d;
  for (int i = 0; i < 74; ++i) {
    float ev = 0.f;
    double wall = 0.0;
    kh_timed_call(t, [] { kh_empty_kernel<<<1, 32>>>(); }, "timer calibration", &ev, &wall);
    if (i >= 10) d.push_back(wall - double(ev));
  }
  std::sort(d.begin(), d.end());
  t.overhead_ms = std::min(0.05, std::max(0.002, d[d.size() * 9 / 10]));
  std::fprintf(stderr, "KH timing: L2 flush %zu MB, fixed overhead %.4f ms (p90 of %zu empty-kernel "
               "trials, median %.4f ms)\n", bytes >> 20, t.overhead_ms, d.size(), d[d.size() / 2]);
  ready = true;
  return t;
}

struct KhTiming {
  float avg_ms, min_ms;
};

/// check() for implementations whose output is not verified during timing (the reference).
[[maybe_unused]] static bool kh_no_check(int) { return true; }

template <typename Prep, typename Call, typename Check>
static KhTiming kh_bench(char const *label, int warmup, int iters, Prep &&prep, Call &&call,
                         Check &&check) {
  KhTimer &t = kh_timer();
  int set = 0;
  for (int i = 0; i < warmup; ++i, set ^= 1) {
    prep(set);
    kh_cuda_ok(cudaDeviceSynchronize(), "input restore");
    call();
    kh_cuda_ok(cudaDeviceSynchronize(), label);
  }
  double total = 0.0;
  float mn = 1e30f;
  for (int it = 0; it < iters; ++it, set ^= 1) {
    prep(set);
    float ev = 0.f;
    double wall = 0.0;
    kh_timed_call(t, call, label, &ev, &wall);
    float const ms = float(std::max<double>(ev, wall - t.overhead_ms));
    kh_trial_log(label, it, ms);
    if (!check(set)) {
      char msg[256];
      std::snprintf(msg, sizeof(msg), "%s: output of timed call %d (input set %d) does not match "
                    "the golden result.", label, it, set);
      kh_fail(msg);
    }
    total += ms;
    mn = std::min(mn, ms);
  }
  return {iters > 0 ? float(total / iters) : 0.f, mn};
}
// ====================================== KH_TIMING_END =======================================

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using ElementAccumulator = float;
using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, int Swizzle = 1>
using BgemmT = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages, 8, 8,
    true /*SplitKSerial*/>;

// Per-size configurations (offline sweep):
//   0: 128x128x32 threadblock, 64x64x32 warp, 3 stages (qkv_dgrad; also small/other shapes)
//   1: 128x256x32 threadblock, 64x64x32 warp, 3 stages, 2-way swizzle (o_proj/mlp dgrad)
using Bgemm0 = BgemmT<128, 128, 32, 64, 64, 32, 3>;
using Bgemm1 = BgemmT<128, 256, 32, 64, 64, 32, 3, 2>;

/// Persistent workspace for the split-K semaphores (allocated once, outside the timed region).
static void *g_ref_workspace = nullptr;
static size_t g_ref_workspace_bytes = 0;

template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha,
                              cutlass::bfloat16_t const *A, int lda,
                              cutlass::bfloat16_t const *B, int ldb, float beta,
                              cutlass::bfloat16_t const *C, int ldc,
                              cutlass::bfloat16_t *D, int ldd, int split_k) {
  typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldd},
                                {alpha, beta}, split_k);
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_ref_workspace_bytes) {
    if (g_ref_workspace) cudaFree(g_ref_workspace);
    if (cudaMalloc(&g_ref_workspace, ws) != cudaSuccess) return cudaErrorMemoryAllocation;
    g_ref_workspace_bytes = ws;
  }
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (gemm_op.initialize(args, g_ref_workspace) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm_op() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaGetLastError();
}

/// Column sums of a column-major BF16 matrix (FP32 accumulation): out[k] = sum_m A[m + k*lda].
/// One warp per column, 16-byte vector loads when the column is 16-byte aligned.
__global__ void __launch_bounds__(256) ColumnSum_kernel(int M, int K, __nv_bfloat16 const *A,
                                                         int lda, float *out) {
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const k = blockIdx.x * 8 + warp;
  if (k >= K) return;
  __nv_bfloat16 const *col = A + size_t(k) * lda;
  float s = 0.f;
  int m = 0;
  if ((reinterpret_cast<uintptr_t>(col) & 15) == 0) {
    int const nvec = M / 8;
    uint4 const *v = reinterpret_cast<uint4 const *>(col);
    for (int i = lane; i < nvec; i += 32) {
      uint4 q = __ldg(v + i);
      __nv_bfloat162 const *h = reinterpret_cast<__nv_bfloat162 const *>(&q);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        float2 f = __bfloat1622float2(h[t]);
        s += f.x + f.y;
      }
    }
    m = nvec * 8;
  }
  for (int i = m + lane; i < M; i += 32) s += __bfloat162float(col[i]);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) out[k] = s;
}

/// Reference: D = alpha * A * B + beta * C (CUTLASS) and row_reduction[k] = sum_m A[m, k].
cudaError_t CutlassGemmWithReduction(
  int M, int N, int K,
  float alpha,
  cutlass::bfloat16_t const *A, int lda,
  cutlass::bfloat16_t const *B, int ldb,
  float beta,
  cutlass::bfloat16_t const *C, int ldc,
  cutlass::bfloat16_t *D, int ldd,
  float *row_reduction) {
  cudaError_t err;
  // qkv_dgrad (16384 x 5120 x 10240) and shapes with M < 4096 use config 0; the other 16K-token
  // shapes use config 1.
  bool const wide = M >= 4096 && !(M == 16384 && N == 5120 && K == 10240);
  if (wide)
    err = RunCutlass<Bgemm1>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, 1);
  else
    err = RunCutlass<Bgemm0>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, 1);
  if (err != cudaSuccess) return err;
  ColumnSum_kernel<<<(K + 7) / 8, 256>>>(M, K, reinterpret_cast<__nv_bfloat16 const *>(A), lda,
                                         row_reduction);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Verification kit: random full-mantissa inputs, golden GEMM on the device (FP64, or FP32 for
// 16-bit inputs whose products are exact in FP32), NaN-safe comparison of every output element
// (max error and relative Frobenius error vs the golden).
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define KH_CHECK(call)                                                                         \
  do {                                                                                         \
    cudaError_t kh_err_ = (call);                                                              \
    if (kh_err_ != cudaSuccess) {                                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(kh_err_) << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                                      \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

__host__ __device__ inline unsigned long long kh_hash64(unsigned long long i, unsigned int seed) {
  unsigned long long x = i * 0x9E3779B97F4A7C15ull + (seed + 1ull) * 0xD1B54A32D192ED03ull;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 33;
  return x;
}

/// Uniform in [-1, 1): a random 25-bit signed integer times 2^-24. Every value with |v| >= 0.5
/// carries a full 24-bit significand, so rounding inputs to TF32/BF16/FP16 changes the result.
__host__ __device__ inline float kh_uniform(unsigned long long i, unsigned int seed) {
  return float(int(unsigned(kh_hash64(i, seed) >> 32)) >> 7) * (1.0f / 16777216.0f);
}

__device__ inline double kh_to_double(float x) { return double(x); }
__device__ inline double kh_to_double(__half x) { return double(__half2float(x)); }
__device__ inline double kh_to_double(__nv_bfloat16 x) { return double(__bfloat162float(x)); }
template <typename T> __device__ inline T kh_from_float(float x);
template <> __device__ inline float kh_from_float<float>(float x) { return x; }
template <> __device__ inline __half kh_from_float<__half>(float x) { return __float2half_rn(x); }
template <> __device__ inline __nv_bfloat16 kh_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

/// Random values: uniform in [-scale, scale) rounded to T (a full T significand).
template <typename T>
__global__ void kh_fill_kernel(T *p, size_t n, unsigned int seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_from_float<T>(kh_uniform(i, seed) * scale);
}

template <typename T>
__global__ void kh_fill_value_kernel(T *p, size_t n, float v) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_from_float<T>(v);
}

template <typename T>
static void kh_fill(T *p, size_t n, unsigned int seed, float scale = 1.0f) {
  kh_fill_kernel<T><<<4096, 256>>>(p, n, seed, scale);
  KH_CHECK(cudaGetLastError());
}

/// Finite poison for outputs that the operation may also read with beta = 0 (0 * poison = 0):
/// any element the solution leaves unwritten keeps this value and fails the comparison
/// (65504 = largest finite FP16 value; 3e38 for FP32 outputs).
static const float kPoison16 = 65504.0f;
static const float kPoison32 = 3.0e38f;

template <typename T>
static void kh_fill_value(T *p, size_t n, float v) {
  kh_fill_value_kernel<T><<<4096, 256>>>(p, n, v);
  KH_CHECK(cudaGetLastError());
}

/// Golden GEMM: G(i,j) = alpha * sum_k A(i,k) B(k,j) + beta * C(i,j), G row-major M x N (FP64),
/// accumulated in Acc (double, or float for 16-bit inputs whose products are exact in FP32).
/// Element (i,k) of A is A[i*sam + k*sak]; B(k,j) = B[k*sbk + j*sbn]; C(i,j) = C[i*scm + j*scn].
template <typename Acc, typename TA, typename TB, typename TC>
__global__ void __launch_bounds__(256) kh_golden_gemm_kernel(
    int M, int N, int K, double alpha,
    TA const *A, long long sam, long long sak,
    TB const *B, long long sbk, long long sbn,
    double beta, TC const *C, long long scm, long long scn,
    double *G) {
  __shared__ Acc As[16][65];
  __shared__ Acc Bs[16][65];
  int const tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  int const m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  Acc acc[4][4];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) acc[r][c] = Acc(0);
  bool const a_m_fast = (sam == 1);
  bool const b_n_fast = (sbn == 1);
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int e = threadIdx.x; e < 16 * 64; e += 256) {
      int kk = a_m_fast ? e / 64 : e % 16;
      int mm = a_m_fast ? e % 64 : e / 16;
      int gi = m0 + mm, gk = k0 + kk;
      As[kk][mm] = (gi < M && gk < K) ? Acc(kh_to_double(A[gi * sam + gk * sak])) : Acc(0);
      int kb = b_n_fast ? e / 64 : e % 16;
      int nn = b_n_fast ? e % 64 : e / 16;
      int gj = n0 + nn, gkb = k0 + kb;
      Bs[kb][nn] = (gj < N && gkb < K) ? Acc(kh_to_double(B[gkb * sbk + gj * sbn])) : Acc(0);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      Acc a[4], b[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) a[r] = As[kk][ty + 16 * r];
#pragma unroll
      for (int c = 0; c < 4; ++c) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
    }
    __syncthreads();
  }
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      int i = m0 + ty + 16 * r, j = n0 + tx + 16 * c;
      if (i < M && j < N) {
        double v = alpha * double(acc[r][c]);
        if (beta != 0.0) v += beta * kh_to_double(C[i * scm + j * scn]);
        G[size_t(i) * N + j] = v;
      }
    }
}

template <typename Acc, typename TA, typename TB, typename TC>
static void kh_golden_gemm(int M, int N, int K, double alpha,
                           TA const *A, long long sam, long long sak,
                           TB const *B, long long sbk, long long sbn,
                           double beta, TC const *C, long long scm, long long scn, double *G) {
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  kh_golden_gemm_kernel<Acc, TA, TB, TC><<<grid, 256>>>(M, N, K, alpha, A, sam, sak, B, sbk, sbn,
                                                       beta, C, scm, scn, G);
  KH_CHECK(cudaGetLastError());
}

struct KhStats {
  double sum_g2, sum_d2;
  unsigned long long max_d_bits, nonfinite, nbad;
};

__device__ inline void kh_block_add(double v, double *dst) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(dst, v);
}

__global__ void kh_sumsq_kernel(double const *G, size_t n, KhStats *st) {
  double s = 0.0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    s += G[i] * G[i];
  kh_block_add(s, &st->sum_g2);
}

/// Compares X(i,j) = X[i*sxm + j*sxn] with G (row-major M x N). Counts non-finite outputs and
/// elements with !(|x - g| <= tol_abs) (NaN-safe), accumulates sum (x-g)^2 and max |x-g|.
template <typename TX>
__global__ void kh_compare_kernel(TX const *X, long long sxm, long long sxn, double const *G,
                                  int M, int N, double tol_abs, double tol_rel, KhStats *st) {
  size_t const n = size_t(M) * N;
  double s = 0.0, mx = 0.0;
  unsigned long long nf = 0, nb = 0;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < n;
       idx += size_t(gridDim.x) * blockDim.x) {
    long long i = (long long)(idx / N), j = (long long)(idx % N);
    double x = kh_to_double(X[i * sxm + j * sxn]);
    double d = fabs(x - G[idx]);
    if (!isfinite(x)) { ++nf; continue; }
    if (!(d <= tol_abs + tol_rel * fabs(G[idx]))) ++nb;
    s += d * d;
    mx = fmax(mx, d);
  }
  kh_block_add(s, &st->sum_d2);
  for (int o = 16; o > 0; o >>= 1) {
    mx = fmax(mx, __shfl_down_sync(0xffffffffu, mx, o));
    nf += __shfl_down_sync(0xffffffffu, nf, o);
    nb += __shfl_down_sync(0xffffffffu, nb, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(&st->max_d_bits, (unsigned long long)__double_as_longlong(mx));
    atomicAdd(&st->nonfinite, nf);
    atomicAdd(&st->nbad, nb);
  }
}

struct KhResult {
  double rms, max_err_rel, fro_rel;
  unsigned long long nonfinite, nbad;
  bool pass;
};

/// Full check of an M x N output against the golden: every element must be finite and within
/// tol_max * RMS(golden) (+ tol_rel * |golden|) of the golden, and the relative Frobenius error
/// must be <= tol_fro.
template <typename TX>
static KhResult kh_check(TX const *X, long long sxm, long long sxn, double const *G,
                         int M, int N, double tol_max, double tol_fro, double tol_rel = 0.0) {
  KhStats *st;
  KH_CHECK(cudaMalloc(&st, sizeof(KhStats)));
  KH_CHECK(cudaMemset(st, 0, sizeof(KhStats)));
  size_t n = size_t(M) * N;
  kh_sumsq_kernel<<<2048, 256>>>(G, n, st);
  KH_CHECK(cudaGetLastError());
  KhStats h;
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  double rms = std::sqrt(h.sum_g2 / double(n));
  kh_compare_kernel<TX><<<2048, 256>>>(X, sxm, sxn, G, M, N, tol_max * rms, tol_rel, st);
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  KH_CHECK(cudaFree(st));
  double max_d;
  std::memcpy(&max_d, &h.max_d_bits, sizeof(max_d));
  KhResult r;
  r.rms = rms;
  r.max_err_rel = rms > 0 ? max_d / rms : INFINITY;
  r.fro_rel = h.sum_g2 > 0 ? std::sqrt(h.sum_d2 / h.sum_g2) : INFINITY;
  r.nonfinite = h.nonfinite;
  r.nbad = h.nbad;
  r.pass = (rms > 0) && h.nonfinite == 0 && h.nbad == 0 && r.fro_rel <= tol_fro;
  return r;
}

static void kh_print(char const *who, KhResult const &r, double tol_max, double tol_fro) {
  std::printf("  %s: max|err|/rms = %.3e (tol %.1e), rel_fro_err = %.3e (tol %.1e), "
              "non-finite = %llu, elements over tol = %llu -> %s\n",
              who, r.max_err_rel, tol_max, r.fro_rel, tol_fro, r.nonfinite, r.nbad,
              r.pass ? "ok" : "FAIL");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Golden column sums in FP64: G[k] = sum_m A[m + k*lda].
__global__ void GoldenColumnSum_kernel(int M, int K, __nv_bfloat16 const *A, int lda, double *G) {
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const k = blockIdx.x * 8 + warp;
  if (k >= K) return;
  double s = 0.0;
  for (int m = lane; m < M; m += 32) s += double(__bfloat162float(A[m + size_t(k) * lda]));
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) G[k] = s;
}

// D (BF16 output of an FP32-accumulated product): a correct result is within half a BF16 ulp
// (<= 2^-8 relative) of the exact value plus a small FP32 accumulation error. Element bound:
// |D - exact| <= kTolRel * |exact| + kTolMax * RMS(exact) (3x the rounding bound, >= 4x the
// accumulation error); relative Frobenius error <= kTolFro (>= 3x a correct result).
// FP8/INT8 operands or BF16/FP16 accumulation exceed both bounds.
static const double kTolRel = 1.2e-2;
static const double kTolMax = 2e-4;
static const double kTolFro = 5e-3;
// row_reduction (FP32 sums of M BF16 values): FP32-accumulation tolerances scaled with sqrt(M);
// BF16/FP16 accumulation fails.
static double RedTolMax(int M) { return 1.5e-6 * std::sqrt(double(M)); }
static double RedTolFro(int M) { return 1.2e-7 * std::sqrt(double(M)); }

/// Golden(s) of the correctness inputs (A seed 1, B seed 2, C seed 3), handed over from TestCorrectness
/// to Profile, where they are the goldens of timing input set 0.
[[maybe_unused]] static double *g_golden_set0 = nullptr;
[[maybe_unused]] static double *g_golden_red_set0 = nullptr;

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against golden
/// results computed on the device. Every element of D and of row_reduction is checked.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
#ifdef KH_TEST_SOLUTION
  cudaFree(g_golden_set0);  // not taken over by Profile (e.g. iterations == 0)
  g_golden_set0 = nullptr;
  cudaFree(g_golden_red_set0);
  g_golden_red_set0 = nullptr;
#endif
  int lda = M;  // col-major
  int ldb = N;  // row-major
  int ldc = M;  // col-major
  int ldd = M;
  size_t count_A = size_t(lda) * K, count_B = size_t(K) * ldb;
  size_t count_C = size_t(ldc) * N, count_D = size_t(ldd) * N;
  double const red_tol_max = RedTolMax(M), red_tol_fro = RedTolFro(M);

  __nv_bfloat16 *A, *B, *C, *D_ref;
  float *red_ref;
  double *G, *Gr;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&C, count_C * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&D_ref, count_D * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&red_ref, K * sizeof(float)));
  KH_CHECK(cudaMalloc(&G, size_t(M) * N * sizeof(double)));
  KH_CHECK(cudaMalloc(&Gr, K * sizeof(double)));
  kh_fill(A, count_A, 1u);
  kh_fill(B, count_B, 2u);
  kh_fill(C, count_C, 3u);
  // Outputs are poisoned (0xFF bytes = NaN in BF16 and FP32): unwritten elements fail.
  KH_CHECK(cudaMemset(D_ref, 0xFF, count_D * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMemset(red_ref, 0xFF, K * sizeof(float)));

  // Golden: FP32 accumulation of exact BF16 products, FP64 column sums.
  kh_golden_gemm<float>(M, N, K, alpha, A, 1, lda, B, ldb, 1, beta, C, 1, ldc, G);
  GoldenColumnSum_kernel<<<(K + 7) / 8, 256>>>(M, K, A, lda, Gr);
  KH_CHECK(cudaGetLastError());

  cudaError_t result = CutlassGemmWithReduction(
      M, N, K, alpha, reinterpret_cast<cutlass::bfloat16_t const *>(A), lda,
      reinterpret_cast<cutlass::bfloat16_t const *>(B), ldb, beta,
      reinterpret_cast<cutlass::bfloat16_t const *>(C), ldc,
      reinterpret_cast<cutlass::bfloat16_t *>(D_ref), ldd, red_ref);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS reference failed: " << cudaGetErrorString(result) << std::endl;
  } else {
    KhResult r = kh_check(D_ref, 1, ldd, G, M, N, kTolMax, kTolFro, kTolRel);
    kh_print("reference D        ", r, kTolMax, kTolFro);
    KhResult rr = kh_check(red_ref, 1, 0, Gr, K, 1, red_tol_max, red_tol_fro);
    kh_print("reference reduction", rr, red_tol_max, red_tol_fro);
    if (!r.pass || !rr.pass) {
      std::cerr << "Reference incorrect vs golden." << std::endl;
      result = cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    __nv_bfloat16 *D_sol;
    float *red_sol;
    KH_CHECK(cudaMalloc(&D_sol, count_D * sizeof(__nv_bfloat16)));
    KH_CHECK(cudaMalloc(&red_sol, K * sizeof(float)));
    KH_CHECK(cudaMemset(D_sol, 0xFF, count_D * sizeof(__nv_bfloat16)));
    KH_CHECK(cudaMemset(red_sol, 0xFF, K * sizeof(float)));
    result = GemmWithReduction(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_sol, ldd, red_sol);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(D_sol, 1, ldd, G, M, N, kTolMax, kTolFro, kTolRel);
      kh_print("solution  D        ", r, kTolMax, kTolFro);
      KhResult rr = kh_check(red_sol, 1, 0, Gr, K, 1, red_tol_max, red_tol_fro);
      kh_print("solution  reduction", rr, red_tol_max, red_tol_fro);
      if (!r.pass) {
        std::cerr << "Solution GEMM incorrect vs golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", elements over tolerance (" << kTolRel << "*|ref| + " << kTolMax
                  << "*rms) = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
      if (!rr.pass) {
        std::cerr << "Solution row_reduction incorrect vs golden: max|err|/rms = "
                  << rr.max_err_rel << ", rel_fro_err = " << rr.fro_rel
                  << ", non-finite = " << rr.nonfinite << ", elements over tolerance = "
                  << rr.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
    cudaFree(red_sol);
    cudaFree(D_sol);
  }
#endif

#ifdef KH_TEST_SOLUTION
  // The golden of the correctness data is reused as the golden of timing input set 0 (Profile
  // regenerates exactly these inputs), instead of being recomputed.
  if (result == cudaSuccess) {
    g_golden_set0 = G;
    G = nullptr;
    g_golden_red_set0 = Gr;
    Gr = nullptr;
  }
#endif
  cudaFree(Gr);
  cudaFree(G);
  cudaFree(red_ref);
  cudaFree(D_ref);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile the reference (and the solution if KH_TEST_SOLUTION) with the KH timing protocol:
/// two different input sets alternate between calls (restored into the same buffers outside the
/// timed region), the outputs are re-poisoned before and verified (solution) after every timed
/// call.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  int const warmup = kh_env_int("KH_WARMUP", 3);
  int lda = M, ldb = N, ldc = M, ldd = M;
  size_t count_A = size_t(lda) * K, count_B = size_t(K) * ldb;
  size_t count_C = size_t(ldc) * N, count_D = size_t(ldd) * N;
  bool const has_C = (beta != 0.0f);

  // Buffers passed to the reference / solution, and two input sets (set 0 = the correctness data).
  // C only enters the result when beta != 0; otherwise it keeps the set-0 data.
  __nv_bfloat16 *A, *B, *C, *D, *A_set[2], *B_set[2], *C_set[2] = {nullptr, nullptr};
  float *red;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&C, count_C * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&D, count_D * sizeof(__nv_bfloat16)));
  KH_CHECK(cudaMalloc(&red, K * sizeof(float)));
  kh_fill(C, count_C, 3u);
  for (int s = 0; s < 2; ++s) {
    KH_CHECK(cudaMalloc(&A_set[s], count_A * sizeof(__nv_bfloat16)));
    KH_CHECK(cudaMalloc(&B_set[s], count_B * sizeof(__nv_bfloat16)));
    kh_fill(A_set[s], count_A, 1u + 10u * s);
    kh_fill(B_set[s], count_B, 2u + 10u * s);
    if (has_C) {
      KH_CHECK(cudaMalloc(&C_set[s], count_C * sizeof(__nv_bfloat16)));
      kh_fill(C_set[s], count_C, 3u + 10u * s);
    }
  }
  // Outputs are poisoned (0xFF bytes = NaN in BF16 and FP32): unwritten elements fail.
  auto prep = [&](int s) {
    kh_copy(A, A_set[s], count_A * sizeof(__nv_bfloat16));
    kh_copy(B, B_set[s], count_B * sizeof(__nv_bfloat16));
    if (has_C) kh_copy(C, C_set[s], count_C * sizeof(__nv_bfloat16));
    kh_cuda_ok(cudaMemsetAsync(D, 0xFF, count_D * sizeof(__nv_bfloat16)), "output poison");
    kh_cuda_ok(cudaMemsetAsync(red, 0xFF, K * sizeof(float)), "output poison");
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, prep, [&] {
    CutlassGemmWithReduction(M, N, K, alpha, reinterpret_cast<cutlass::bfloat16_t const *>(A), lda,
                             reinterpret_cast<cutlass::bfloat16_t const *>(B), ldb, beta,
                             reinterpret_cast<cutlass::bfloat16_t const *>(C), ldc,
                             reinterpret_cast<cutlass::bfloat16_t *>(D), ldd, red);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  // Goldens of both input sets (computed once, outside the timed region): FP32 accumulation of
  // exact BF16 products, FP64 column sums.
  double const red_tol_max = RedTolMax(M), red_tol_fro = RedTolFro(M);
  double *G[2], *Gr[2];
  for (int s = 0; s < 2; ++s) {
    if (s == 0 && g_golden_set0 && g_golden_red_set0) {  // set 0 = the correctness inputs: golden already computed
      G[0] = g_golden_set0;
      g_golden_set0 = nullptr;
      Gr[0] = g_golden_red_set0;
      g_golden_red_set0 = nullptr;
      continue;
    }
    KH_CHECK(cudaMalloc(&G[s], size_t(M) * N * sizeof(double)));
    KH_CHECK(cudaMalloc(&Gr[s], K * sizeof(double)));
    kh_golden_gemm<float>(M, N, K, alpha, A_set[s], 1, lda, B_set[s], ldb, 1, beta,
                          has_C ? C_set[s] : C, 1, ldc, G[s]);
    GoldenColumnSum_kernel<<<(K + 7) / 8, 256>>>(M, K, A_set[s], lda, Gr[s]);
    KH_CHECK(cudaGetLastError());
  }
  KH_CHECK(cudaDeviceSynchronize());
  auto check = [&](int s) {
    KhResult r = kh_check(D, 1, ldd, G[s], M, N, kTolMax, kTolFro, kTolRel);
    KhResult rr = kh_check(red, 1, 0, Gr[s], K, 1, red_tol_max, red_tol_fro);
    if (!r.pass) kh_print("solution  D (timed call)        ", r, kTolMax, kTolFro);
    if (!rr.pass) kh_print("solution  reduction (timed call)", rr, red_tol_max, red_tol_fro);
    return r.pass && rr.pass;
  };

  KhTiming sol = kh_bench("Kernel time", warmup, iterations, prep, [&] {
    GemmWithReduction(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, red);
  }, check);
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  for (int s = 0; s < 2; ++s) {
    cudaFree(G[s]);
    cudaFree(Gr[s]);
  }
#endif

  for (int s = 0; s < 2; ++s) {
    cudaFree(A_set[s]);
    cudaFree(B_set[s]);
    cudaFree(C_set[s]);
  }
  cudaFree(red);
  cudaFree(D);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 128, N = 128, K = 128;
    std::stringstream(arg[1]) >> M;
    std::stringstream(arg[2]) >> N;
    std::stringstream(arg[3]) >> K;
    if (argc > 4) std::stringstream(arg[4]) >> alpha;
    if (argc > 5) std::stringstream(arg[5]) >> beta;
    if (argc > 6) std::stringstream(arg[6]) >> iterations;
    configs.push_back({"custom", M, N, K});
  } else {
    // Backward data-gradient GEMM of the Qwen3-32B linear layers with the fused bias gradient,
    // one 16K-token chunk (hidden 5120, fused QKV 10240, attention out 8192, FFN 25600):
    //   dX[M=tokens, N=in_features] = dY[M, K=out_features] * W[K, N],
    //   row_reduction[k] = sum_m dY[m, k]  (= bias gradient of the layer)
    configs = {
      {"qkv_dgrad",      16384,  5120, 10240},   // fused QKV proj (5120 -> 10240)
      {"o_proj_dgrad",   16384,  8192,  5120},   // attention output proj (8192 -> 5120)
      {"mlp_up_dgrad",   16384,  5120, 51200},   // fused gate+up proj (5120 -> 51200)
      {"mlp_down_dgrad", 16384, 25600,  5120},   // down proj (25600 -> 5120)
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
