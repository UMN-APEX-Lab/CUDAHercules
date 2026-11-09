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

// Standard Library includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

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

// Helper methods to check for errors
#include "helper.h"

// CUTLASS GEMM
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS quaternion SIMT GEMM (cutlass::Quaternion<float>, Hamilton product, FP32
// accumulation) — A row-major, B column-major, C row-major, as in CUTLASS example 21 — with a
// per-size configuration chosen by an offline sweep on the benchmark GPU (sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using Element = cutlass::Quaternion<float>;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Element, 1, Element, Element>;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, typename Arch>
using QuatGemmT = cutlass::gemm::device::Gemm<
    Element, LayoutInputA, Element, LayoutInputB, Element, LayoutOutput, Element,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<1, 1, 1>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, 1, 1,
    true /*SplitKSerial*/>;

// Per-size configurations (offline sweep):
//   0: 64x128x8 threadblock, 32x32x8 warp, 3-stage cp.async pipeline   (16K-token prefill shapes)
//   1: 32x64x4  threadblock, 32x16x4 warp, 2-stage, serial split-K     (decode shapes)
using QuatGemm0 = QuatGemmT<64, 128, 8, 32, 32, 8, 3, cutlass::arch::Sm80>;
using QuatGemm1 = QuatGemmT<32, 64, 4, 32, 16, 4, 2, cutlass::arch::Sm50>;

struct RefPlan { int config; int split_k; };

static RefPlan RefPlanFor(int M, int N, int K) {
  if (M == 128 && N == 2560 && K == 1280) return {1, 8};   // decode_qkv
  if (M >= 2048) return {0, 1};
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  long long tiles = (long long)((M + 31) / 32) * ((N + 63) / 64);
  int split = int(std::min<long long>(8, std::max<long long>(1, (4LL * sms + tiles - 1) / tiles)));
  return {1, std::max(1, std::min(split, K / 64))};
}


/// Persistent workspace for the split-K semaphores (allocated once, outside the timed region).
static void *g_ref_workspace = nullptr;
static size_t g_ref_workspace_bytes = 0;

template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha, float const *A, int lda,
                              float const *B, int ldb, float beta, float *C, int ldc, int split_k) {
  Element const *qA = reinterpret_cast<Element const *>(A);
  Element const *qB = reinterpret_cast<Element const *>(B);
  Element *qC = reinterpret_cast<Element *>(C);
  typename Gemm::Arguments args({M, N, K}, {qA, lda}, {qB, ldb}, {qC, ldc}, {qC, ldc},
                                {Element(alpha), Element(beta)}, split_k);
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

/// CUTLASS Quaternion GEMM wrapper.
/// A: row-major (M x K), B: column-major (K x N), C: row-major (M x N).
/// Each quaternion = 4 consecutive floats in CUTLASS order: (x, y, z, w).
/// A: quaternion (i,k) at A[(i * lda + k) * 4], lda = K
/// B: quaternion (k,j) at B[(k + j * ldb) * 4], ldb = K
/// C: quaternion (i,j) at C[(i * ldc + j) * 4], ldc = N
cudaError_t CutlassQuaternionGemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {
  RefPlan p = RefPlanFor(M, N, K);
  if (p.config == 1)
    return RunCutlass<QuatGemm1>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, p.split_k);
  return RunCutlass<QuatGemm0>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 1);
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

/// Golden quaternion GEMM in FP64 (Hamilton product, (x, y, z, w) storage), written as an
/// M x 4N row-major array of doubles: G[i*4N + 4j + c] = component c of output (i, j).
__global__ void GoldenQuaternionGemm_kernel(int M, int N, int K, double alpha, float const *A,
                                            int lda, float const *B, int ldb, double beta,
                                            float const *C, int ldc, double *G) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= M || j >= N) return;
  double x = 0, y = 0, z = 0, w = 0;
  for (int k = 0; k < K; ++k) {
    float4 a = *reinterpret_cast<float4 const *>(A + (size_t(i) * lda + k) * 4);
    float4 b = *reinterpret_cast<float4 const *>(B + (size_t(j) * ldb + k) * 4);
    double ax = a.x, ay = a.y, az = a.z, aw = a.w, bx = b.x, by = b.y, bz = b.z, bw = b.w;
    x += aw * bx + bw * ax + ay * bz - az * by;
    y += aw * by + bw * ay + az * bx - ax * bz;
    z += aw * bz + bw * az + ax * by - ay * bx;
    w += aw * bw - ax * bx - ay * by - az * bz;
  }
  double r[4] = {alpha * x, alpha * y, alpha * z, alpha * w};
  if (beta != 0.0)
    for (int c = 0; c < 4; ++c) r[c] += beta * double(C[(size_t(i) * ldc + j) * 4 + c]);
  for (int c = 0; c < 4; ++c) G[size_t(i) * 4 * N + 4 * size_t(j) + c] = r[c];
}

// FP32 tolerances, relative to the RMS of the exact (FP64) result and scaled with sqrt(4K)
// (each output component accumulates 4K products) like FP32 accumulation error. At the benchmark
// sizes plain sequential FP32 accumulation stays >= 3x below the Frobenius bound (>= 5x below the
// max bound). Single-pass TF32 (also 2xTF32), BF16 or FP16 products raise the Frobenius error to
// >= 1.8e-4, > 9x above the bound; 3xBF16 (2-term split, error floor ~4e-6) exceeds the bound at
// K = 1280. FP32-accurate emulations (3xTF32, 6xBF16, 9xBF16) pass.
static double TolMax(int K) { return 1.5e-6 * std::sqrt(4.0 * K); }
static double TolFro(int K) { return 5.5e-8 * std::sqrt(4.0 * K); }

/// Golden(s) of the correctness inputs (A seed 1, B seed 2, initial C seed 3 or poison), handed over from TestCorrectness
/// to Profile, where they are the goldens of timing input set 0.
[[maybe_unused]] static double *g_golden_set0 = nullptr;

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against an FP64
/// golden result computed on the device. Every output component is checked.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
#ifdef KH_TEST_SOLUTION
  cudaFree(g_golden_set0);  // not taken over by Profile (e.g. iterations == 0)
  g_golden_set0 = nullptr;
#endif
  int lda = K, ldb = K, ldc = N;
  size_t floats_A = size_t(M) * lda * 4, floats_B = size_t(N) * ldb * 4;
  size_t floats_C = size_t(M) * ldc * 4;
  double const kTolMax = TolMax(K), kTolFro = TolFro(K);

  float *A, *B, *C_ref;
  double *G;
  KH_CHECK(cudaMalloc(&A, floats_A * sizeof(float)));
  KH_CHECK(cudaMalloc(&B, floats_B * sizeof(float)));
  KH_CHECK(cudaMalloc(&C_ref, floats_C * sizeof(float)));
  KH_CHECK(cudaMalloc(&G, size_t(M) * N * 4 * sizeof(double)));
  kh_fill(A, floats_A, 1u);
  kh_fill(B, floats_B, 2u);
  // Initial C: random when beta != 0; otherwise a finite poison (0 * poison = 0) so that
  // components the solution never writes fail.
  if (beta != 0.0f) kh_fill(C_ref, floats_C, 3u);
  else kh_fill_value(C_ref, floats_C, kPoison32);

  {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    GoldenQuaternionGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc, G);
    KH_CHECK(cudaGetLastError());
  }

  cudaError_t result = cudaSuccess;
#ifdef KH_TEST_SOLUTION
  float *C_sol;
  KH_CHECK(cudaMalloc(&C_sol, floats_C * sizeof(float)));
  KH_CHECK(cudaMemcpy(C_sol, C_ref, floats_C * sizeof(float), cudaMemcpyDeviceToDevice));
#endif

  result = CutlassQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS quaternion GEMM failed: " << cudaGetErrorString(result) << std::endl;
  } else {
    KhResult r = kh_check(C_ref, 4LL * ldc, 1, G, M, 4 * N, kTolMax, kTolFro);
    kh_print("reference", r, kTolMax, kTolFro);
    if (!r.pass) {
      std::cerr << "CUTLASS reference incorrect vs FP64 golden." << std::endl;
      result = cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    result = QuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_sol, ldc);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(C_sol, 4LL * ldc, 1, G, M, 4 * N, kTolMax, kTolFro);
      kh_print("solution ", r, kTolMax, kTolFro);
      if (!r.pass) {
        std::cerr << "Solution incorrect vs FP64 golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", components over tolerance = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
  }
  cudaFree(C_sol);
#endif

#ifdef KH_TEST_SOLUTION
  // The golden of the correctness data is reused as the golden of timing input set 0 (Profile
  // regenerates exactly these inputs), instead of being recomputed.
  if (result == cudaSuccess) {
    g_golden_set0 = G;
    G = nullptr;
  }
#endif
  cudaFree(G);
  cudaFree(C_ref);
  cudaFree(B);
  cudaFree(A);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile the reference (and the solution if KH_TEST_SOLUTION) with the KH timing protocol:
/// two different input sets alternate between calls (restored into the same buffers outside the
/// timed region), the output is re-poisoned before and verified (solution) after every timed call.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  int const warmup = kh_env_int("KH_WARMUP", 3);
  int lda = K, ldb = K, ldc = N;
  size_t floats_A = size_t(M) * lda * 4, floats_B = size_t(N) * ldb * 4;
  size_t floats_C = size_t(M) * ldc * 4;
  bool const has_C = (beta != 0.0f);

  // Buffers passed to the reference / solution, and two input sets (set 0 = the correctness data).
  float *A, *B, *C, *A_set[2], *B_set[2], *C_set[2] = {nullptr, nullptr};
  KH_CHECK(cudaMalloc(&A, floats_A * sizeof(float)));
  KH_CHECK(cudaMalloc(&B, floats_B * sizeof(float)));
  KH_CHECK(cudaMalloc(&C, floats_C * sizeof(float)));
  for (int s = 0; s < 2; ++s) {
    KH_CHECK(cudaMalloc(&A_set[s], floats_A * sizeof(float)));
    KH_CHECK(cudaMalloc(&B_set[s], floats_B * sizeof(float)));
    kh_fill(A_set[s], floats_A, 1u + 10u * s);
    kh_fill(B_set[s], floats_B, 2u + 10u * s);
    if (has_C) {
      KH_CHECK(cudaMalloc(&C_set[s], floats_C * sizeof(float)));
      kh_fill(C_set[s], floats_C, 3u + 10u * s);
    }
  }
  // Initial C: the input set when beta != 0, otherwise the finite poison (0 * poison = 0).
  auto prep = [&](int s) {
    kh_copy(A, A_set[s], floats_A * sizeof(float));
    kh_copy(B, B_set[s], floats_B * sizeof(float));
    if (has_C) kh_copy(C, C_set[s], floats_C * sizeof(float));
    else kh_fill_value(C, floats_C, kPoison32);
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, prep, [&] {
    CutlassQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  // FP64 goldens of both input sets (computed once, outside the timed region).
  double const kTolMax = TolMax(K), kTolFro = TolFro(K);
  double *G[2];
  for (int s = 0; s < 2; ++s) {
    if (s == 0 && g_golden_set0) {  // set 0 = the correctness inputs: golden already computed
      G[0] = g_golden_set0;
      g_golden_set0 = nullptr;
      continue;
    }
    KH_CHECK(cudaMalloc(&G[s], size_t(M) * N * 4 * sizeof(double)));
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    GoldenQuaternionGemm_kernel<<<grid, block>>>(M, N, K, alpha, A_set[s], lda, B_set[s], ldb, beta,
                                                 has_C ? C_set[s] : C, ldc, G[s]);
    KH_CHECK(cudaGetLastError());
  }
  KH_CHECK(cudaDeviceSynchronize());
  auto check = [&](int s) {
    KhResult r = kh_check(C, 4LL * ldc, 1, G[s], M, 4 * N, kTolMax, kTolFro);
    if (!r.pass) kh_print("solution (timed call)", r, kTolMax, kTolFro);
    return r.pass;
  };

  KhTiming sol = kh_bench("Kernel time", warmup, iterations, prep, [&] {
    QuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }, check);
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  for (int s = 0; s < 2; ++s) cudaFree(G[s]);
#endif

  for (int s = 0; s < 2; ++s) {
    cudaFree(A_set[s]);
    cudaFree(B_set[s]);
    cudaFree(C_set[s]);
  }
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
    // Quaternion linear layers of Qwen3-32B: real dims / 4 (hidden 5120 -> 1280, fused QKV
    // 10240 -> 2560, attention out 8192 -> 2048, FFN 25600 -> 6400); M = tokens
    // (16K-token prefill chunk, 128-sequence decode).
    configs = {
      {"decode_qkv",        128,  2560, 1280},   // fused QKV proj, 128 decode tokens
      {"prefill_qkv",     16384,  2560, 1280},   // fused QKV proj, 16K-token prefill chunk
      {"prefill_o_proj",  16384,  1280, 2048},   // attention output proj
      {"prefill_mlp_up",  16384, 12800, 1280},   // fused gate+up proj (2 x 6400)
      {"prefill_mlp_down",16384,  1280, 6400},   // down proj
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
