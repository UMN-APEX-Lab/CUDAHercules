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

// This example fuses gather before GEMM and scatter after GEMM into the same
// GEMM kernel.  Gather and scatter operation is controlled by an index vector
// to select rows or columns from A, B, C or D matrices.
//
// Suppose, all matrices are column major.  The pseudo code of the fused kernel
// in this example is essentially
//
//    for (int i = 0; i < problem_size.m(); ++i) {
//      for (int j = 0; j < options.index_size; ++j) {
//        int b_c_d_col = tensor_indices.at({j, 0});
//
//        for (int k = 0; k < options.index_size; ++k) {
//            tensor_d_ref.at({i, b_c_d_col}) +=
//              alpha * tensor_a.at({i, k}) * tensor_b.at({k, b_c_d_col});
//        }
//      }
//
// Note that the index vector contains unique random integers with max to be N - 1
//
// The gather/scatter operation works best when we can still keep the biggest
// alignment. For example, when the matrix is row major, we select rows. When
// the matrix is column major, we select columns.
//
// Not all the combination of gather and scatter are legal. For example, if A is
// row major and C/D is column major, we cannot gather A and scatter C/D at the
// same time.
//
// Also, we don't check the index value is legal and index array point is valid
// for the sake of the performance.


#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS fused gather-GEMM-scatter (GemmUniversal with GatherA / ScatterD), FP16
// tensor cores with FP32 accumulation and FP32 output. All matrices are row-major: gathering
// rows of A and scattering rows of D keeps the full vector alignment along K / N. Per-size
// configuration chosen by an offline sweep on the benchmark GPU (RTX PRO 6000 Blackwell, sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue>;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, int Swizzle = 1>
using GatherScatterGemmT = cutlass::gemm::device::GemmUniversal<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages,
    8, 8,                                  // alignment A, B
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,
    true,  /*GatherA*/
    false, /*GatherB*/
    true   /*ScatterD*/>;

// Per-size configurations (offline sweep):
//   0: 128x128x32 threadblock, 64x64x32 warp, 5 stages               (qwen3_30b_a3b)
//   1: 128x128x32 threadblock, 64x64x32 warp, 3 stages, serial split-K (other shapes; FP32 output,
//      so the split-K partial sums stay in FP32)
using GatherScatterGemm0 = GatherScatterGemmT<128, 128, 32, 64, 64, 32, 5>;
using GatherScatterGemm1 = GatherScatterGemmT<128, 128, 32, 64, 64, 32, 3>;

struct RefPlan { int config; int split_k; };

static RefPlan RefPlanFor(int M, int N, int K) {
  if (M == 1024 && N == 1536 && K == 2048)   return {0, 1};   // qwen3_30b_a3b
  if (M == 1536 && N == 2816 && K == 2048)   return {1, 2};   // deepseek_v2_lite
  if (M == 4096 && N == 14336 && K == 4096)  return {1, 1};   // mixtral_8x7b_tp2
  if (M == 512 && N == 4096 && K == 7168)    return {1, 4};   // deepseek_v3_ep
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  long long tiles = (long long)((M + 127) / 128) * ((N + 127) / 128);
  int split = int(std::min<long long>(4, std::max<long long>(1, (sms + tiles - 1) / tiles)));
  return {1, std::max(1, std::min(split, K / 1024))};
}


/// Persistent workspace (allocated once, outside the timed region).
static void *g_ref_workspace = nullptr;
static size_t g_ref_workspace_bytes = 0;

template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha, __half const *dA, int lda,
                              int const *d_gather_A, __half const *dB, int ldb, float beta,
                              float const *dC, int ldc, float *dD, int ldd,
                              int const *d_scatter_D, int split_k) {
  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord(M, N, K),
      split_k,                             // serial split-K slices
      {alpha, beta},
      reinterpret_cast<cutlass::half_t const *>(dA),
      reinterpret_cast<cutlass::half_t const *>(dB), dC, dD,
      0, 0, 0, 0,                          // batch strides (unused)
      int64_t(lda), int64_t(ldb), int64_t(ldc), int64_t(ldd),
      d_gather_A,                          // gather rows of A
      nullptr,                             // no gather on B
      d_scatter_D);                        // scatter rows of D (and read rows of C)
  size_t ws = Gemm::get_workspace_size(arguments);
  if (ws > g_ref_workspace_bytes) {
    if (g_ref_workspace) cudaFree(g_ref_workspace);
    if (cudaMalloc(&g_ref_workspace, ws) != cudaSuccess) return cudaErrorMemoryAllocation;
    g_ref_workspace_bytes = ws;
  }
  Gemm gemm_op;
  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (gemm_op.initialize(arguments, g_ref_workspace) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm_op() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaGetLastError();
}

/// CUTLASS fused gather-GEMM-scatter (reference that is timed).
/// D[scatter_D[i], j] = alpha * sum_k A[gather_A[i], k] * B[k, j] + beta * C[scatter_D[i], j]
/// CUTLASS reads C through the scatter indices, so it matches the task semantics when
/// gather_C == scatter_D (which is how the harness builds the indices).
cudaError_t CutlassGatherScatterGemm(
    int M, int N, int K, float alpha,
    __half const *dA, int lda, int const *d_gather_A,
    __half const *dB, int ldb,
    float beta,
    float const *dC, int ldc,
    float *dD, int ldd, int const *d_scatter_D) {
  RefPlan p = RefPlanFor(M, N, K);
  if (p.config == 0)
    return RunCutlass<GatherScatterGemm0>(M, N, K, alpha, dA, lda, d_gather_A, dB, ldb, beta, dC,
                                          ldc, dD, ldd, d_scatter_D, 1);
  return RunCutlass<GatherScatterGemm1>(M, N, K, alpha, dA, lda, d_gather_A, dB, ldb, beta, dC,
                                        ldc, dD, ldd, d_scatter_D, p.split_k);
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
/// MoE token dispatch for one expert (vLLM fused-MoE style). A_rows tokens each pick top_k
/// experts; the expert under test receives M distinct tokens (in token order). Row i of the
/// GEMM gathers token gather_A[i] and is scattered to its (token, k) slot
/// scatter_D[i] = token * top_k + r in a [A_rows * top_k, N] output buffer; C is read at the
/// same slot (gather_C == scatter_D). Returns the number of rows of C/D. `set` selects an
/// independent routing (input set 0 = the correctness data, 1 = the alternate timing set).
static int BuildMoeIndices(int M, int A_rows, int top_k, std::vector<int> &gather_A,
                           std::vector<int> &gather_C, std::vector<int> &scatter_D,
                           unsigned set = 0) {
  std::mt19937 rng(2024u + unsigned(M) * 31u + unsigned(A_rows) + set * 0x9E3779B9u);
  std::vector<int> tokens(A_rows);
  std::iota(tokens.begin(), tokens.end(), 0);
  for (int i = 0; i < M; ++i) {            // partial Fisher-Yates: M distinct tokens
    int j = i + int(rng() % unsigned(A_rows - i));
    std::swap(tokens[i], tokens[j]);
  }
  std::sort(tokens.begin(), tokens.begin() + M);
  gather_A.resize(M); gather_C.resize(M); scatter_D.resize(M);
  for (int i = 0; i < M; ++i) {
    int slot = tokens[i] * top_k + int(rng() % unsigned(top_k));
    gather_A[i] = tokens[i];
    scatter_D[i] = slot;
    gather_C[i] = slot;
  }
  return A_rows * top_k;
}

/// Device buffers for one problem. Input set `set` (0 = correctness data, 1 = alternate timing
/// set) selects the random data and the routing (gather / scatter indices).
struct Problem {
  int M, N, K, A_rows, D_rows, lda, ldb, ldc, ldd;
  __half *dA = nullptr, *dB = nullptr;
  float *dC = nullptr;
  int *dGA = nullptr, *dGC = nullptr, *dSD = nullptr;
  size_t count_A, count_B, count_C, count_D;
  std::vector<int> gather_A, gather_C, scatter_D;

  Problem(int M_, int N_, int K_, int A_rows_, int top_k, unsigned set = 0)
      : M(M_), N(N_), K(K_), A_rows(A_rows_) {
    D_rows = BuildMoeIndices(M, A_rows, top_k, gather_A, gather_C, scatter_D, set);
    lda = K; ldb = N; ldc = N; ldd = N;
    count_A = size_t(A_rows) * K; count_B = size_t(K) * N;
    count_C = size_t(D_rows) * N; count_D = size_t(D_rows) * N;
    CUDA_CHECK(cudaMalloc(&dA, count_A * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, count_B * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, count_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGA, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dGC, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSD, M * sizeof(int)));
    // Random inputs with a full FP16 (A, B) / FP32 (C) significand.
    kh_fill(dA, count_A, 1u + 10u * set);
    kh_fill(dB, count_B, 2u + 10u * set);
    kh_fill(dC, count_C, 3u + 10u * set);
    CUDA_CHECK(cudaMemcpy(dGA, gather_A.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dGC, gather_C.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSD, scatter_D.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  }
  ~Problem() {
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
  }
  float *AllocD() const {
    float *d;
    CUDA_CHECK(cudaMalloc(&d, count_D * sizeof(float)));
    // Same finite sentinel (0x7F7F7F7F ~ 3.4e38) in every D buffer: rows that must be written
    // and are not, and rows that must NOT be written but are, both fail the comparison.
    CUDA_CHECK(cudaMemset(d, 0x7F, count_D * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());  // complete before a call that may use other streams
    return d;
  }
};

/// Gathers rows: dst[i, :] = src[idx[i], :] (row-major, ld = cols).
template <typename T>
__global__ void GatherRows_kernel(T const *src, int const *idx, int rows, int cols, T *dst) {
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(rows) * cols;
       e += size_t(gridDim.x) * blockDim.x) {
    size_t i = e / cols, j = e % cols;
    dst[e] = src[size_t(idx[i]) * cols + j];
  }
}

/// Counts elements of D rows that are NOT scatter targets whose bits differ from the sentinel.
__global__ void UntouchedRows_kernel(float const *D, unsigned char const *is_target, int D_rows,
                                     int N, unsigned long long *bad) {
  unsigned long long nb = 0;
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(D_rows) * N;
       e += size_t(gridDim.x) * blockDim.x) {
    if (!is_target[e / N] && __float_as_uint(D[e]) != 0x7F7F7F7Fu) ++nb;
  }
  for (int o = 16; o > 0; o >>= 1) nb += __shfl_down_sync(0xffffffffu, nb, o);
  if ((threadIdx.x & 31) == 0 && nb) atomicAdd(bad, nb);
}

/// Golden output rows (FP64): G[i, :] = alpha * A[gather_A[i], :] * B + beta * C[gather_C[i], :].
struct Golden {
  double *G = nullptr;
  float *Ds = nullptr;              // dense M x N staging buffer for gathered output rows
  unsigned char *is_target = nullptr;
  unsigned long long *bad = nullptr;
  Golden(Problem const &p, float alpha, float beta) {
    __half *Ag;
    float *Cg;
    CUDA_CHECK(cudaMalloc(&Ag, size_t(p.M) * p.K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Cg, size_t(p.M) * p.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&G, size_t(p.M) * p.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Ds, size_t(p.M) * p.N * sizeof(float)));
    GatherRows_kernel<<<4096, 256>>>(p.dA, p.dGA, p.M, p.K, Ag);
    GatherRows_kernel<<<4096, 256>>>(p.dC, p.dGC, p.M, p.N, Cg);
    kh_golden_gemm<double>(p.M, p.N, p.K, alpha, Ag, p.K, 1, p.dB, p.ldb, 1, beta, Cg, p.N, 1, G);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(Ag);
    cudaFree(Cg);
    std::vector<unsigned char> mask(p.D_rows, 0);
    for (int r : p.scatter_D) mask[r] = 1;
    CUDA_CHECK(cudaMalloc(&is_target, p.D_rows));
    CUDA_CHECK(cudaMemcpy(is_target, mask.data(), p.D_rows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&bad, sizeof(unsigned long long)));
  }
  ~Golden() { cudaFree(G); cudaFree(Ds); cudaFree(is_target); cudaFree(bad); }

  /// Checks every element of D: the M scattered rows against the golden, and every other row
  /// for the untouched sentinel. verbose = false prints only on failure.
  bool Check(Problem const &p, float const *D, char const *who, double tol_max, double tol_fro,
             bool verbose = true) {
    GatherRows_kernel<<<4096, 256>>>(D, p.dSD, p.M, p.N, Ds);
    KhResult r = kh_check(Ds, p.N, 1, G, p.M, p.N, tol_max, tol_fro);
    CUDA_CHECK(cudaMemset(bad, 0, sizeof(unsigned long long)));
    UntouchedRows_kernel<<<4096, 256>>>(D, is_target, p.D_rows, p.N, bad);
    unsigned long long nbad = 0;
    CUDA_CHECK(cudaMemcpy(&nbad, bad, sizeof(nbad), cudaMemcpyDeviceToHost));
    if (verbose || !r.pass || nbad) {
      kh_print(who, r, tol_max, tol_fro);
      std::printf("  %s: elements written outside the scatter_D rows = %llu\n", who, nbad);
    }
    if (!r.pass || nbad) {
      std::cerr << who << " incorrect vs FP64 golden: max|err|/rms = " << r.max_err_rel
                << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                << ", elements over tolerance = " << r.nbad
                << ", elements written outside scatter_D rows = " << nbad << std::endl;
      return false;
    }
    return true;
  }
};

// FP32-accumulated FP16 products, FP32 output: tolerances relative to the RMS of the exact
// (FP64) result, scaled with sqrt(K) like FP32 accumulation error (tensor-core FP32
// accumulation of the reference stays >= 3.7x below both). Lower-precision paths (FP16
// accumulation or output rounding, FP8/INT8 operands) exceed the Frobenius bound by >= 10x.
static double TolMax(int K) { return 2e-6 * std::sqrt(double(K)); }
static double TolFro(int K) { return 3e-7 * std::sqrt(double(K)); }

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness
cudaError_t TestCorrectness(int M, int N, int K, int A_rows, int top_k, float alpha, float beta) {
  Problem p(M, N, K, A_rows, top_k);
  Golden g(p, alpha, beta);
  double const tol_max = TolMax(K), tol_fro = TolFro(K);

  float *dD_ref = p.AllocD();
  cudaError_t result = CutlassGatherScatterGemm(M, N, K, alpha, p.dA, p.lda, p.dGA, p.dB, p.ldb,
    beta, p.dC, p.ldc, dD_ref, p.ldd, p.dSD);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS gather-scatter GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(dD_ref);
    return result;
  }
  bool ok = g.Check(p, dD_ref, "reference", tol_max, tol_fro);
  cudaFree(dD_ref);
  if (!ok) return cudaErrorUnknown;

#ifdef KH_TEST_SOLUTION
  float *dD_solution = p.AllocD();
  result = GatherScatterGemm(M, N, K, alpha, p.dA, p.lda, p.dGA, p.dB, p.ldb,
    beta, p.dC, p.ldc, p.dGC, dD_solution, p.ldd, p.dSD);
  cudaError_t sync = cudaDeviceSynchronize();
  if (result == cudaSuccess) result = sync;
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(dD_solution);
    return result;
  }
  ok = g.Check(p, dD_solution, "solution ", tol_max, tol_fro);
  cudaFree(dD_solution);
  if (!ok) return cudaErrorUnknown;
#endif

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile the reference (and the solution if KH_TEST_SOLUTION) with the KH timing protocol:
/// two input sets (different data AND different routing indices) alternate between calls,
/// restored into the same buffers outside the timed region; D is re-filled with the sentinel
/// before and verified (solution) after every timed call.
void Profile(int M, int N, int K, int A_rows, int top_k, float alpha, float beta, int iterations) {
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  int const warmup = kh_env_int("KH_WARMUP", 3);
  Problem p(M, N, K, A_rows, top_k);   // buffers passed to the reference / solution
  Problem p0(M, N, K, A_rows, top_k, 0), p1(M, N, K, A_rows, top_k, 1);
  Problem const *ps[2] = {&p0, &p1};
  float *dD = p.AllocD();
  auto restore = [&](int s) {
    Problem const &q = *ps[s];
    kh_copy(p.dA, q.dA, q.count_A * sizeof(__half));
    kh_copy(p.dB, q.dB, q.count_B * sizeof(__half));
    kh_copy(p.dC, q.dC, q.count_C * sizeof(float));
    kh_copy(p.dGA, q.dGA, M * sizeof(int));
    kh_copy(p.dGC, q.dGC, M * sizeof(int));
    kh_copy(p.dSD, q.dSD, M * sizeof(int));
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, restore, [&] {
    CutlassGatherScatterGemm(M, N, K, alpha, p.dA, p.lda, p.dGA, p.dB, p.ldb,
      beta, p.dC, p.ldc, dD, p.ldd, p.dSD);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  {
  // FP64 goldens of both input sets (computed once, outside the timed region).
  Golden g0(p0, alpha, beta), g1(p1, alpha, beta);
  Golden *gs[2] = {&g0, &g1};
  double const tol_max = TolMax(K), tol_fro = TolFro(K);
  KhTiming sol = kh_bench("Kernel time", warmup, iterations,
    [&](int s) {
      restore(s);
      kh_cuda_ok(cudaMemsetAsync(dD, 0x7F, p.count_D * sizeof(float)), "output sentinel");
    },
    [&] {
      GatherScatterGemm(M, N, K, alpha, p.dA, p.lda, p.dGA, p.dB, p.ldb,
        beta, p.dC, p.ldc, p.dGC, dD, p.ldd, p.dSD);
    },
    [&](int s) { return gs[s]->Check(*ps[s], dD, "solution (timed call)", tol_max, tol_fro, false); });
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  }
#endif

  cudaFree(dD);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Check for SM >= 80
  bool notSupported = false;
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (!(props.major >= 8)) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80." << std::endl;
    notSupported = true;
  }
  if (notSupported) return 0;

  struct TestConfig {
    const char* label;
    int m, n, k, a_rows, top_k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 1024, N = 1536, K = 2048, A_rows = 16384, top_k = 8;
    std::stringstream ss1(arg[1]); ss1 >> M;
    std::stringstream ss2(arg[2]); ss2 >> N;
    std::stringstream ss3(arg[3]); ss3 >> K;
    if (argc > 4) { std::stringstream ss(arg[4]); ss >> A_rows; }
    if (argc > 5) { std::stringstream ss(arg[5]); ss >> alpha; }
    if (argc > 6) { std::stringstream ss(arg[6]); ss >> beta; }
    if (argc > 7) { std::stringstream ss(arg[7]); ss >> iterations; }
    if (argc > 8) { std::stringstream ss(arg[8]); ss >> top_k; }
    if (M > A_rows || top_k < 1) {
      std::cerr << "Need M <= A_rows and top_k >= 1." << std::endl;
      return -1;
    }
    configs.push_back({"custom", M, N, K, A_rows, top_k});
  } else {
    // MoE expert gate+up projection with token dispatch for one expert during a 16K-token
    // prefill chunk on one GPU: A_rows = 16384 tokens, M = tokens routed to this expert
    // (~A_rows * top_k / num_experts), K = hidden size, N = per-GPU gate+up width.
    configs = {
      // Qwen3-30B-A3B: 128 experts, top-8, expert FFN 768 -> gate+up 1536
      {"qwen3_30b_a3b",      1024,  1536, 2048, 16384, 8},
      // DeepSeek-V2-Lite: 64 routed experts, top-6, expert FFN 1408 -> gate+up 2816
      {"deepseek_v2_lite",   1536,  2816, 2048, 16384, 6},
      // Mixtral-8x7B with tensor parallel 2: 8 experts, top-2, FFN 14336 / 2 -> gate+up 14336
      {"mixtral_8x7b_tp2",   4096, 14336, 4096, 16384, 2},
      // DeepSeek-V3 with expert parallelism (whole experts per GPU): 256 experts, top-8,
      // expert FFN 2048 -> gate+up 4096
      {"deepseek_v3_ep",      512,  4096, 7168, 16384, 8},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, A_rows=%d, top_k=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.a_rows, cfg.top_k);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, cfg.a_rows, cfg.top_k, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, cfg.a_rows, cfg.top_k, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
