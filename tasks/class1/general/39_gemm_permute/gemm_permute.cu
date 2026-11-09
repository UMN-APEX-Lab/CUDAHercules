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

// GEMM Permute example — half precision, row-major
// Based on CUTLASS example 39_gemm_permute, restructured for CUDA-Hercules testing.
// D = alpha * A * B + beta * D  (row-major, FP16 with FP32 accumulation)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_fp16.h>

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

#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS FP16 tensor-core GEMM (FP32 accumulation, FP16 output), all row-major,
// with a per-size configuration chosen by an offline sweep on the benchmark GPU
// (RTX PRO 6000 Blackwell, sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using RowMajor = cutlass::layout::RowMajor;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, int Swizzle = 1>
using HgemmT = cutlass::gemm::device::Gemm<
    cutlass::half_t, RowMajor,    // A
    cutlass::half_t, RowMajor,    // B
    cutlass::half_t, RowMajor,    // C/D
    float,                        // ElementAccumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages, 8, 8>;

// Per-size configurations (offline sweep). No serial split-K: it would round the partial sums to
// FP16 in D, i.e. not accumulate in FP32.
//   0: 128x128x32 threadblock, 64x64x32 warp, 3 stages            (decode_lm_head, other shapes)
//   1: 128x256x32 threadblock, 64x64x32 warp, 3 stages, swizzle 2 (16K-token prefill shapes)
//   2:  64x128x32 threadblock, 32x64x32 warp, 6 stages            (decode_qkv, small M)
using Hgemm0 = HgemmT<128, 128, 32, 64, 64, 32, 3>;
using Hgemm1 = HgemmT<128, 256, 32, 64, 64, 32, 3, 2>;
using Hgemm2 = HgemmT<64, 128, 32, 32, 64, 32, 6>;

static int RefConfigFor(int M, int N, int K) {
  if (M == 128 && N == 151936 && K == 5120) return 0;   // decode_lm_head
  if (M >= 4096) return 1;                               // prefill shapes
  if (M <= 512) return 2;                                // decode_qkv and other small-M shapes
  return 0;
}


template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha, cutlass::half_t const *A, int lda,
                              cutlass::half_t const *B, int ldb, float beta, cutlass::half_t *D,
                              int ldd) {
  typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {D, ldd}, {D, ldd}, {alpha, beta});
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (gemm_op.initialize(args, nullptr) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm_op() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaGetLastError();
}

/// Run CUTLASS GEMM: D = alpha * A * B + beta * D (row-major, half precision)
cudaError_t CutlassHgemm(
    int M, int N, int K,
    float alpha,
    cutlass::half_t const *A, int lda,
    cutlass::half_t const *B, int ldb,
    float beta,
    cutlass::half_t *D, int ldd) {
  switch (RefConfigFor(M, N, K)) {
    case 1: return RunCutlass<Hgemm1>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    case 2: return RunCutlass<Hgemm2>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    default: return RunCutlass<Hgemm0>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  }
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

// FP16 output of an FP32-accumulated product: a correct result is within half an FP16 ulp
// (2^-11 relative) of the exact value plus a small FP32 accumulation error. Element bound:
// |D - exact| <= kTolRel * |exact| + kTolMax * RMS(exact) (3x the FP16 rounding bound, >= 4x
// the FP32 accumulation error); relative Frobenius error <= kTolFro (~3x a correct result).
// FP16 accumulation (~1e-2) or FP8/INT8 operands (> 1e-2) fail both bounds.
static const double kTolRel = 1.5e-3;
static const double kTolMax = 2e-4;
static const double kTolFro = 7.5e-4;

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against a golden
/// result computed on the device (FP16 products are exact in FP32). Every element is checked.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  int lda = K, ldb = N, ldd = N;  // row-major
  size_t count_A = size_t(M) * lda, count_B = size_t(K) * ldb, count_D = size_t(M) * ldd;

  __half *A, *B, *D_ref;
  double *G;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__half)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__half)));
  KH_CHECK(cudaMalloc(&D_ref, count_D * sizeof(__half)));
  KH_CHECK(cudaMalloc(&G, count_D * sizeof(double)));
  kh_fill(A, count_A, 1u);
  kh_fill(B, count_B, 2u);
  // Initial D: random when beta != 0; otherwise a finite poison (0 * poison = 0) so that
  // elements the solution never writes fail.
  if (beta != 0.0f) kh_fill(D_ref, count_D, 3u);
  else kh_fill_value(D_ref, count_D, kPoison16);

  kh_golden_gemm<float>(M, N, K, alpha, A, lda, 1, B, ldb, 1, beta, D_ref, ldd, 1, G);

  cudaError_t result = cudaSuccess;
#ifdef KH_TEST_SOLUTION
  __half *D_sol;
  KH_CHECK(cudaMalloc(&D_sol, count_D * sizeof(__half)));
  KH_CHECK(cudaMemcpy(D_sol, D_ref, count_D * sizeof(__half), cudaMemcpyDeviceToDevice));
#endif

  result = CutlassHgemm(M, N, K, alpha, reinterpret_cast<cutlass::half_t const *>(A), lda,
                        reinterpret_cast<cutlass::half_t const *>(B), ldb, beta,
                        reinterpret_cast<cutlass::half_t *>(D_ref), ldd);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
  } else {
    KhResult r = kh_check(D_ref, ldd, 1, G, M, N, kTolMax, kTolFro, kTolRel);
    kh_print("reference", r, kTolMax, kTolFro);
    if (!r.pass) {
      std::cerr << "CUTLASS reference incorrect vs golden." << std::endl;
      result = cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    result = GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D_sol, ldd);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(D_sol, ldd, 1, G, M, N, kTolMax, kTolFro, kTolRel);
      kh_print("solution ", r, kTolMax, kTolFro);
      if (!r.pass) {
        std::cerr << "Solution incorrect vs golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", elements over tolerance (" << kTolRel << "*|ref| + " << kTolMax
                  << "*rms) = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
  }
  cudaFree(D_sol);
#endif

  cudaFree(G);
  cudaFree(D_ref);
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
  int lda = K, ldb = N, ldd = N;
  size_t count_A = size_t(M) * lda, count_B = size_t(K) * ldb, count_D = size_t(M) * ldd;
  bool const has_C = (beta != 0.0f);  // D is also an input (in place) when beta != 0

  // Buffers passed to the reference / solution, and two input sets (set 0 = the correctness data).
  __half *A, *B, *D, *A_set[2], *B_set[2], *D_set[2] = {nullptr, nullptr};
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__half)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__half)));
  KH_CHECK(cudaMalloc(&D, count_D * sizeof(__half)));
  for (int s = 0; s < 2; ++s) {
    KH_CHECK(cudaMalloc(&A_set[s], count_A * sizeof(__half)));
    KH_CHECK(cudaMalloc(&B_set[s], count_B * sizeof(__half)));
    kh_fill(A_set[s], count_A, 1u + 10u * s);
    kh_fill(B_set[s], count_B, 2u + 10u * s);
    if (has_C) {
      KH_CHECK(cudaMalloc(&D_set[s], count_D * sizeof(__half)));
      kh_fill(D_set[s], count_D, 3u + 10u * s);
    }
  }
  auto prep = [&](int s) {
    kh_copy(A, A_set[s], count_A * sizeof(__half));
    kh_copy(B, B_set[s], count_B * sizeof(__half));
    if (has_C) kh_copy(D, D_set[s], count_D * sizeof(__half));
    else kh_fill_value(D, count_D, kPoison16);
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, prep, [&] {
    CutlassHgemm(M, N, K, alpha, reinterpret_cast<cutlass::half_t const *>(A), lda,
                 reinterpret_cast<cutlass::half_t const *>(B), ldb, beta,
                 reinterpret_cast<cutlass::half_t *>(D), ldd);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  // Goldens of both input sets (FP16 products are exact in FP32; computed once, outside the
  // timed region).
  double *G[2];
  for (int s = 0; s < 2; ++s) {
    KH_CHECK(cudaMalloc(&G[s], count_D * sizeof(double)));
    kh_golden_gemm<float>(M, N, K, alpha, A_set[s], lda, 1, B_set[s], ldb, 1, beta,
                          has_C ? D_set[s] : D, ldd, 1, G[s]);
  }
  KH_CHECK(cudaDeviceSynchronize());
  auto check = [&](int s) {
    KhResult r = kh_check(D, ldd, 1, G[s], M, N, kTolMax, kTolFro, kTolRel);
    if (!r.pass) kh_print("solution (timed call)", r, kTolMax, kTolFro);
    return r.pass;
  };

  KhTiming sol = kh_bench("Kernel time", warmup, iterations, prep, [&] {
    GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  }, check);
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  for (int s = 0; s < 2; ++s) cudaFree(G[s]);
#endif

  for (int s = 0; s < 2; ++s) {
    cudaFree(A_set[s]);
    cudaFree(B_set[s]);
    cudaFree(D_set[s]);
  }
  cudaFree(D);
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
    // Qwen3-32B linear layers on one GPU (hidden 5120, 64 q / 8 kv heads x 128 -> fused QKV
    // 10240, FFN 25600, vocab 151936); M = tokens: 16K-token prefill chunk, 128-sequence decode.
    configs = {
      {"decode_qkv",        128,  10240,  5120},   // fused QKV proj, 128 decode tokens
      {"prefill_qkv",     16384,  10240,  5120},   // fused QKV proj, 16K-token prefill chunk
      {"prefill_o_proj",  16384,   5120,  8192},   // attention output proj (64 heads x 128)
      {"prefill_mlp_up",  16384,  51200,  5120},   // fused gate+up proj (2 x 25600)
      {"prefill_mlp_down",16384,   5120, 25600},   // down proj
      {"decode_lm_head",    128, 151936,  5120},   // LM head, 128 decode tokens
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
