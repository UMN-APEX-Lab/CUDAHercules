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


/*! \file
    \brief Strided batched SGEMM (FP32, column-major) benchmark harness.

    C_b = alpha * A_b * B_b + beta * C_b for b in [0, batch_count), all matrices FP32 column-major,
    B batches interleaved along its leading dimension (ldb = K * batch_count, batch stride K).

    The reference is CUTLASS GemmBatched (SIMT FP32) with a per-shape threadblock configuration;
    the candidate (solution.h, -DKH_TEST_SOLUTION) implements StridedBatchedSgemm(). Both are verified
    on the device against a golden result with FP64 accumulation.

    Usage:
      ./test                                  # all problem sizes
      ./test M N K batch [alpha beta iterations]
      ./test ... --skip-verification          # do not verify the reference
*/

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

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
#include "cutlass/layout/matrix.h"

#pragma warning( disable : 4503)

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Reference and golden-reference headers (included after the solution on purpose)
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/util/reference/device/gemm_complex.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS GemmBatched, FP32 SIMT, per-shape threadblock configuration
// (offline sweep on RTX PRO 6000 Blackwell)

template <int TbM, int TbN, int WarpM, int WarpN, int Stages>
cudaError_t kh_ref_sgemm_config(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float *C, int ldc, long long int batch_stride_C,
  float beta, int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM, TbN, 8>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    Stages,
    1, 1>;

  Gemm gemm_op;
  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, batch_stride_A,
    {B, ldb}, batch_stride_B,
    {C, ldc}, batch_stride_C,
    {C, ldc}, batch_stride_C,
    {alpha, beta},
    batch_count
  });
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t kh_ref_strided_batched_sgemm(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float *C, int ldc, long long int batch_stride_C,
  float beta, int batch_count) {
  // Skinny M (GQA decode): 128x64 threadblocks (more CTAs along N); otherwise CUTLASS's default
  // FP32 SIMT configuration (128x128x8, 32x64 warps, 2 stages), fastest at the other shapes.
  if (m <= 64) {
    return kh_ref_sgemm_config<128, 64, 64, 32, 4>(m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B,
                                                  C, ldc, batch_stride_C, beta, batch_count);
  }
  return kh_ref_sgemm_config<128, 128, 32, 64, 2>(m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B,
                                                 C, ldc, batch_stride_C, beta, batch_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Golden reference for strided batched GEMM NN (column-major), computed on the device with
/// FP64 accumulation (CUTLASS device reference GEMM). Reads C before it is overwritten.
cudaError_t kh_golden_strided_batched_gemm(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float const *C, int ldc, long long int batch_stride_C,
  float beta, float *D, int batch_count) {

  using Layout = cutlass::layout::ColumnMajor;
  cutlass::reference::device::GemmComplex<
    float, Layout, float, Layout, float, Layout, double, double
  >(
    {m, n, k},
    double(alpha),
    cutlass::TensorRef<float, Layout>(const_cast<float *>(A), Layout(lda)),
    cutlass::ComplexTransform::kNone,
    cutlass::TensorRef<float, Layout>(const_cast<float *>(B), Layout(ldb)),
    cutlass::ComplexTransform::kNone,
    double(beta),
    cutlass::TensorRef<float, Layout>(const_cast<float *>(C), Layout(ldc)),
    cutlass::TensorRef<float, Layout>(D, Layout(ldc)),
    double(0),
    batch_count, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_C);
  return cudaGetLastError();
}

/// Deterministic fill with uniform values in [-1, 1) carrying 24 random bits each, so that the
/// data needs the full FP32 mantissa (lower-precision arithmetic cannot reproduce the result).
__global__ void FillRandomUniform_kernel(float *ptr, size_t count, uint32_t seed) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9E3779B9u) ^ (seed * 0x85EBCA6Bu);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    ptr[i] = float(int32_t(h >> 8) - (1 << 23)) * (1.0f / float(1 << 23));
  }
}

cudaError_t FillRandomUniform(float *ptr, size_t count, uint32_t seed) {
  FillRandomUniform_kernel<<<1024, 256>>>(ptr, count, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side element-wise check against the golden result.
//
// With inputs in [-1, 1), sequential FP32 accumulation over K reaches an absolute error of about
// 7e-8 * K * |alpha| (rounding errors accumulate along K). The bound is
// kTolK * |alpha| * K + kTolBeta * |beta|: sequential FP32 accumulation stays >= 3.1x below it at
// every benchmark size (K = 128 .. 8192). TF32/FP16 operands exceed it by >= 8x, BF16 operands by
// > 100x; 3xBF16 (2-term split, error ~ 1e-5 * sqrt(K)) exceeds it at K = 128 (decode and
// prefill scores). FP32-accurate emulations (3xTF32, 6xBF16, 9xBF16) pass at every size.
// Non-finite values always fail.

static constexpr double kTolK = 3.7e-7;
static constexpr double kTolBeta = 3e-5;

struct CheckStats {
  unsigned long long fails;
  unsigned long long first_fail;   // linear index of the first failing element
  unsigned int max_err_bits;       // float bits of max |err| (non-negative floats order like ints)
  double err_sq;
  double gold_sq;
};

__global__ void check_kernel(float const *x, float const *gold, size_t count, float tol, CheckStats *stats) {
  __shared__ double red_e[32], red_g[32];
  __shared__ float red_m[32];
  double es = 0, gs = 0;
  float mx = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    float const v = x[i], g = gold[i];
    float err = fabsf(v - g);
    if (!isfinite(v) || !(err <= tol)) {
      atomicAdd(&stats->fails, 1ull);
      atomicMin(&stats->first_fail, (unsigned long long)i);
      if (!isfinite(v)) err = INFINITY;
    }
    mx = fmaxf(mx, err);
    es += double(err) * double(err);
    gs += double(g) * double(g);
  }
  for (int o = 16; o > 0; o >>= 1) {
    es += __shfl_xor_sync(0xffffffffu, es, o);
    gs += __shfl_xor_sync(0xffffffffu, gs, o);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { red_e[warp] = es; red_g[warp] = gs; red_m[warp] = mx; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < int(blockDim.x >> 5); ++w) { es += red_e[w]; gs += red_g[w]; mx = fmaxf(mx, red_m[w]); }
    atomicAdd(&stats->err_sq, es);
    atomicAdd(&stats->gold_sq, gs);
    atomicMax(&stats->max_err_bits, __float_as_uint(mx));
  }
}

bool check_output(char const *who, float const *d_x, float const *d_gold,
                  int m, int n, int k, long long int batch_stride_C, int batch_count,
                  float alpha, float beta, bool quiet = false) {
  double const tol = kTolK * std::fabs(double(alpha)) * double(k) + kTolBeta * std::fabs(double(beta));
  size_t const count = size_t(batch_stride_C) * batch_count;  // packed: ldc = m

  CheckStats init{0ull, ~0ull, 0u, 0.0, 0.0}, stats;
  CheckStats *d_stats = nullptr;
  cudaMalloc(&d_stats, sizeof(CheckStats));
  cudaMemcpy(d_stats, &init, sizeof(CheckStats), cudaMemcpyHostToDevice);
  check_kernel<<<2048, 256>>>(d_x, d_gold, count, float(tol), d_stats);
  cudaError_t err = cudaMemcpy(&stats, d_stats, sizeof(CheckStats), cudaMemcpyDeviceToHost);
  cudaFree(d_stats);
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: verification kernel failed: %s\n", who, cudaGetErrorString(err));
    return false;
  }

  float max_err;
  unsigned int bits = stats.max_err_bits;
  std::memcpy(&max_err, &bits, sizeof(float));
  double const rel_rms = std::sqrt(stats.err_sq / std::max(stats.gold_sq, 1e-300));
  if (stats.fails) {
    size_t const i = stats.first_fail;
    float got, expected;
    cudaMemcpy(&got, d_x + i, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&expected, d_gold + i, sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%s: mismatch at batch %lld, row %lld, col %lld: got %.8g, expected %.8g (tol %.3g)\n",
            who, (long long)(i / batch_stride_C), (long long)(i % m), (long long)((i % batch_stride_C) / m),
            got, expected, tol);
  }
  if (quiet && stats.fails == 0) return true;  // timed calls: details only on failure
  fprintf(stdout, "%s: max |err| = %.3g (tol %.3g, max |err|/tol = %.3f), rms(err)/rms(ref) = %.3g, "
          "%llu of %zu elements out of tolerance\n",
          who, max_err, tol, max_err / tol, rel_rms, stats.fails, count);
  return stats.fails == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies the reference (unless skipped) and the solution (if KH_TEST_SOLUTION) against the
/// golden result, then profiles both. Returns false on any failure.
bool RunSize(int m, int n, int k, int batch_count, float alpha, float beta, int iterations,
             bool skip_reference_verification) {

  // Column-major layout matching the original example
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  size_t const count_A = size_t(batch_count) * lda * k;
  size_t const count_B = size_t(ldb) * n;
  size_t const count_C = size_t(batch_count) * ldc * n;

  long long int const batch_stride_A = (long long int)(lda) * (long long int)(k);
  long long int const batch_stride_B = (long long int)(k);
  long long int const batch_stride_C = (long long int)(ldc) * (long long int)(n);

  size_t const sizeof_C = count_C * sizeof(float);
  uint32_t const seed_C = 3;

#ifdef KH_TEST_SOLUTION
  bool const need_golden = true;
#else
  bool const need_golden = !skip_reference_verification;
#endif

  float *d_A = nullptr, *d_B = nullptr, *d_C_ref = nullptr, *d_golden = nullptr, *d_C_sol = nullptr;
  auto free_all = [&]() {
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_golden); cudaFree(d_C_sol);
  };

  cudaError_t result;
  if ((result = cudaMalloc(&d_A, count_A * sizeof(float))) != cudaSuccess ||
      (result = cudaMalloc(&d_B, count_B * sizeof(float))) != cudaSuccess ||
      (result = cudaMalloc(&d_C_ref, sizeof_C)) != cudaSuccess ||
      (need_golden && (result = cudaMalloc(&d_golden, sizeof_C)) != cudaSuccess)
#ifdef KH_TEST_SOLUTION
      || (result = cudaMalloc(&d_C_sol, sizeof_C)) != cudaSuccess
#endif
      ) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(result) << std::endl;
    free_all();
    return false;
  }

  // Random inputs; C holds random values as well (it is an input when beta != 0), so unwritten
  // outputs are detected.
  FillRandomUniform(d_A, count_A, 1);
  FillRandomUniform(d_B, count_B, 2);
  FillRandomUniform(d_C_ref, count_C, seed_C);

  // Golden result (reads the initial C)
  if (need_golden) {
    result = kh_golden_strided_batched_gemm(
      m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
      d_C_ref, ldc, batch_stride_C, beta, d_golden, batch_count);
    if (result == cudaSuccess) result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Golden reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
      free_all();
      return false;
    }
  }

  // Reference
  result = kh_ref_strided_batched_sgemm(
    m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
    d_C_ref, ldc, batch_stride_C, beta, batch_count);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Reference batched GEMM failed: " << cudaGetErrorString(result) << std::endl;
    free_all();
    return false;
  }
  if (!skip_reference_verification) {
    if (!check_output("Reference", d_C_ref, d_golden, m, n, k, batch_stride_C, batch_count, alpha, beta)) {
      free_all();
      return false;
    }
  } else {
    std::cout << "Baseline reference verification skipped." << std::endl;
  }

#ifdef KH_TEST_SOLUTION
  // Runs the solution once on a freshly initialized C and verifies it.
  auto run_and_verify_solution = [&](char const *who) -> bool {
    FillRandomUniform(d_C_sol, count_C, seed_C);
    cudaError_t err = StridedBatchedSgemm(
      m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
      d_C_sol, ldc, batch_stride_C, beta, batch_count);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output(who, d_C_sol, d_golden, m, n, k, batch_stride_C, batch_count, alpha, beta);
  };
  if (!run_and_verify_solution("Solution")) {
    std::cerr << "Solution incorrect." << std::endl;
    free_all();
    return false;
  }
#endif

  std::cout << "Passed" << std::endl;

  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  if (iterations > 0) {
    int const warmup = kh_env_int("KH_WARMUP", 3);
    // KH timing protocol: two input sets (set 0 = the correctness data, set 1 = new seeds). The
    // generator is deterministic, so prep(s) regenerates set s in place (same pointers) instead of
    // copying it from a stored pristine copy; outputs are re-initialized before every call.
    uint32_t const seed_A[2] = {1, 11}, seed_B[2] = {2, 12}, seed_Cs[2] = {seed_C, 13};
    auto restore_AB = [&](int s) {
      kh_cuda_ok(FillRandomUniform(d_A, count_A, seed_A[s]), "input restore");
      kh_cuda_ok(FillRandomUniform(d_B, count_B, seed_B[s]), "input restore");
    };
    // C holds the random initial values of set s (an input when beta != 0; with beta == 0 they act
    // as a finite poison, 0 * C = 0, so elements the call does not write fail the check).
    auto init_C = [&](float *C, int s) {
      kh_cuda_ok(FillRandomUniform(C, count_C, seed_Cs[s]), "output re-initialization");
    };

    KhTiming const ref_t = kh_bench("Ref time", warmup, iterations,
      [&](int s) { restore_AB(s); if (beta != 0.0f) init_C(d_C_ref, s); },
      [&] {
        kh_ref_strided_batched_sgemm(m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
                                     d_C_ref, ldc, batch_stride_C, beta, batch_count);
      }, kh_no_check);
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            ref_t.avg_ms, iterations, ref_t.min_ms);
    float const ref_min_ms = ref_t.min_ms;
#ifdef KH_TEST_SOLUTION
    // Golden of set 1 (computed once, outside the timed region; reads the initial C of set 1).
    float *d_golden1 = nullptr;
    kh_cuda_ok(cudaMalloc(&d_golden1, sizeof_C), "golden (timing set 1)");
    restore_AB(1);
    kh_cuda_ok(FillRandomUniform(d_C_sol, count_C, seed_Cs[1]), "golden (timing set 1)");
    kh_cuda_ok(kh_golden_strided_batched_gemm(
      m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
      d_C_sol, ldc, batch_stride_C, beta, d_golden1, batch_count), "golden (timing set 1)");
    kh_cuda_ok(cudaDeviceSynchronize(), "golden (timing set 1)");
    float const *gold[2] = {d_golden, d_golden1};

    KhTiming const sol_t = kh_bench("Kernel time", warmup, iterations,
      [&](int s) { restore_AB(s); init_C(d_C_sol, s); },
      [&] {
        StridedBatchedSgemm(m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
                            d_C_sol, ldc, batch_stride_C, beta, batch_count);
      },
      [&](int s) {
        return check_output("Solution (timed call)", d_C_sol, gold[s], m, n, k, batch_stride_C,
                            batch_count, alpha, beta, /*quiet=*/true);
      });
    cudaFree(d_golden1);
    fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            sol_t.avg_ms, iterations, sol_t.min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_t.min_ms);
    // The output must still be produced by a fresh call after the timed runs (set 0 inputs).
    restore_AB(0);
    if (!run_and_verify_solution("Solution (re-run after timing)")) {
      free_all();
      return false;
    }
#endif
    printf("Reference: %.1f TFLOP/s\n", 2.0 * m * n * double(k) * batch_count / ref_min_ms / 1e9);
  }
  free_all();
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k, batch_count;
  };

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;
  bool skip_reference_verification = false;
  std::vector<const char*> positional;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(arg[i], "--skip-verification") || !strcmp(arg[i], "--no_verif")) {
      skip_reference_verification = true;
    } else {
      positional.push_back(arg[i]);
    }
  }

  bool explicit_size = (positional.size() >= 4);

  if (explicit_size) {
    int M = 128, N = 128, K = 64, B = 4;
    std::stringstream(positional[0]) >> M;
    std::stringstream(positional[1]) >> N;
    std::stringstream(positional[2]) >> K;
    std::stringstream(positional[3]) >> B;
    if (positional.size() > 4) std::stringstream(positional[4]) >> alpha;
    if (positional.size() > 5) std::stringstream(positional[5]) >> beta;
    if (positional.size() > 6) std::stringstream(positional[6]) >> iterations;
    configs.push_back({"custom", M, N, K, B});
  } else {
    // FP32 batched GEMMs as run on one RTX PRO 6000 (96 GB); column-major, batch = heads or experts:
    //   attn_scores_prefill:    S = Q K^T per head, Qwen3-32B (64 query heads, head_dim 128),
    //                           4K-token query chunk against 8K keys
    //   attn_pv_prefill:        O = P V per head, same shapes
    //   attn_scores_decode_gqa: Qwen3-32B decode, 16 seqs x 8 KV heads, 8 query heads per KV head,
    //                           6K-token context
    //   moe_experts:            Qwen3-30B-A3B, 128 experts x 1024 tokens (equal capacity),
    //                           2048 -> 1536 (gate + up projection, 2 x 768)
    configs = {
      {"attn_scores_prefill",    4096, 8192,  128,  64},
      {"attn_pv_prefill",        4096,  128, 8192,  64},
      {"attn_scores_decode_gqa",    8, 6144,  128, 128},
      {"moe_experts",            1024, 1536, 2048, 128},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch_count);
    fflush(stdout);

    if (!RunSize(cfg.m, cfg.n, cfg.k, cfg.batch_count, alpha, beta, iterations, skip_reference_verification)) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
