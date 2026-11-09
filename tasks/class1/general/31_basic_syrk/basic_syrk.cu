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
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>

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

//
// CUTLASS includes needed for double-precision SYRK kernel
//

// Defines cutlass::gemm::device::RankK, the generic Syrk computation template class.
#include "cutlass/gemm/device/rank_k.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS SYRK template and launch a SYRK kernel.
cudaError_t CutlassSsyrkNN(
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassSyrk = cutlass::gemm::device::RankK<
    double,
    ColumnMajor,
    double,
    ColumnMajor,
    cutlass::FillMode::kLower,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,     // Stages
    1,     // AlignmentA
    false, // SplitKSerail
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
  >;

  CutlassSyrk syrk_operator;

  CutlassSyrk::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                              {N, N, K},
                              1,
                              {alpha, beta},
                              reinterpret_cast<void const *>(A),
                              const_cast<void *>(reinterpret_cast<void *>(C)),
                              reinterpret_cast<void *>(C),
                              (int64_t)N*K,
                              (int64_t)N*N,
                              (int64_t)N*N,
                              lda,
                              ldc,
                              ldc);

  cutlass::Status status = syrk_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
/// Pseudo-random double in [-1, 1) from (offset, seed) with 52 random mantissa bits.
/// Full-mantissa FP64 data: FP32 (or lower-precision) arithmetic cannot reproduce the result.
__host__ __device__ inline double kh_rand_double(unsigned long long offset, unsigned long long seed) {
  unsigned long long x = offset * 0x9E3779B97F4A7C15ull + (seed + 1ull) * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27; x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return double(x >> 11) * 0x1.0p-52 - 1.0;
}

__global__ void InitializeMatrix_kernel(
  double *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    size_t offset = size_t(i) + size_t(j) * ldm;
    matrix[offset] = kh_rand_double(offset, (unsigned long long)seed);
  }
}

/// Simple function to initialize a matrix with full-mantissa pseudo-random values in [-1, 1).
cudaError_t InitializeMatrix(double *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills it with pseudo-random values in [-1, 1).
cudaError_t AllocateMatrix(double **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(double) * ldm * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference SYRK computation.
__global__ void ReferenceSyrk_kernel(
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N && i >= j ) { // Since C is in Lower Fill Mode
    double accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * A[j + k * lda];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference SYRK computation.
cudaError_t ReferenceSyrk(
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (N + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceSyrk_kernel<<< grid, block >>>(N, K, alpha, A, lda, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// FP64 accuracy requirement: every lower-triangle element must satisfy
///   |x - ref| <= kFp64RelTol * max|ref|   (NaN/Inf always fail).
/// Two correct FP64 implementations (any summation order) agree to ~1e-14 of max|ref| at
/// these sizes; FP32 arithmetic or FP32-rounded inputs give ~1e-7, so they fail.
static const double kFp64RelTol = 1e-10;

/// Returns the max lower-triangle error relative to max|ref| (inf if any element is NaN/Inf
/// or the strict upper triangle of `x` differs from `upper_expected`).
static double SyrkRelError(std::vector<double> const &x, std::vector<double> const &ref,
                           std::vector<double> const &upper_expected, int N, int ldc,
                           bool *upper_ok) {
  double ref_scale = 0;
  for (int jj = 0; jj < N; ++jj)
    for (int ii = jj; ii < N; ++ii)
      ref_scale = std::fmax(ref_scale, std::fabs(ref[ii + jj * ldc]));
  if (ref_scale == 0) ref_scale = 1;
  double max_rel = 0;
  bool bad = false;
  *upper_ok = true;
  for (int jj = 0; jj < N; ++jj) {
    for (int ii = 0; ii < jj; ++ii) {   // strict upper triangle must be left unchanged
      double a = x[ii + jj * ldc], b = upper_expected[ii + jj * ldc];
      if (std::memcmp(&a, &b, sizeof(double)) != 0) *upper_ok = false;
    }
    for (int ii = jj; ii < N; ++ii) {
      double rel = std::fabs(x[ii + jj * ldc] - ref[ii + jj * ldc]) / ref_scale;
      if (!(rel <= kFp64RelTol)) bad = true;   // also catches NaN/Inf
      if (!std::isfinite(rel)) rel = INFINITY;
      if (rel > max_rel) max_rel = rel;
    }
  }
  return bad ? std::fmax(max_rel, 2 * kFp64RelTol) : max_rel;
}

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int N, int K, double alpha, double beta) {
  cudaError_t result;

  int lda = N;
  int ldc = N;
  size_t sizeof_C = sizeof(double) * ldc * N;

  double *A;
  double *C_cutlass;
  double *C_reference;

  result = AllocateMatrix(&A, lda, N, K, 0);
  if (result != cudaSuccess) return result;

  result = AllocateMatrix(&C_cutlass, ldc, N, N, 101);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  result = AllocateMatrix(&C_reference, ldc, N, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(C_cutlass); return result; }

  cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  // Run CUTLASS reference
  result = CutlassSsyrkNN(N, K, alpha, A, lda, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS SYRK failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
    return result;
  }

  // Run naive reference
  result = ReferenceSyrk(N, K, alpha, A, lda, beta, C_reference, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Reference SYRK failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
    return result;
  }

  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive (lower triangle within FP64 tolerance, upper triangle untouched)
  std::vector<double> host_cutlass(size_t(ldc) * N);
  std::vector<double> host_reference(size_t(ldc) * N);
  std::vector<double> host_c_init(size_t(ldc) * N);

  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
  {
    double *C_init;
    result = AllocateMatrix(&C_init, ldc, N, N, 101);
    if (result != cudaSuccess) { cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A); return result; }
    cudaMemcpy(host_c_init.data(), C_init, sizeof_C, cudaMemcpyDeviceToHost);
    cudaFree(C_init);
  }

  {
    bool upper_ok = true;
    double rel = SyrkRelError(host_cutlass, host_reference, host_c_init, N, ldc, &upper_ok);
    if (!(rel <= kFp64RelTol) || !upper_ok) {
      std::cerr << "CUTLASS reference incorrect vs naive (rel err " << rel
                << (upper_ok ? "" : ", upper triangle modified") << ")" << std::endl;
      cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
      return cudaErrorUnknown;
    }
    std::cout << "Reference vs naive FP64 rel err: " << rel << std::endl;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output (same initial C)
  double *C_solution;
  result = AllocateMatrix(&C_solution, ldc, N, N, 101);
  if (result != cudaSuccess) {
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
    return result;
  }

  // Run solution
  result = Dsyrk(N, K, alpha, A, lda, beta, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference: lower triangle within FP64 tolerance (NaN/Inf fail),
  // strict upper triangle must be unchanged.
  std::vector<double> host_solution(size_t(ldc) * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  bool upper_ok = true;
  double sol_rel = SyrkRelError(host_solution, host_cutlass, host_c_init, N, ldc, &upper_ok);
  std::cout << "Solution vs reference FP64 rel err: " << sol_rel << " (tolerance " << kFp64RelTol << ")" << std::endl;
  if (!(sol_rel <= kFp64RelTol) || !upper_ok) {
    std::cerr << "Solution incorrect. Max error vs reference (relative to max|ref|): " << sol_rel
              << (upper_ok ? "" : "; strict upper triangle of C was modified") << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(A);
    return cudaErrorUnknown;
  }

  cudaFree(C_solution);
#endif

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
/// Device-side form of the SyrkRelError rule, used after every timed solution call (no host
/// copies): every lower-triangle element must satisfy |x - ref| / ref_scale <= kFp64RelTol
/// (NaN/Inf fail) with ref_scale = max|ref| over the lower triangle (1 if 0), and the strict upper
/// triangle of x must be bitwise equal to the initial C.
struct KhSyrkStats {
  unsigned long long n_bad;          // lower-triangle elements outside tolerance (incl. NaN/Inf)
  unsigned long long n_upper;        // strict-upper elements that differ from the initial C
  unsigned long long max_bits;       // bits of max |x - ref| / ref_scale (non-negative double)
};

__global__ void kh_lower_absmax_kernel(double const *X, int N, int ldc, unsigned long long *max_bits) {
  double m = 0.0;
  size_t const n = size_t(N) * N;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < n;
       idx += size_t(gridDim.x) * blockDim.x) {
    int i = int(idx % N), j = int(idx / N);
    if (i >= j) m = fmax(m, fabs(X[i + size_t(j) * ldc]));
  }
  for (int o = 16; o > 0; o >>= 1) m = fmax(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(max_bits, (unsigned long long)__double_as_longlong(m));
}

__global__ void kh_syrk_compare_kernel(double const *x, double const *ref, double const *c_init,
                                       int N, int ldc, double ref_scale, KhSyrkStats *st) {
  unsigned long long bad = 0, upper = 0;
  double mx = 0.0;
  size_t const n = size_t(N) * N;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < n;
       idx += size_t(gridDim.x) * blockDim.x) {
    int i = int(idx % N), j = int(idx / N);
    size_t o = i + size_t(j) * ldc;
    if (i < j) {
      if (__double_as_longlong(x[o]) != __double_as_longlong(c_init[o])) ++upper;
    } else {
      double rel = fabs(x[o] - ref[o]) / ref_scale;
      if (!(rel <= kFp64RelTol)) ++bad;   // also catches NaN/Inf
      if (!isfinite(rel)) rel = INFINITY;
      mx = fmax(mx, rel);
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
    upper += __shfl_xor_sync(0xffffffffu, upper, o);
    mx = fmax(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  if ((threadIdx.x & 31) == 0) {
    if (bad) atomicAdd(&st->n_bad, bad);
    if (upper) atomicAdd(&st->n_upper, upper);
    atomicMax(&st->max_bits, (unsigned long long)__double_as_longlong(mx));
  }
}

/// max|X| over the lower triangle, or 1 if it is 0 (the ref_scale of SyrkRelError).
static double kh_syrk_ref_scale(double const *X, int N, int ldc) {
  unsigned long long *d = nullptr, h = 0;
  kh_cuda_ok(cudaMalloc(&d, sizeof(h)), "syrk check");
  kh_cuda_ok(cudaMemset(d, 0, sizeof(h)), "syrk check");
  kh_lower_absmax_kernel<<<1024, 256>>>(X, N, ldc, d);
  kh_cuda_ok(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "syrk check");
  cudaFree(d);
  double s;
  std::memcpy(&s, &h, sizeof(s));
  return s == 0 ? 1.0 : s;
}

/// Returns true if x matches ref under the SyrkRelError rule; prints details on failure.
static bool kh_syrk_check(char const *who, double const *x, double const *ref, double const *c_init,
                          double ref_scale, int N, int ldc) {
  KhSyrkStats *d = nullptr, h;
  kh_cuda_ok(cudaMalloc(&d, sizeof(h)), "syrk check");
  kh_cuda_ok(cudaMemset(d, 0, sizeof(h)), "syrk check");
  kh_syrk_compare_kernel<<<1024, 256>>>(x, ref, c_init, N, ldc, ref_scale, d);
  kh_cuda_ok(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "syrk check");
  cudaFree(d);
  if (h.n_bad == 0 && h.n_upper == 0) return true;
  double mx;
  std::memcpy(&mx, &h.max_bits, sizeof(mx));
  std::cerr << who << ": max error vs reference (relative to max|ref|) " << mx << " (tolerance "
            << kFp64RelTol << "), " << h.n_bad << " lower-triangle elements out of tolerance, "
            << h.n_upper << " strict-upper elements modified" << std::endl;
  return false;
}

#endif  // KH_TEST_SOLUTION

/// Profile the reference (and the solution if KH_TEST_SOLUTION) with the KH timing protocol:
/// two input sets (A, C) alternate between calls and are restored into the same buffers outside
/// the timed region (C is updated in place); every timed solution call is verified.
void Profile(int N, int K, double alpha, double beta, int iterations) {
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  int const warmup = kh_env_int("KH_WARMUP", 3);
  int lda = N, ldc = N;
  size_t const bytes_A = sizeof(double) * size_t(lda) * K, bytes_C = sizeof(double) * size_t(ldc) * N;

  // Buffers passed to the reference / solution, and two input sets (set 0 = the correctness data).
  int const seed_A[2] = {0, 7}, seed_C[2] = {101, 207};
  double *A, *C, *A_set[2], *C_set[2];
  if (AllocateMatrix(&A, lda, N, K, seed_A[0]) != cudaSuccess ||
      AllocateMatrix(&C, ldc, N, N, seed_C[0]) != cudaSuccess) kh_fail("Profile: allocation failed");
  for (int s = 0; s < 2; ++s) {
    if (AllocateMatrix(&A_set[s], lda, N, K, seed_A[s]) != cudaSuccess ||
        AllocateMatrix(&C_set[s], ldc, N, N, seed_C[s]) != cudaSuccess) kh_fail("Profile: allocation failed");
  }
  auto prep = [&](int s) {
    kh_copy(A, A_set[s], bytes_A);
    kh_copy(C, C_set[s], bytes_C);
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, prep, [&] {
    CutlassSsyrkNN(N, K, alpha, A, lda, beta, C, ldc);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  // Golden of each set = the CUTLASS reference result on that set (computed once, outside the
  // timed region), as in TestCorrectness. Set 1's CUTLASS result is checked against the naive
  // FP64 kernel first, exactly like set 0's in TestCorrectness.
  double *G[2], ref_scale[2];
  for (int s = 0; s < 2; ++s) {
    kh_cuda_ok(cudaMalloc(&G[s], bytes_C), "golden");
    kh_copy(G[s], C_set[s], bytes_C);
    kh_cuda_ok(CutlassSsyrkNN(N, K, alpha, A_set[s], lda, beta, G[s], ldc), "golden (CUTLASS)");
  }
  {
    double *naive;
    kh_cuda_ok(cudaMalloc(&naive, bytes_C), "golden check");
    kh_copy(naive, C_set[1], bytes_C);
    kh_cuda_ok(ReferenceSyrk(N, K, alpha, A_set[1], lda, beta, naive, ldc), "golden check (naive)");
    kh_cuda_ok(cudaDeviceSynchronize(), "golden check");
    if (!kh_syrk_check("CUTLASS reference vs naive (timing set 1)", G[1], naive, C_set[1],
                       kh_syrk_ref_scale(naive, N, ldc), N, ldc))
      kh_fail("CUTLASS reference incorrect vs naive on timing set 1.");
    cudaFree(naive);
  }
  for (int s = 0; s < 2; ++s) ref_scale[s] = kh_syrk_ref_scale(G[s], N, ldc);

  KhTiming sol = kh_bench("Kernel time", warmup, iterations, prep, [&] {
    Dsyrk(N, K, alpha, A, lda, beta, C, ldc);
  }, [&](int s) {
    return kh_syrk_check("Solution (timed call)", C, G[s], C_set[s], ref_scale[s], N, ldc);
  });
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  for (int s = 0; s < 2; ++s) cudaFree(G[s]);
#endif

  for (int s = 0; s < 2; ++s) {
    cudaFree(A_set[s]);
    cudaFree(C_set[s]);
  }
  cudaFree(C);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  bool notSupported = false;

  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "NVIDIA Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "This example requires compute capability at least 80." << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  struct TestConfig {
    const char* label;
    int n, k;
  };

  bool explicit_size = (argc >= 3);

  std::vector<TestConfig> configs;
  // Non-trivial alpha/beta so the alpha*A*A^T + beta*C epilogue is exercised.
  double alpha = 1.1, beta = -0.7;
  int iterations = 20;

  if (explicit_size) {
    int N = 128, K = 128;
    std::stringstream(arg[1]) >> N;
    std::stringstream(arg[2]) >> K;
    if (argc > 3) std::stringstream(arg[3]) >> alpha;
    if (argc > 4) std::stringstream(arg[4]) >> beta;
    if (argc > 5) std::stringstream(arg[5]) >> iterations;
    configs.push_back({"custom", N, K});
  } else {
    configs = {
      {"small",   1024, 1024},
      {"medium",  4096, 4096},
      {"large",   4096, 8192},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: N=%d, K=%d ===\n", cfg.label, cfg.n, cfg.k);

    cudaError_t result = TestCorrectness(cfg.n, cfg.k, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.n, cfg.k, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
