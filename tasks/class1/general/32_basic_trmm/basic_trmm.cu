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
// CUTLASS includes needed for double-precision TRMM kernel
//

// Defines cutlass::gemm::device::Trmm, the generic Trmm computation template class.
#include "cutlass/gemm/device/trmm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS TRMM template and launch a TRMM kernel.
cudaError_t CutlassStrmmNN(
  int M,
  int N,
  double alpha,
  double const *A,
  int lda,
  double const *B,
  int ldb,
  double *C,
  int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassTrmm = cutlass::gemm::device::Trmm<
    double,
    ColumnMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kLower,
    cutlass::DiagType::kNonUnit,
    double,
    ColumnMajor,
    double,
    ColumnMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd
  >;

  CutlassTrmm trmm_operator;

  CutlassTrmm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                              {M, N, M},
                              1,
                              {alpha},
                              reinterpret_cast<void const *>(A),
                              reinterpret_cast<void const *>(B),
                              reinterpret_cast<void *>(C),
                              (int64_t)M*M,
                              (int64_t)M*N,
                              (int64_t)M*N,
                              lda,
                              ldb,
                              ldc);

  cutlass::Status status = trmm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Pseudo-random double in [-1, 1) from (offset, seed) with 52 random mantissa bits.
/// Full-mantissa FP64 data: FP32 (or lower-precision) arithmetic cannot reproduce the result.
__host__ __device__ inline double kh_rand_double(unsigned long long offset, unsigned long long seed) {
  unsigned long long x = offset * 0x9E3779B97F4A7C15ull + (seed + 1ull) * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27; x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return double(x >> 11) * 0x1.0p-52 - 1.0;
}

/// Kernel to initialize a matrix with full-mantissa pseudo-random values in [-1, 1).
__global__ void InitializeMatrix_kernel(
  double *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0,
  cutlass::FillMode fill_mode = cutlass::FillMode::kInvalid) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    if (fill_mode == cutlass::FillMode::kLower && i < j) return;
    else if (fill_mode == cutlass::FillMode::kUpper && i > j) return;
    size_t offset = size_t(i) + size_t(j) * ldm;
    matrix[offset] = kh_rand_double(offset, (unsigned long long)seed);
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(double *matrix, int ldm, int rows, int columns, int seed = 0,
                             cutlass::FillMode fill_mode = cutlass::FillMode::kInvalid) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed, fill_mode);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills it with pseudo-random values in [-1, 1).
cudaError_t AllocateMatrix(double **matrix, int ldm, int rows, int columns, int seed = 0,
                           cutlass::FillMode fill_mode = cutlass::FillMode::kInvalid) {
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

  result = InitializeMatrix(*matrix, ldm, rows, columns, seed, fill_mode);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference TRMM computation.
__global__ void ReferenceTrmm_kernel(
  int M,
  int N,
  double alpha,
  double const *A,
  int lda,
  double const *B,
  int ldb,
  double *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    double accumulator = 0;

    for (int k = 0; k < M; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator;
  }
}

/// Reference TRMM computation.
cudaError_t ReferenceTrmm(
  int M,
  int N,
  double alpha,
  double const *A,
  int lda,
  double const *B,
  int ldb,
  double *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceTrmm_kernel<<< grid, block >>>(M, N, alpha, A, lda, B, ldb, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// FP64 accuracy requirement: every element must satisfy |x - ref| <= kFp64RelTol * max|ref|
/// (NaN/Inf always fail). Two correct FP64 implementations (any summation order) agree to
/// ~1e-14 of max|ref| at these sizes; FP32 arithmetic or FP32-rounded inputs give ~1e-7.
static const double kFp64RelTol = 1e-10;

/// Max element error relative to max|ref|; >= 2*kFp64RelTol (or inf) if any element is NaN/Inf.
static double TrmmRelError(std::vector<double> const &x, std::vector<double> const &ref) {
  double ref_scale = 0;
  for (double r : ref) ref_scale = std::fmax(ref_scale, std::fabs(r));
  if (ref_scale == 0) ref_scale = 1;
  double max_rel = 0;
  bool bad = false;
  for (size_t i = 0; i < ref.size(); ++i) {
    double rel = std::fabs(x[i] - ref[i]) / ref_scale;
    if (!(rel <= kFp64RelTol)) bad = true;   // also catches NaN/Inf
    if (!std::isfinite(rel)) rel = INFINITY;
    if (rel > max_rel) max_rel = rel;
  }
  return bad ? std::fmax(max_rel, 2 * kFp64RelTol) : max_rel;
}

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, double alpha) {
  cudaError_t result;

  int lda = M;
  int ldb = M;
  int ldc = M;
  size_t sizeof_C = sizeof(double) * ldc * N;

  double *A;
  double *B;
  double *C_cutlass;
  double *C_reference;

  result = AllocateMatrix(&A, lda, M, M, 0, cutlass::FillMode::kLower);
  if (result != cudaSuccess) return result;

  result = AllocateMatrix(&B, ldb, M, N, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }

  result = AllocateMatrix(&C_reference, ldc, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(C_cutlass); return result; }

  cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  // Run CUTLASS reference
  result = CutlassStrmmNN(M, N, alpha, A, lda, B, ldb, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS TRMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run naive reference
  result = ReferenceTrmm(M, N, alpha, A, lda, B, ldb, C_reference, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Reference TRMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive (FP64 tolerance)
  std::vector<double> host_cutlass(size_t(ldc) * N);
  std::vector<double> host_reference(size_t(ldc) * N);

  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  {
    double rel = TrmmRelError(host_cutlass, host_reference);
    if (!(rel <= kFp64RelTol)) {
      std::cerr << "CUTLASS reference incorrect vs naive (rel err " << rel << ")." << std::endl;
      cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
      return cudaErrorUnknown;
    }
    std::cout << "Reference vs naive FP64 rel err: " << rel << std::endl;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output, poisoned with NaN (0xFF bytes) so unwritten elements fail
  double *C_solution;
  cudaMalloc(&C_solution, sizeof_C);
  cudaMemset(C_solution, 0xFF, sizeof_C);

  // Run solution
  result = Dtrmm(M, N, alpha, A, lda, B, ldb, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference (FP64 tolerance; NaN/Inf and unwritten elements fail)
  std::vector<double> host_solution(size_t(ldc) * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  double sol_rel = TrmmRelError(host_solution, host_cutlass);
  std::cout << "Solution vs reference FP64 rel err: " << sol_rel << " (tolerance " << kFp64RelTol << ")" << std::endl;
  if (!(sol_rel <= kFp64RelTol)) {
    std::cerr << "Solution incorrect. Max error vs reference (relative to max|ref|): " << sol_rel << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

  cudaFree(C_solution);
#endif

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
/// Device-side form of the TrmmRelError rule, used after every timed solution call (no host
/// copies): every element must satisfy |x - ref| / ref_scale <= kFp64RelTol (NaN/Inf fail) with
/// ref_scale = max|ref| (1 if 0).
struct KhTrmmStats {
  unsigned long long n_bad;          // elements outside tolerance (incl. NaN/Inf)
  unsigned long long max_bits;       // bits of max |x - ref| / ref_scale (non-negative double)
};

__global__ void kh_absmax_kernel(double const *X, size_t n, unsigned long long *max_bits) {
  double m = 0.0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    m = fmax(m, fabs(X[i]));
  for (int o = 16; o > 0; o >>= 1) m = fmax(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(max_bits, (unsigned long long)__double_as_longlong(m));
}

__global__ void kh_trmm_compare_kernel(double const *x, double const *ref, size_t n, double ref_scale,
                                       KhTrmmStats *st) {
  unsigned long long bad = 0;
  double mx = 0.0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    double rel = fabs(x[i] - ref[i]) / ref_scale;
    if (!(rel <= kFp64RelTol)) ++bad;   // also catches NaN/Inf
    if (!isfinite(rel)) rel = INFINITY;
    mx = fmax(mx, rel);
  }
  for (int o = 16; o > 0; o >>= 1) {
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
    mx = fmax(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  if ((threadIdx.x & 31) == 0) {
    if (bad) atomicAdd(&st->n_bad, bad);
    atomicMax(&st->max_bits, (unsigned long long)__double_as_longlong(mx));
  }
}

/// max|X|, or 1 if it is 0 (the ref_scale of TrmmRelError).
static double kh_trmm_ref_scale(double const *X, size_t n) {
  unsigned long long *d = nullptr, h = 0;
  kh_cuda_ok(cudaMalloc(&d, sizeof(h)), "trmm check");
  kh_cuda_ok(cudaMemset(d, 0, sizeof(h)), "trmm check");
  kh_absmax_kernel<<<1024, 256>>>(X, n, d);
  kh_cuda_ok(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "trmm check");
  cudaFree(d);
  double s;
  std::memcpy(&s, &h, sizeof(s));
  return s == 0 ? 1.0 : s;
}

/// Returns true if x matches ref under the TrmmRelError rule; prints details on failure.
static bool kh_trmm_check(char const *who, double const *x, double const *ref, double ref_scale, size_t n) {
  KhTrmmStats *d = nullptr, h;
  kh_cuda_ok(cudaMalloc(&d, sizeof(h)), "trmm check");
  kh_cuda_ok(cudaMemset(d, 0, sizeof(h)), "trmm check");
  kh_trmm_compare_kernel<<<1024, 256>>>(x, ref, n, ref_scale, d);
  kh_cuda_ok(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "trmm check");
  cudaFree(d);
  if (h.n_bad == 0) return true;
  double mx;
  std::memcpy(&mx, &h.max_bits, sizeof(mx));
  std::cerr << who << ": max error vs reference (relative to max|ref|) " << mx << " (tolerance "
            << kFp64RelTol << "), " << h.n_bad << " elements out of tolerance" << std::endl;
  return false;
}

#endif  // KH_TEST_SOLUTION

/// Profile the reference (and the solution if KH_TEST_SOLUTION) with the KH timing protocol:
/// two input sets (A, B) alternate between calls and are restored into the same buffers outside
/// the timed region; the output is poisoned (NaN) before and verified after every timed
/// solution call.
void Profile(int M, int N, double alpha, int iterations) {
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  int const warmup = kh_env_int("KH_WARMUP", 3);
  int lda = M, ldb = M, ldc = M;
  size_t const bytes_A = sizeof(double) * size_t(lda) * M, bytes_B = sizeof(double) * size_t(ldb) * N;

  // Buffers passed to the reference / solution, and two input sets (set 0 = the correctness data;
  // A is lower triangular with a zero strict upper triangle in both sets).
  int const seed_A[2] = {0, 7}, seed_B[2] = {17, 27};
  double *A, *B, *C, *A_set[2], *B_set[2];
  if (AllocateMatrix(&A, lda, M, M, seed_A[0], cutlass::FillMode::kLower) != cudaSuccess ||
      AllocateMatrix(&B, ldb, M, N, seed_B[0]) != cudaSuccess ||
      AllocateMatrix(&C, ldc, M, N, 101) != cudaSuccess) kh_fail("Profile: allocation failed");
  for (int s = 0; s < 2; ++s) {
    if (AllocateMatrix(&A_set[s], lda, M, M, seed_A[s], cutlass::FillMode::kLower) != cudaSuccess ||
        AllocateMatrix(&B_set[s], ldb, M, N, seed_B[s]) != cudaSuccess) kh_fail("Profile: allocation failed");
  }
  auto restore_inputs = [&](int s) {
    kh_copy(A, A_set[s], bytes_A);
    kh_copy(B, B_set[s], bytes_B);
  };

  KhTiming ref = kh_bench("Ref time", warmup, iterations, restore_inputs, [&] {
    CutlassStrmmNN(M, N, alpha, A, lda, B, ldb, C, ldc);
  }, kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref.avg_ms, iterations, ref.min_ms);

#ifdef KH_TEST_SOLUTION
  // Golden of each set = the CUTLASS reference result on that set (computed once, outside the
  // timed region), as in TestCorrectness. Set 1's CUTLASS result is checked against the naive
  // FP64 kernel first, exactly like set 0's in TestCorrectness.
  size_t const count_C = size_t(ldc) * N, bytes_C = sizeof(double) * count_C;
  double *G[2], ref_scale[2];
  for (int s = 0; s < 2; ++s) {
    kh_cuda_ok(cudaMalloc(&G[s], bytes_C), "golden");
    kh_cuda_ok(CutlassStrmmNN(M, N, alpha, A_set[s], lda, B_set[s], ldb, G[s], ldc), "golden (CUTLASS)");
  }
  {
    double *naive;
    kh_cuda_ok(cudaMalloc(&naive, bytes_C), "golden check");
    kh_cuda_ok(ReferenceTrmm(M, N, alpha, A_set[1], lda, B_set[1], ldb, naive, ldc), "golden check (naive)");
    kh_cuda_ok(cudaDeviceSynchronize(), "golden check");
    if (!kh_trmm_check("CUTLASS reference vs naive (timing set 1)", G[1], naive,
                       kh_trmm_ref_scale(naive, count_C), count_C))
      kh_fail("CUTLASS reference incorrect vs naive on timing set 1.");
    cudaFree(naive);
  }
  for (int s = 0; s < 2; ++s) ref_scale[s] = kh_trmm_ref_scale(G[s], count_C);

  KhTiming sol = kh_bench("Kernel time", warmup, iterations, [&](int s) {
    restore_inputs(s);
    kh_cuda_ok(cudaMemsetAsync(C, 0xFF, bytes_C), "output poison");   // NaN: unwritten elements fail
  }, [&] {
    Dtrmm(M, N, alpha, A, lda, B, ldb, C, ldc);
  }, [&](int s) {
    return kh_trmm_check("Solution (timed call)", C, G[s], ref_scale[s], count_C);
  });
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol.avg_ms, iterations, sol.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  for (int s = 0; s < 2; ++s) cudaFree(G[s]);
#endif

  for (int s = 0; s < 2; ++s) {
    cudaFree(A_set[s]);
    cudaFree(B_set[s]);
  }
  cudaFree(C);
  cudaFree(B);
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
    int m, n;
  };

  bool explicit_size = (argc >= 3);

  std::vector<TestConfig> configs;
  double alpha = 1.1;   // non-trivial alpha so the alpha scaling is exercised
  int iterations = 20;

  if (explicit_size) {
    int M = 128, N = 128;
    std::stringstream(arg[1]) >> M;
    std::stringstream(arg[2]) >> N;
    if (argc > 3) std::stringstream(arg[3]) >> alpha;
    if (argc > 4) std::stringstream(arg[4]) >> iterations;
    configs.push_back({"custom", M, N});
  } else {
    configs = {
      {"small",   1024, 1024},
      {"medium",  4096, 4096},
      {"large",   4096, 8192},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d ===\n", cfg.label, cfg.m, cfg.n);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, alpha);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, alpha, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
