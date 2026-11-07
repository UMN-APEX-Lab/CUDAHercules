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

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int N, int K, double alpha, double beta, int iterations) {
  int lda = N, ldc = N;

  double *A, *C;
  AllocateMatrix(&A, lda, N, K, 0);
  AllocateMatrix(&C, ldc, N, N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassSsyrkNN(N, K, alpha, A, lda, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassSsyrkNN(N, K, alpha, A, lda, beta, C, ldc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ref_total_ms += ms;
    if (ms < ref_min_ms) ref_min_ms = ms;
  }
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref_total_ms / iterations, iterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
  // Warmup solution
  InitializeMatrix(C, ldc, N, N, 101);
  for (int i = 0; i < 3; i++) {
    Dsyrk(N, K, alpha, A, lda, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Dsyrk(N, K, alpha, A, lda, beta, C, ldc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sol_total_ms += ms;
    if (ms < sol_min_ms) sol_min_ms = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol_total_ms / iterations, iterations, sol_min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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
