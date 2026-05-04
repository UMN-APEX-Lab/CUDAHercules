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

/// Kernel to initialize a matrix with small integers.
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
    int offset = i + j * ldm;

    int const k = 16807;
    int const m = 16;
    double value = double(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
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

/// Allocates device memory for a matrix then fills with arbitrary small integers.
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

  // Verify CUTLASS vs naive
  std::vector<double> host_cutlass(ldc * N);
  std::vector<double> host_reference(ldc * N);

  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS reference incorrect vs naive." << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  double *C_solution;
  cudaMalloc(&C_solution, sizeof_C);
  cudaMemset(C_solution, 0, sizeof_C);

  // Run solution
  result = Dtrmm(M, N, alpha, A, lda, B, ldb, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<double> host_solution(ldc * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  double max_diff = 0;
  for (int i = 0; i < ldc * N; ++i) {
    max_diff = fmax(max_diff, fabs(host_solution[i] - host_cutlass[i]));
  }
  if (max_diff > 1e-3) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
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

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, double alpha, int iterations) {
  int lda = M, ldb = M, ldc = M;

  double *A, *B, *C;
  AllocateMatrix(&A, lda, M, M, 0, cutlass::FillMode::kLower);
  AllocateMatrix(&B, ldb, M, N, 17);
  AllocateMatrix(&C, ldc, M, N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassStrmmNN(M, N, alpha, A, lda, B, ldb, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassStrmmNN(M, N, alpha, A, lda, B, ldb, C, ldc);
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
  cudaMemset(C, 0, sizeof(double) * ldc * N);
  for (int i = 0; i < 3; i++) {
    Dtrmm(M, N, alpha, A, lda, B, ldb, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Dtrmm(M, N, alpha, A, lda, B, ldb, C, ldc);
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
  double alpha = 1.0;
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
