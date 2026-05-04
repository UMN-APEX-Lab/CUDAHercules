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

// CUTLASS GEMM
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D
                              {alpha, beta}); // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix, int rows, int columns, int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = InitializeMatrix(*matrix, rows, columns, seed);
  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;
    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }
    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  int lda = M;
  int ldb = K;
  int ldc = M;
  size_t sizeof_C = sizeof(float) * ldc * N;

  float *A, *B, *C_cutlass, *C_reference;

  result = AllocateMatrix(&A, M, K, 0);
  if (result != cudaSuccess) return result;

  result = AllocateMatrix(&B, K, N, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  result = AllocateMatrix(&C_cutlass, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }

  result = AllocateMatrix(&C_reference, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(C_cutlass); return result; }

  // Both C matrices start with same values
  cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  // Run CUTLASS reference
  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run naive reference
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Verify CUTLASS vs naive
  std::vector<float> host_cutlass(ldc * N);
  std::vector<float> host_reference(ldc * N);

  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS reference incorrect vs naive." << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output (same initial C)
  float *C_solution;
  result = AllocateMatrix(&C_solution, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run solution
  result = SgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> host_solution(ldc * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < ldc * N; ++i) {
    max_diff = fmaxf(max_diff, fabsf(host_solution[i] - host_cutlass[i]));
  }
  if (max_diff > 1e-3f) {
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
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = M, ldb = K, ldc = M;

  float *A, *B, *C;
  AllocateMatrix(&A, M, K, 0);
  AllocateMatrix(&B, K, N, 17);
  AllocateMatrix(&C, M, N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
  InitializeMatrix(C, M, N, 101);
  for (int i = 0; i < 3; i++) {
    SgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    SgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  // Check for explicit size on command line
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
    configs = {
      {"small",   1024, 1024, 1024},
      {"medium",  4096, 4096, 4096},
      {"large",   8192, 8192, 8192},
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
