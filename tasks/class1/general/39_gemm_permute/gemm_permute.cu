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

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>

#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS type definitions for row-major half-precision GEMM
using RowMajor = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,   RowMajor,    // A
    cutlass::half_t,   RowMajor,    // B
    cutlass::half_t,   RowMajor,    // C/D
    float,                          // ElementAccumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Run CUTLASS GEMM: D = alpha * A * B + beta * D (row-major, half precision)
cudaError_t CutlassHgemm(
    int M, int N, int K,
    float alpha,
    cutlass::half_t const *A, int lda,
    cutlass::half_t const *B, int ldb,
    float beta,
    cutlass::half_t *D, int ldd) {

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args(
      {M, N, K},
      {A, lda},
      {B, ldb},
      {D, ldd},
      {D, ldd},
      {alpha, beta}
  );

  cutlass::Status status = gemm_operator(args);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a half-precision matrix with small integers.
__global__ void InitializeMatrix_kernel(
    __half *matrix, int rows, int columns, int seed = 0) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < rows && j < columns) {
    int offset = i * columns + j;  // row-major
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2) * 0.01f;
    matrix[offset] = __float2half(value);
  }
}

cudaError_t InitializeMatrix(__half *matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);
  return cudaGetLastError();
}

cudaError_t AllocateMatrix(__half **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;
  size_t sizeof_matrix = sizeof(__half) * rows * columns;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sizeof_matrix);
  if (result != cudaSuccess) return result;
  result = InitializeMatrix(*matrix, rows, columns, seed);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: D = alpha * A * B + beta * D, row-major, half precision
__global__ void ReferenceGemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[i * lda + k]) * __half2float(B[k * ldb + j]);
    }
    D[i * ldd + j] = __float2half(alpha * acc + beta * __half2float(D[i * ldd + j]));
  }
}

cudaError_t ReferenceGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS ref vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  int lda = K, ldb = N, ldd = N;  // row-major
  size_t sizeof_D = sizeof(__half) * M * N;

  __half *A, *B, *D_cutlass, *D_reference;

  result = AllocateMatrix(&A, M, K, 0);
  if (result != cudaSuccess) return result;
  result = AllocateMatrix(&B, K, N, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }
  result = AllocateMatrix(&D_cutlass, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }
  result = AllocateMatrix(&D_reference, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(D_cutlass); return result; }

  // Both D matrices start with same values
  cudaMemcpy(D_reference, D_cutlass, sizeof_D, cudaMemcpyDeviceToDevice);

  // Run CUTLASS reference
  result = CutlassHgemm(M, N, K, alpha,
      reinterpret_cast<cutlass::half_t const*>(A), lda,
      reinterpret_cast<cutlass::half_t const*>(B), ldb,
      beta,
      reinterpret_cast<cutlass::half_t*>(D_cutlass), ldd);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run naive reference
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, D_reference, ldd);
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive
  std::vector<__half> host_cutlass(M * N);
  std::vector<__half> host_reference(M * N);
  cudaMemcpy(host_cutlass.data(), D_cutlass, sizeof_D, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), D_reference, sizeof_D, cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < M * N; ++i) {
    max_diff = fmaxf(max_diff,
        fabsf(__half2float(host_cutlass[i]) - __half2float(host_reference[i])));
  }
  if (max_diff > 1.0f) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  {
    __half *D_solution;
    result = AllocateMatrix(&D_solution, M, N, 101);
    if (result != cudaSuccess) {
      cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
      return result;
    }

    result = GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D_solution, ldd);
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
      cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
      return result;
    }
    cudaDeviceSynchronize();

    std::vector<__half> host_solution(M * N);
    cudaMemcpy(host_solution.data(), D_solution, sizeof_D, cudaMemcpyDeviceToHost);

    float sol_max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
      sol_max_diff = fmaxf(sol_max_diff,
          fabsf(__half2float(host_solution[i]) - __half2float(host_cutlass[i])));
    }
    if (sol_max_diff > 1.0f) {
      std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
      cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
      return cudaErrorUnknown;
    }

    cudaFree(D_solution);
  }
#endif

  cudaFree(D_reference);
  cudaFree(D_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = N, ldd = N;

  __half *A, *B, *D;
  AllocateMatrix(&A, M, K, 0);
  AllocateMatrix(&B, K, N, 17);
  AllocateMatrix(&D, M, N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassHgemm(M, N, K, alpha,
        reinterpret_cast<cutlass::half_t const*>(A), lda,
        reinterpret_cast<cutlass::half_t const*>(B), ldb,
        beta,
        reinterpret_cast<cutlass::half_t*>(D), ldd);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassHgemm(M, N, K, alpha,
        reinterpret_cast<cutlass::half_t const*>(A), lda,
        reinterpret_cast<cutlass::half_t const*>(B), ldb,
        beta,
        reinterpret_cast<cutlass::half_t*>(D), ldd);
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
  InitializeMatrix(D, M, N, 101);
  for (int i = 0; i < 3; i++) {
    GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
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
