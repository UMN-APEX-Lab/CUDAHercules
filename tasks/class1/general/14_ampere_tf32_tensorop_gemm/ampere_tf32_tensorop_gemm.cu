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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// Data types (TF32: float in, float out, internally uses TF32 tensor cores)
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

// Layouts: A row-major, B column-major, C/D row-major
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// CUTLASS TF32 GEMM wrapper.
/// A row-major [M,K] (lda=K), B col-major [K,N] (ldb=K), C/D row-major [M,N] (ldc=N)
/// D = alpha * A * B + beta * C
cudaError_t CutlassTf32Gemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename Gemm::Arguments arguments{
    problem_size,
    {A, lda},
    {B, ldb},
    {C, ldc},
    {C, ldc},
    {alpha, beta},
    1};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Initialize a matrix with small integers.
/// Uses generalized layout: element (i,j) is at matrix[i * ldm + j].
__global__ void InitializeMatrix_kernel(
  float *matrix, int ldm, int rows, int columns, int seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < rows && j < columns) {
    int offset = i * ldm + j;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}

/// Init row-major matrix [rows, columns] with leading dim = columns
cudaError_t InitRowMajor(float *matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + 15) / 16, (columns + 15) / 16);
  InitializeMatrix_kernel<<<grid, block>>>(matrix, columns, rows, columns, seed);
  return cudaGetLastError();
}

/// Init column-major matrix linearly
cudaError_t InitColMajorLinear(float *matrix, int count, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((count + 15) / 16, 1);
  InitializeMatrix_kernel<<<grid, block>>>(matrix, 1, count, 1, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: A row-major [M,K], B col-major [K,N], C row-major [M,N]
__global__ void ReferenceGemm_kernel(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,   // row-major, lda = K
  float const *B, int ldb,   // col-major, ldb = K
  float beta,
  float *C, int ldc) {       // row-major, ldc = N

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[i * lda + k] * B[k + j * ldb];
    }
    C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
  }
}

cudaError_t ReferenceGemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS TF32 vs naive FP32, and (if KH_TEST_SOLUTION) solution vs naive.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  int lda = K, ldb = K, ldc = N;
  size_t sizeof_C = sizeof(float) * M * N;

  float *A, *B, *C_cutlass, *C_naive;

  cudaMalloc(&A, sizeof(float) * M * K);
  cudaMalloc(&B, sizeof(float) * K * N);
  cudaMalloc(&C_cutlass, sizeof_C);
  cudaMalloc(&C_naive, sizeof_C);

  // Initialize A row-major
  InitRowMajor(A, M, K, 0);
  // Initialize B col-major (linear fill)
  {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    InitializeMatrix_kernel<<<grid, block>>>(B, 1, K * N, 1, 17);
    cudaGetLastError();
  }
  cudaMemset(C_cutlass, 0, sizeof_C);
  cudaMemset(C_naive, 0, sizeof_C);
  cudaDeviceSynchronize();

  // Run CUTLASS TF32 reference
  result = CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS TF32 GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_naive); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run naive FP32 reference
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_naive, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_naive); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS TF32 vs naive FP32 (relaxed tolerance -- TF32 truncates mantissa)
  std::vector<float> host_cutlass(M * N);
  std::vector<float> host_naive(M * N);
  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_naive.data(), C_naive, sizeof_C, cudaMemcpyDeviceToHost);

  float max_rel_diff = 0;
  for (int idx = 0; idx < M * N; ++idx) {
    float diff = fabsf(host_cutlass[idx] - host_naive[idx]);
    float rel = diff / (fabsf(host_naive[idx]) + 1e-6f);
    max_rel_diff = fmaxf(max_rel_diff, rel);
  }
  if (max_rel_diff > 0.01f) {
    std::cerr << "CUTLASS TF32 reference incorrect vs naive. Max relative diff: " << max_rel_diff << std::endl;
    cudaFree(C_naive); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  float *C_solution;
  cudaMalloc(&C_solution, sizeof_C);
  cudaMemset(C_solution, 0, sizeof_C);

  // Run solution
  result = Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_naive); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs naive FP32 (relaxed tolerance for TF32-style implementations)
  std::vector<float> host_solution(M * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  float sol_max_rel_diff = 0;
  for (int idx = 0; idx < M * N; ++idx) {
    float diff = fabsf(host_solution[idx] - host_naive[idx]);
    float rel = diff / (fabsf(host_naive[idx]) + 1e-6f);
    sol_max_rel_diff = fmaxf(sol_max_rel_diff, rel);
  }
  if (sol_max_rel_diff > 0.01f) {
    std::cerr << "Solution incorrect. Max relative diff vs naive: " << sol_max_rel_diff << std::endl;
    cudaFree(C_solution); cudaFree(C_naive); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

  cudaFree(C_solution);
#endif

  cudaFree(C_naive);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = K, ldc = N;
  size_t sizeof_C = sizeof(float) * M * N;

  float *A, *B, *C;
  cudaMalloc(&A, sizeof(float) * M * K);
  cudaMalloc(&B, sizeof(float) * K * N);
  cudaMalloc(&C, sizeof_C);

  InitRowMajor(A, M, K, 0);
  {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    InitializeMatrix_kernel<<<grid, block>>>(B, 1, K * N, 1, 17);
  }
  cudaMemset(C, 0, sizeof_C);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
  cudaMemset(C, 0, sizeof_C);
  for (int i = 0; i < 3; i++) {
    Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
