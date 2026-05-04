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
#include <ctime>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

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

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Define the epilogue operation as LinearCombination.
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

// Number of pipelines you want to use
constexpr int NumStages = 5;

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
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
                                                  NumStages,
                                                  8,     /*alignmentA*/
                                                  8,     /*alignmentB*/
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  false, /*GatherA*/
                                                  true,  /*GatherB*/
                                                  true   /*ScatterD*/
                                                 >;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Reference gather-scatter GEMM computed on host (float precision).
/// Row-major layout for all matrices.
/// D[scatter_D[i], j] = alpha * sum_k A[gather_A[i], k] * B[k, j] + beta * C[gather_C[i], j]
void CutlassGatherScatterGemmHost(
    int M, int N, int K, float alpha,
    std::vector<float> const &fA, int lda, std::vector<int> const &gather_A,
    std::vector<float> const &fB, int ldb,
    float beta,
    std::vector<float> const &fC, int ldc, std::vector<int> const &gather_C,
    std::vector<float> &fD, int ldd, std::vector<int> const &scatter_D) {

  for (int i = 0; i < M; ++i) {
    int a_row = gather_A[i];
    int c_row = gather_C[i];
    int d_row = scatter_D[i];
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += fA[a_row * lda + k] * fB[k * ldb + j];
      fD[d_row * ldd + j] = alpha * acc + beta * fC[c_row * ldc + j];
    }
  }
}

/// Wrapper that copies data from device, runs host reference, and writes result back.
cudaError_t CutlassGatherScatterGemmRef(
    int M, int N, int K, float alpha,
    __half const *dA, int lda, int const *d_gather_A,
    __half const *dB, int ldb,
    float beta,
    float const *dC, int ldc, int const *d_gather_C,
    float *dD, int ldd, int const *d_scatter_D,
    int A_rows) {

  // Copy indices
  std::vector<int> gather_A(M), gather_C(M), scatter_D(M);
  cudaMemcpy(gather_A.data(), d_gather_A, M * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(gather_C.data(), d_gather_C, M * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(scatter_D.data(), d_scatter_D, M * sizeof(int), cudaMemcpyDeviceToHost);

  int D_rows = 0;
  for (int i = 0; i < M; ++i) D_rows = std::max(D_rows, scatter_D[i] + 1);
  int C_rows = 0;
  for (int i = 0; i < M; ++i) C_rows = std::max(C_rows, gather_C[i] + 1);

  int count_A = A_rows * K, count_B = K * N;
  int count_C = C_rows * N, count_D = D_rows * N;

  // Copy half data
  std::vector<__half> hA(count_A), hB(count_B);
  cudaMemcpy(hA.data(), dA, count_A * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hB.data(), dB, count_B * sizeof(__half), cudaMemcpyDeviceToHost);

  // Convert half to float
  std::vector<float> fA(count_A), fB(count_B);
  for (int i = 0; i < count_A; ++i) fA[i] = __half2float(hA[i]);
  for (int i = 0; i < count_B; ++i) fB[i] = __half2float(hB[i]);

  // Copy C (float)
  std::vector<float> fC(count_C, 0);
  cudaMemcpy(fC.data(), dC, count_C * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute
  std::vector<float> fD(count_D, 0);
  CutlassGatherScatterGemmHost(M, N, K, alpha, fA, lda, gather_A, fB, ldb,
    beta, fC, ldc, gather_C, fD, ldd, scatter_D);

  // Copy result back
  cudaMemcpy(dD, fD.data(), count_D * sizeof(float), cudaMemcpyHostToDevice);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness
cudaError_t TestCorrectness(int M, int N, int K, int A_rows, float alpha, float beta) {
  int lda = K, ldb = N, ldc = N, ldd = N;

  // Create gather/scatter indices
  std::vector<int> gather_A(M), gather_C(M), scatter_D(M);
  for (int i = 0; i < M; ++i) {
    gather_A[i] = (i * 2) % A_rows;
    gather_C[i] = i;
    scatter_D[i] = i;
  }

  int count_A = A_rows * K, count_B = K * N, count_C = M * N, count_D = M * N;

  std::vector<float> fA(count_A), fB(count_B), fC(count_C, 0), fD_naive(count_D, 0);
  for (int i = 0; i < count_A; ++i) fA[i] = float((i * 16807 % 16) - 8) * 0.01f;
  for (int i = 0; i < count_B; ++i) fB[i] = float((i * 48271 % 16) - 8) * 0.01f;

  // Convert through half to match what the GPU kernel sees
  std::vector<float> fA_h(count_A), fB_h(count_B);
  for (int i = 0; i < count_A; ++i) fA_h[i] = __half2float(__float2half(fA[i]));
  for (int i = 0; i < count_B; ++i) fB_h[i] = __half2float(__float2half(fB[i]));

  // Naive reference on host (using half-converted data)
  CutlassGatherScatterGemmHost(M, N, K, alpha, fA_h, lda, gather_A, fB_h, ldb,
    beta, fC, ldc, gather_C, fD_naive, ldd, scatter_D);

  // Allocate device memory
  std::vector<__half> hA(count_A), hB(count_B);
  for (int i = 0; i < count_A; ++i) hA[i] = __float2half(fA[i]);
  for (int i = 0; i < count_B; ++i) hB[i] = __float2half(fB[i]);

  __half *dA, *dB; float *dC, *dD_ref; int *dGA, *dGC, *dSD;
  CUDA_CHECK(cudaMalloc(&dA, count_A * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, count_B * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD_ref, count_D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dGA, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dGC, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dSD, M * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), count_A * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), count_B * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(dD_ref, 0, count_D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dGA, gather_A.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dGC, gather_C.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSD, scatter_D.data(), M * sizeof(int), cudaMemcpyHostToDevice));

  // Run reference (host-based computation stored on device)
  cudaError_t result = CutlassGatherScatterGemmRef(M, N, K, alpha,
    dA, lda, dGA, dB, ldb, beta, dC, ldc, dGC, dD_ref, ldd, dSD, A_rows);
  if (result != cudaSuccess) {
    std::cerr << "Reference gather-scatter GEMM failed." << std::endl;
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD_ref);
    cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
    return result;
  }

  // Verify reference vs naive (should be identical since both compute in float on host)
  std::vector<float> h_D_ref(count_D);
  CUDA_CHECK(cudaMemcpy(h_D_ref.data(), dD_ref, count_D * sizeof(float), cudaMemcpyDeviceToHost));

  float max_diff = 0;
  for (int i = 0; i < count_D; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_D_ref[i] - fD_naive[i]));
  if (max_diff > 1e-3f) {
    std::cerr << "Reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD_ref);
    cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate and run solution
  float *dD_solution;
  CUDA_CHECK(cudaMalloc(&dD_solution, count_D * sizeof(float)));
  CUDA_CHECK(cudaMemset(dD_solution, 0, count_D * sizeof(float)));

  result = GatherScatterGemm(M, N, K, alpha, dA, lda, dGA, dB, ldb,
    beta, dC, ldc, dGC, dD_solution, ldd, dSD);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(dD_solution); cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD_ref);
    cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs reference
  std::vector<float> h_D_solution(count_D);
  CUDA_CHECK(cudaMemcpy(h_D_solution.data(), dD_solution, count_D * sizeof(float), cudaMemcpyDeviceToHost));

  float sol_max_diff = 0;
  for (int i = 0; i < count_D; ++i)
    sol_max_diff = fmaxf(sol_max_diff, fabsf(h_D_solution[i] - h_D_ref[i]));
  if (sol_max_diff > 1.0f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
    cudaFree(dD_solution); cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD_ref);
    cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
    return cudaErrorUnknown;
  }

  cudaFree(dD_solution);
#endif

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD_ref);
  cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, int A_rows, float alpha, float beta, int iterations) {
  int lda = K, ldb = N, ldc = N, ldd = N;

  std::vector<int> gather_A(M), gather_C(M), scatter_D(M);
  for (int i = 0; i < M; ++i) {
    gather_A[i] = (i * 2) % A_rows;
    gather_C[i] = i;
    scatter_D[i] = i;
  }

  int count_A = A_rows * K, count_B = K * N, count_C = M * N, count_D = M * N;

  std::vector<float> fA(count_A), fB(count_B);
  for (int i = 0; i < count_A; ++i) fA[i] = float((i * 16807 % 16) - 8) * 0.01f;
  for (int i = 0; i < count_B; ++i) fB[i] = float((i * 48271 % 16) - 8) * 0.01f;

  std::vector<__half> hA(count_A), hB(count_B);
  for (int i = 0; i < count_A; ++i) hA[i] = __float2half(fA[i]);
  for (int i = 0; i < count_B; ++i) hB[i] = __float2half(fB[i]);

  __half *dA, *dB; float *dC, *dD; int *dGA, *dGC, *dSD;
  CUDA_CHECK(cudaMalloc(&dA, count_A * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, count_B * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, count_D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dGA, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dGC, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dSD, M * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), count_A * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), count_B * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(dD, 0, count_D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dGA, gather_A.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dGC, gather_C.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSD, scatter_D.data(), M * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++)
    CutlassGatherScatterGemmRef(M, N, K, alpha, dA, lda, dGA, dB, ldb,
      beta, dC, ldc, dGC, dD, ldd, dSD, A_rows);
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(dD, 0, count_D * sizeof(float));
    cudaEventRecord(start);
    CutlassGatherScatterGemmRef(M, N, K, alpha, dA, lda, dGA, dB, ldb,
      beta, dC, ldc, dGC, dD, ldd, dSD, A_rows);
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
  cudaMemset(dD, 0, count_D * sizeof(float));
  for (int i = 0; i < 3; i++)
    GatherScatterGemm(M, N, K, alpha, dA, lda, dGA, dB, ldb,
      beta, dC, ldc, dGC, dD, ldd, dSD);
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(dD, 0, count_D * sizeof(float));
    cudaEventRecord(start);
    GatherScatterGemm(M, N, K, alpha, dA, lda, dGA, dB, ldb,
      beta, dC, ldc, dGC, dD, ldd, dSD);
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
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  cudaFree(dGA); cudaFree(dGC); cudaFree(dSD);
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
    int m, n, k, a_rows;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 248, N = 256, K = 256, A_rows = 512;
    std::stringstream ss1(arg[1]); ss1 >> M;
    std::stringstream ss2(arg[2]); ss2 >> N;
    std::stringstream ss3(arg[3]); ss3 >> K;
    if (argc > 4) { std::stringstream ss(arg[4]); ss >> A_rows; }
    if (argc > 5) { std::stringstream ss(arg[5]); ss >> alpha; }
    if (argc > 6) { std::stringstream ss(arg[6]); ss >> beta; }
    if (argc > 7) { std::stringstream ss(arg[7]); ss >> iterations; }
    configs.push_back({"custom", M, N, K, A_rows});
  } else {
    configs = {
      {"small",   1024, 1024, 1024, 2048},
      {"medium",  4096, 4096, 4096, 8192},
      {"large",   8192, 8192, 8192, 16384},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, A_rows=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.a_rows);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, cfg.a_rows, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, cfg.a_rows, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
