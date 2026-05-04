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
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// Quaternion GEMM types — matching original example 21 layouts:
// A: RowMajor, B: ColumnMajor, C: RowMajor
using precision = float;
using Element = cutlass::Quaternion<float>;
using ElementComputeEpilogue = Element;
using ElementAccumulator = Element;
using ElementInputA = Element;
using ElementInputB = Element;
using ElementOutput = Element;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm50;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 4>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 16, 4>;
using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;

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

/// CUTLASS Quaternion GEMM wrapper.
/// A: row-major (M x K), B: column-major (K x N), C: row-major (M x N).
/// Each quaternion = 4 consecutive floats in CUTLASS order: (x, y, z, w).
/// A: quaternion (i,k) at A[(i * lda + k) * 4], lda = K
/// B: quaternion (k,j) at B[(k + j * ldb) * 4], ldb = K
/// C: quaternion (i,j) at C[(i * ldc + j) * 4], ldc = N
cudaError_t CutlassQuaternionGemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  Element const *qA = reinterpret_cast<Element const *>(A);
  Element const *qB = reinterpret_cast<Element const *>(B);
  Element *qC = reinterpret_cast<Element *>(C);

  Element q_alpha(alpha);
  Element q_beta(beta);

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename Gemm::Arguments arguments{
    problem_size,
    {qA, lda},     // A: row-major, stride = lda = K
    {qB, ldb},     // B: col-major, stride = ldb = K
    {qC, ldc},     // C: row-major, stride = ldc = N
    {qC, ldc},     // D: row-major, stride = ldc = N (in-place)
    {q_alpha, q_beta},
    1};

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  status = gemm_op.initialize(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix of quaternions with small values.
__global__ void InitializeQuaternionMatrix_kernel(
  float *matrix, int total_floats, int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < total_floats) {
    int const k = 16807;
    int const m = 16;
    float value = float(((i + seed) * k % m) - m / 2) * 0.1f;
    matrix[i] = value;
  }
}

cudaError_t InitializeQuaternionMatrix(float *matrix, int total_floats, int seed = 0) {
  int block = 256;
  int grid = (total_floats + block - 1) / block;
  InitializeQuaternionMatrix_kernel<<<grid, block>>>(matrix, total_floats, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t AllocateQuaternionMatrix(float **matrix, int total_floats, int seed = 0) {
  cudaError_t result;
  size_t bytes = sizeof(float) * total_floats;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), bytes);
  if (result != cudaSuccess) return result;

  result = cudaMemset(*matrix, 0, bytes);
  if (result != cudaSuccess) return result;

  result = InitializeQuaternionMatrix(*matrix, total_floats, seed);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference: GPU quaternion GEMM.
/// A: row-major, B: column-major, C: row-major.
/// Each quaternion = 4 floats in order (x, y, z, w).
/// A: quaternion (i,k) at A[(i * lda + k) * 4], lda = K
/// B: quaternion (k,j) at B[(k + j * ldb) * 4], ldb = K
/// C: quaternion (i,j) at C[(i * ldc + j) * 4], ldc = N
__global__ void ReferenceQuaternionGemm_kernel(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc_x = 0, acc_y = 0, acc_z = 0, acc_w = 0;
    for (int kk = 0; kk < K; ++kk) {
      int a_off = (i * lda + kk) * 4;  // row-major
      int b_off = (kk + j * ldb) * 4;  // col-major
      float ax = A[a_off], ay = A[a_off+1], az = A[a_off+2], aw = A[a_off+3];
      float bx = B[b_off], by = B[b_off+1], bz = B[b_off+2], bw = B[b_off+3];
      // Hamilton product (matching CUTLASS convention)
      acc_x += aw*bx + bw*ax + ay*bz - az*by;
      acc_y += aw*by + bw*ay + az*bx - ax*bz;
      acc_z += aw*bz + bw*az + ax*by - ay*bx;
      acc_w += aw*bw - ax*bx - ay*by - az*bz;
    }
    int c_off = (i * ldc + j) * 4;  // row-major
    float cx = C[c_off], cy = C[c_off+1], cz = C[c_off+2], cw = C[c_off+3];
    C[c_off]   = alpha * acc_x + beta * cx;
    C[c_off+1] = alpha * acc_y + beta * cy;
    C[c_off+2] = alpha * acc_z + beta * cz;
    C[c_off+3] = alpha * acc_w + beta * cw;
  }
}

cudaError_t ReferenceQuaternionGemm(
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

  ReferenceQuaternionGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  // Row-major A (M x K): lda = K;  Col-major B (K x N): ldb = K;  Row-major C (M x N): ldc = N
  int lda = K, ldb = K, ldc = N;
  int floats_A = M * K * 4;
  int floats_B = K * N * 4;
  int floats_C = M * N * 4;
  size_t sizeof_C = sizeof(float) * floats_C;

  float *A, *B, *C_cutlass, *C_reference;

  result = AllocateQuaternionMatrix(&A, floats_A, 0);
  if (result != cudaSuccess) return result;

  result = AllocateQuaternionMatrix(&B, floats_B, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  result = AllocateQuaternionMatrix(&C_cutlass, floats_C, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }

  result = AllocateQuaternionMatrix(&C_reference, floats_C, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(C_cutlass); return result; }

  // Both C matrices start with same values
  cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  // Run CUTLASS reference
  result = CutlassQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  result = ReferenceQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive
  std::vector<float> host_cutlass(floats_C);
  std::vector<float> host_reference(floats_C);

  cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  // Use relative tolerance that scales with K
  float cutlass_naive_tol = fmaxf(1.0f, K * 1e-3f);
  float max_diff = 0;
  for (int i = 0; i < floats_C; ++i) {
    max_diff = fmaxf(max_diff, fabsf(host_cutlass[i] - host_reference[i]));
  }
  if (max_diff > cutlass_naive_tol) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff
              << " (tolerance: " << cutlass_naive_tol << ")" << std::endl;
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output (same initial C)
  float *C_solution;
  result = AllocateQuaternionMatrix(&C_solution, floats_C, 101);
  if (result != cudaSuccess) {
    cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }

  // Run solution
  result = QuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_solution, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> host_solution(floats_C);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  float sol_max_diff = 0;
  for (int i = 0; i < floats_C; ++i) {
    sol_max_diff = fmaxf(sol_max_diff, fabsf(host_solution[i] - host_cutlass[i]));
  }
  if (sol_max_diff > cutlass_naive_tol) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
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
  int lda = K, ldb = K, ldc = N;
  int floats_A = M * K * 4;
  int floats_B = K * N * 4;
  int floats_C = M * N * 4;

  float *A, *B, *C;
  AllocateQuaternionMatrix(&A, floats_A, 0);
  AllocateQuaternionMatrix(&B, floats_B, 17);
  AllocateQuaternionMatrix(&C, floats_C, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassQuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
  InitializeQuaternionMatrix(C, floats_C, 101);
  for (int i = 0; i < 3; i++) {
    QuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    QuaternionGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
