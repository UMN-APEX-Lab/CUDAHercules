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

/**
The example demonstrates how to reduce one of the operands of the GEMM along the k-dimension when
computing GEMM.  So the output also contains either a Mx1 or 1XN vector.  It only works with Ampere
16x8x16 FP16/BF16 tensor cores, though it is not difficult to apply to other Turing/Ampere tensor
core instructions.

Most of the reduction is done in gemm/warp level, see gemm/warp/mma_with_reduction_tensor_op.h
A few bit of reduction is done in the epilogue before storing the vector, see
epilogue/threadblock/epilogue_gemm_k_reduction.h
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_with_k_reduction.h"
#include "cutlass/gemm/kernel/default_gemm_with_k_reduction.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// The code section below describes datatype for input, output tensors and computation between
// elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
using ElementInputA = cutlass::bfloat16_t;         // Data type of elements in input tensor
using ElementInputB = cutlass::bfloat16_t;         // Data type of elements in input tensor
using ElementOutput = cutlass::bfloat16_t;         // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;
// Layout of the output vector
using LayoutGemmKReduction = cutlass::layout::PitchLinear;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

// Reduce A or B operand along the K dimension
constexpr bool ReduceKForA = true;

// Alignment of A operand
constexpr int AlignmentA = 8;

// Alignment of B operand
constexpr int AlignmentB = 8;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
    ElementAccumulator,                                   // Data type of accumulator
    ElementComputeEpilogue>;

using Gemm = typename cutlass::gemm::device::GemmWithKReduction<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  ReduceKForA,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  AlignmentA,
  AlignmentB,
  cutlass::arch::OpMultiplyAdd,
  cutlass::ComplexTransform::kNone,
  cutlass::ComplexTransform::kNone
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers (bf16, column-major).
__global__ void InitializeMatrix_kernel_bf16(
  cutlass::bfloat16_t *matrix, int rows, int columns, int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;  // column-major
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2) * 0.01f;
    matrix[offset] = cutlass::bfloat16_t(value);
  }
}

/// Kernel to initialize a row-major matrix with small integers (bf16).
__global__ void InitializeMatrix_kernel_bf16_row(
  cutlass::bfloat16_t *matrix, int rows, int columns, int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * columns + j;  // row-major
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2) * 0.01f;
    matrix[offset] = cutlass::bfloat16_t(value);
  }
}

cudaError_t InitializeMatrix_bf16(cutlass::bfloat16_t *matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  InitializeMatrix_kernel_bf16<<<grid, block>>>(matrix, rows, columns, seed);
  return cudaGetLastError();
}

cudaError_t InitializeMatrix_bf16_row(cutlass::bfloat16_t *matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  InitializeMatrix_kernel_bf16_row<<<grid, block>>>(matrix, rows, columns, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: A col-major, B row-major, D col-major
__global__ void ReferenceGemmReduction_kernel(
  int M, int N, int K,
  float alpha,
  cutlass::bfloat16_t const *A, int lda,
  cutlass::bfloat16_t const *B, int ldb,
  float beta,
  cutlass::bfloat16_t const *C, int ldc,
  cutlass::bfloat16_t *D, int ldd) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += float(A[i + k * lda]) * float(B[k * ldb + j]);
    }
    D[i + j * ldd] = cutlass::bfloat16_t(alpha * acc + beta * float(C[i + j * ldc]));
  }
}

/// Naive reduction: for each column k of A (col-major), sum over rows
__global__ void ReferenceReduction_kernel(
  int M, int K,
  cutlass::bfloat16_t const *A, int lda,
  float *reduction) {

  int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k < K) {
    float sum = 0;
    for (int m = 0; m < M; ++m) {
      sum += float(A[m + k * lda]);
    }
    reduction[k] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Run the CUTLASS GemmWithKReduction reference
cudaError_t CutlassGemmWithReduction(
  int M, int N, int K,
  float alpha,
  cutlass::bfloat16_t const *A, int lda,
  cutlass::bfloat16_t const *B, int ldb,
  float beta,
  cutlass::bfloat16_t const *C, int ldc,
  cutlass::bfloat16_t *D, int ldd,
  cutlass::bfloat16_t *reduction_buf) {

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  int reduce_vector_length = M;  // ReduceKForA => length is M

  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
  int batch_count = 1;

  typename Gemm::Arguments arguments(
    mode,
    problem_size,
    batch_count,
    {alpha, beta},
    A,
    B,
    C,
    D,
    reduction_buf,
    (int64_t)(M * K),
    (int64_t)(N * K),
    (int64_t)(M * N),
    (int64_t)(M * N),
    (int64_t)reduce_vector_length,
    lda,
    ldb,
    ldc,
    ldd,
    0  // reduction stride
  );

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  int lda = M;  // col-major
  int ldb = N;  // row-major (stride = N = number of columns)
  int ldc = M;  // col-major
  int ldd = M;

  size_t sizeof_A = sizeof(cutlass::bfloat16_t) * lda * K;
  size_t sizeof_B = sizeof(cutlass::bfloat16_t) * K * ldb;
  size_t sizeof_C = sizeof(cutlass::bfloat16_t) * ldc * N;
  size_t sizeof_D = sizeof(cutlass::bfloat16_t) * ldd * N;
  size_t sizeof_red = sizeof(float) * K;
  // The CUTLASS reduction output is bf16 with length M (ReduceKForA)
  size_t sizeof_red_cutlass = sizeof(cutlass::bfloat16_t) * M;

  cutlass::bfloat16_t *A, *B, *C_cutlass, *D_cutlass, *D_naive, *reduction_cutlass;
  float *reduction_naive;

  result = cudaMalloc((void**)&A, sizeof_A); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&B, sizeof_B); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&C_cutlass, sizeof_C); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&D_cutlass, sizeof_D); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&D_naive, sizeof_D); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&reduction_cutlass, sizeof_red_cutlass); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&reduction_naive, sizeof_red); if (result != cudaSuccess) return result;

  InitializeMatrix_bf16(A, M, K, 0);       // col-major A: M rows, K cols
  InitializeMatrix_bf16_row(B, K, N, 17);  // row-major B: K rows, N cols
  InitializeMatrix_bf16(C_cutlass, M, N, 101); // col-major C

  cudaMemset(D_cutlass, 0, sizeof_D);
  cudaMemset(D_naive, 0, sizeof_D);
  cudaMemset(reduction_cutlass, 0, sizeof_red_cutlass);
  cudaMemset(reduction_naive, 0, sizeof_red);

  // Run CUTLASS reference
  result = CutlassGemmWithReduction(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc,
                                     D_cutlass, ldd, reduction_cutlass);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GemmWithReduction failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(D_cutlass);
    cudaFree(D_naive); cudaFree(reduction_cutlass); cudaFree(reduction_naive);
    return result;
  }

  // Run naive reference GEMM
  {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    ReferenceGemmReduction_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, D_naive, ldd);
  }

  // Run naive reduction
  {
    dim3 block(256);
    dim3 grid((K + 255) / 256);
    ReferenceReduction_kernel<<<grid, block>>>(M, K, A, lda, reduction_naive);
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS GEMM output vs naive
  int count_D = ldd * N;
  std::vector<float> host_cutlass_D(count_D), host_naive_D(count_D);
  {
    std::vector<cutlass::bfloat16_t> tmp(count_D);
    cudaMemcpy(tmp.data(), D_cutlass, count_D * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count_D; i++) host_cutlass_D[i] = float(tmp[i]);
    cudaMemcpy(tmp.data(), D_naive, count_D * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count_D; i++) host_naive_D[i] = float(tmp[i]);
  }

  float max_diff = 0;
  for (int i = 0; i < count_D; i++) {
    max_diff = fmaxf(max_diff, fabsf(host_cutlass_D[i] - host_naive_D[i]));
  }
  if (max_diff > 1.0f) {
    std::cerr << "CUTLASS reference incorrect vs naive GEMM. Max diff: " << max_diff << std::endl;
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(D_cutlass);
    cudaFree(D_naive); cudaFree(reduction_cutlass); cudaFree(reduction_naive);
    return cudaErrorUnknown;
  }

  // Verify CUTLASS reduction vs naive reduction
  // CUTLASS outputs bf16 reduction of length M (sum along K for each row m)
  // But the harness wants: reduction[k] = sum_m A[m,k]  (length K)
  // Actually CUTLASS ReduceKForA=true reduces along K for the A operand,
  // producing a vector of length M. The harness wants column-reduction of A.
  // We only check the GEMM part for CUTLASS vs naive. Reduction check is done
  // between naive and the solution.
  std::vector<float> host_naive_red(K);
  cudaMemcpy(host_naive_red.data(), reduction_naive, K * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  cutlass::bfloat16_t *D_solution;
  float *reduction_solution;
  result = cudaMalloc((void**)&D_solution, sizeof_D); if (result != cudaSuccess) return result;
  result = cudaMalloc((void**)&reduction_solution, sizeof_red); if (result != cudaSuccess) return result;
  cudaMemset(D_solution, 0, sizeof_D);
  cudaMemset(reduction_solution, 0, sizeof_red);

  // Re-initialize C for solution (same seed)
  cutlass::bfloat16_t *C_solution;
  result = cudaMalloc((void**)&C_solution, sizeof_C); if (result != cudaSuccess) return result;
  InitializeMatrix_bf16(C_solution, M, N, 101);

  // Cast to __nv_bfloat16* for solution interface
  result = GemmWithReduction(M, N, K, alpha,
    reinterpret_cast<__nv_bfloat16 const*>(A), lda,
    reinterpret_cast<__nv_bfloat16 const*>(B), ldb,
    beta,
    reinterpret_cast<__nv_bfloat16 const*>(C_solution), ldc,
    reinterpret_cast<__nv_bfloat16*>(D_solution), ldd,
    reduction_solution);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(D_cutlass);
    cudaFree(D_naive); cudaFree(reduction_cutlass); cudaFree(reduction_naive);
    cudaFree(D_solution); cudaFree(reduction_solution); cudaFree(C_solution);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution GEMM vs naive
  {
    std::vector<cutlass::bfloat16_t> tmp(count_D);
    std::vector<float> host_sol_D(count_D);
    cudaMemcpy(tmp.data(), D_solution, count_D * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count_D; i++) host_sol_D[i] = float(tmp[i]);

    float sol_max_diff = 0;
    for (int i = 0; i < count_D; i++) {
      sol_max_diff = fmaxf(sol_max_diff, fabsf(host_sol_D[i] - host_naive_D[i]));
    }
    if (sol_max_diff > 2.0f) {
      std::cerr << "Solution GEMM incorrect. Max diff vs naive: " << sol_max_diff << std::endl;
      cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(D_cutlass);
      cudaFree(D_naive); cudaFree(reduction_cutlass); cudaFree(reduction_naive);
      cudaFree(D_solution); cudaFree(reduction_solution); cudaFree(C_solution);
      return cudaErrorUnknown;
    }
  }

  // Verify solution reduction vs naive
  {
    std::vector<float> host_sol_red(K);
    cudaMemcpy(host_sol_red.data(), reduction_solution, K * sizeof(float), cudaMemcpyDeviceToHost);

    float red_max_diff = 0;
    for (int i = 0; i < K; i++) {
      red_max_diff = fmaxf(red_max_diff, fabsf(host_sol_red[i] - host_naive_red[i]));
    }
    if (red_max_diff > 2.0f) {
      std::cerr << "Solution reduction incorrect. Max diff: " << red_max_diff << std::endl;
      cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(D_cutlass);
      cudaFree(D_naive); cudaFree(reduction_cutlass); cudaFree(reduction_naive);
      cudaFree(D_solution); cudaFree(reduction_solution); cudaFree(C_solution);
      return cudaErrorUnknown;
    }
  }

  cudaFree(D_solution);
  cudaFree(reduction_solution);
  cudaFree(C_solution);
#endif

  cudaFree(A);
  cudaFree(B);
  cudaFree(C_cutlass);
  cudaFree(D_cutlass);
  cudaFree(D_naive);
  cudaFree(reduction_cutlass);
  cudaFree(reduction_naive);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = M, ldb = N, ldc = M, ldd = M;

  size_t sizeof_A = sizeof(cutlass::bfloat16_t) * lda * K;
  size_t sizeof_B = sizeof(cutlass::bfloat16_t) * K * ldb;
  size_t sizeof_C = sizeof(cutlass::bfloat16_t) * ldc * N;
  size_t sizeof_D = sizeof(cutlass::bfloat16_t) * ldd * N;
  size_t sizeof_red_cutlass = sizeof(cutlass::bfloat16_t) * M;

  cutlass::bfloat16_t *A, *B, *C, *D, *reduction_cutlass;
  cudaMalloc((void**)&A, sizeof_A);
  cudaMalloc((void**)&B, sizeof_B);
  cudaMalloc((void**)&C, sizeof_C);
  cudaMalloc((void**)&D, sizeof_D);
  cudaMalloc((void**)&reduction_cutlass, sizeof_red_cutlass);

  InitializeMatrix_bf16(A, M, K, 0);
  InitializeMatrix_bf16_row(B, K, N, 17);
  InitializeMatrix_bf16(C, M, N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassGemmWithReduction(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, reduction_cutlass);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassGemmWithReduction(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, reduction_cutlass);
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
  // Allocate solution buffers
  size_t sizeof_red = sizeof(float) * K;
  cutlass::bfloat16_t *D_sol;
  float *reduction_sol;
  cudaMalloc((void**)&D_sol, sizeof_D);
  cudaMalloc((void**)&reduction_sol, sizeof_red);

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    GemmWithReduction(M, N, K, alpha,
      reinterpret_cast<__nv_bfloat16 const*>(A), lda,
      reinterpret_cast<__nv_bfloat16 const*>(B), ldb,
      beta,
      reinterpret_cast<__nv_bfloat16 const*>(C), ldc,
      reinterpret_cast<__nv_bfloat16*>(D_sol), ldd,
      reduction_sol);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    GemmWithReduction(M, N, K, alpha,
      reinterpret_cast<__nv_bfloat16 const*>(A), lda,
      reinterpret_cast<__nv_bfloat16 const*>(B), ldb,
      beta,
      reinterpret_cast<__nv_bfloat16 const*>(C), ldc,
      reinterpret_cast<__nv_bfloat16*>(D_sol), ldd,
      reduction_sol);
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

  cudaFree(D_sol);
  cudaFree(reduction_sol);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);
  cudaFree(reduction_cutlass);
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
