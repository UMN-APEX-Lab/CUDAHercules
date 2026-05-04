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

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning( disable : 4503)

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t cutlass_strided_batched_sgemm(
  int m,
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float beta,
  int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda},
    batch_stride_A,
    {B, ldb},
    batch_stride_B,
    {C, ldc},
    batch_stride_C,
    {C, ldc},
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference for strided batched GEMM NN (column-major), CPU version.
template<typename T>
cudaError_t strided_batched_gemm_nn_reference(
  int m,
  int n,
  int k,
  T alpha,
  std::vector<T> const &A,
  int lda,
  long long int batch_stride_A,
  std::vector<T> const &B,
  int ldb,
  long long int batch_stride_B,
  std::vector<T> &C,
  int ldc,
  long long int batch_stride_C,
  T beta,
  int batch_count) {

  cudaError_t result = cudaSuccess;

  if (A.size() < size_t(lda * k * batch_count)) {
    std::cout << "the size of A is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (B.size() < size_t(ldb * n)) {
    std::cout << "the size of B is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (C.size() < size_t(ldc * n * batch_count)) {
    std::cout << "the size of C is too small" << std::endl;
    return cudaErrorInvalidValue;
  }

  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix, int ldm, int rows, int columns, int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;
    int const kk = 16807;
    int const mm = 16;
    float value = float(((offset + seed) * kk % mm) - mm / 2);
    matrix[offset] = value;
  }
}

/// Initialize a matrix stored with given leading dimension.
cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );
  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int m, int n, int k, int batch_count, float alpha, float beta) {
  cudaError_t result;

  // Column-major layout matching original example
  int lda = m;
  int ldb = k * batch_count;
  int ldc = m;

  int count_A = batch_count * lda * k;
  int count_B = ldb * n;
  int count_C = batch_count * ldc * n;

  long long int batch_stride_A = (long long int)(lda) * (long long int)(k);
  long long int batch_stride_B = (long long int)(k);
  long long int batch_stride_C = (long long int)(ldc) * (long long int)(n);

  size_t sizeof_A = count_A * sizeof(float);
  size_t sizeof_B = count_B * sizeof(float);
  size_t sizeof_C = count_C * sizeof(float);

  // Host data
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C, 1.0f);

  int const kRange = 8;
  for (int b = 0; b < batch_count; b++)
    for (int col = 0; col < k; col++)
      for (int row = 0; row < m; row++)
        host_A[row + col * lda + b * lda * k] =
          static_cast<float>((row + col * lda + b * lda * k) % kRange);

  for (int b = 0; b < batch_count; b++)
    for (int col = 0; col < n; col++)
      for (int row = 0; row < k; row++)
        host_B[row + col * ldb + b * k] =
          static_cast<float>(((n + k * ldb + batch_count * k)
            - (row + col * ldb + b * k)) % kRange);

  // CPU reference
  std::vector<float> ref_C(host_C);
  strided_batched_gemm_nn_reference<float>(
    m, n, k, alpha, host_A, lda, batch_stride_A, host_B, ldb, batch_stride_B,
    ref_C, ldc, batch_stride_C, beta, batch_count);

  // Device memory
  float *d_A, *d_B, *d_C_cutlass;
  result = cudaMalloc(&d_A, sizeof_A);
  if (result != cudaSuccess) return result;
  result = cudaMalloc(&d_B, sizeof_B);
  if (result != cudaSuccess) { cudaFree(d_A); return result; }
  result = cudaMalloc(&d_C_cutlass, sizeof_C);
  if (result != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return result; }

  cudaMemcpy(d_A, host_A.data(), sizeof_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, host_B.data(), sizeof_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_cutlass, host_C.data(), sizeof_C, cudaMemcpyHostToDevice);

  // Run CUTLASS reference
  result = cutlass_strided_batched_sgemm(
    m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
    d_C_cutlass, ldc, batch_stride_C, beta, batch_count);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS batched GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive
  std::vector<float> host_cutlass(count_C);
  cudaMemcpy(host_cutlass.data(), d_C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  for (int i = 0; i < count_C; ++i) {
    if (host_cutlass[i] != ref_C[i]) {
      std::cerr << "CUTLASS reference incorrect vs naive at index " << i
                << ": cutlass=" << host_cutlass[i] << " ref=" << ref_C[i] << std::endl;
      cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
      return cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output with same initial C
  float *d_C_solution;
  result = cudaMalloc(&d_C_solution, sizeof_C);
  if (result != cudaSuccess) {
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
    return result;
  }
  cudaMemcpy(d_C_solution, host_C.data(), sizeof_C, cudaMemcpyHostToDevice);

  // Run solution
  result = StridedBatchedSgemm(
    m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
    d_C_solution, ldc, batch_stride_C, beta, batch_count);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_C_solution); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> host_solution(count_C);
  cudaMemcpy(host_solution.data(), d_C_solution, sizeof_C, cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < count_C; ++i) {
    max_diff = fmaxf(max_diff, fabsf(host_solution[i] - host_cutlass[i]));
  }
  if (max_diff > 1e-3f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
    cudaFree(d_C_solution); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
    return cudaErrorUnknown;
  }

  cudaFree(d_C_solution);
#endif

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cutlass);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int m, int n, int k, int batch_count, float alpha, float beta, int iterations) {
  int lda = m;
  int ldb = k * batch_count;
  int ldc = m;

  int count_A = batch_count * lda * k;
  int count_B = ldb * n;
  int count_C = batch_count * ldc * n;

  long long int batch_stride_A = (long long int)(lda) * (long long int)(k);
  long long int batch_stride_B = (long long int)(k);
  long long int batch_stride_C = (long long int)(ldc) * (long long int)(n);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, count_A * sizeof(float));
  cudaMalloc(&d_B, count_B * sizeof(float));
  cudaMalloc(&d_C, count_C * sizeof(float));

  // Initialize matrices on device
  for (int b = 0; b < batch_count; b++) {
    InitializeMatrix(d_A + b * lda * k, lda, m, k, b);
    InitializeMatrix(d_C + b * ldc * n, ldc, m, n, 101 + b);
  }
  // Initialize B (shared across batches with stride k in B's storage)
  InitializeMatrix(d_B, ldb, ldb, n, 17);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    cutlass_strided_batched_sgemm(m, n, k, alpha, d_A, lda, batch_stride_A,
      d_B, ldb, batch_stride_B, d_C, ldc, batch_stride_C, beta, batch_count);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    cutlass_strided_batched_sgemm(m, n, k, alpha, d_A, lda, batch_stride_A,
      d_B, ldb, batch_stride_B, d_C, ldc, batch_stride_C, beta, batch_count);
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
  for (int i = 0; i < 3; i++) {
    StridedBatchedSgemm(m, n, k, alpha, d_A, lda, batch_stride_A,
      d_B, ldb, batch_stride_B, d_C, ldc, batch_stride_C, beta, batch_count);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    StridedBatchedSgemm(m, n, k, alpha, d_A, lda, batch_stride_A,
      d_B, ldb, batch_stride_B, d_C, ldc, batch_stride_C, beta, batch_count);
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
  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k, batch_count;
  };

  bool explicit_size = (argc >= 5);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 128, N = 128, K = 64, B = 4;
    std::stringstream(arg[1]) >> M;
    std::stringstream(arg[2]) >> N;
    std::stringstream(arg[3]) >> K;
    std::stringstream(arg[4]) >> B;
    if (argc > 5) std::stringstream(arg[5]) >> alpha;
    if (argc > 6) std::stringstream(arg[6]) >> beta;
    if (argc > 7) std::stringstream(arg[7]) >> iterations;
    configs.push_back({"custom", M, N, K, B});
  } else {
    configs = {
      {"small",   128,  128,   64,  4},
      {"medium",  512,  512,  256, 16},
      {"large",  2048, 2048, 1024,  4},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch_count);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, cfg.batch_count, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, cfg.batch_count, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
