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
  This example is almost the same as example 27 which uses 3xTF32 to run GEMM.  The only
  difference is that this example uses 3xtf32 on complex gemm.

  To enable this feature, the only change needs to make is to change OpMultiplyAddComplex
  to OpMultiplyAddComplexFastF32.
*/

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<64, 64, 16>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::complex<float>,
    1,
    cutlass::complex<float>,
    cutlass::complex<float>>;

// Number of pipelines you want to use
constexpr int NumStages = 3;
// Transform
constexpr cutlass::ComplexTransform TransformA = cutlass::ComplexTransform::kNone;
constexpr cutlass::ComplexTransform TransformB = cutlass::ComplexTransform::kNone;

// Gemm_3xTF32
using Gemm_3xTF32 = cutlass::gemm::device::GemmComplex<
                                              cutlass::complex<float>,
                                              LayoutInputA,
                                              cutlass::complex<float>,
                                              LayoutInputB,
                                              cutlass::complex<float>,
                                              LayoutOutput,
                                              cutlass::complex<float>,
                                              MMAOp,
                                              SmArch,
                                              ShapeMMAThreadBlock,
                                              ShapeMMAWarp,
                                              ShapeMMAOp,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              NumStages,
                                              TransformA,
                                              TransformB,
                                              cutlass::arch::OpMultiplyAddComplexFastF32>;

///////////////////////////////////////////////////////////////////////////////////////////////////

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

/// Run the CUTLASS 3xTF32 complex GEMM reference.
/// A is column-major complex, B is row-major complex, C/D are row-major complex.
/// Complex stored as interleaved (re,im) float pairs.
cudaError_t Cutlass3xTF32ComplexGemm(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const *A, int lda,
    float const *B, int ldb,
    float beta_real, float beta_imag,
    float const *C, int ldc,
    float *D, int ldd) {

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Interpret raw float pointers as complex<float> pointers
  auto *A_complex = reinterpret_cast<cutlass::complex<float> const*>(A);
  auto *B_complex = reinterpret_cast<cutlass::complex<float> const*>(B);
  auto *C_complex = reinterpret_cast<cutlass::complex<float> const*>(C);
  auto *D_complex = reinterpret_cast<cutlass::complex<float>*>(D);

  cutlass::complex<float> alpha(alpha_real, alpha_imag);
  cutlass::complex<float> beta(beta_real, beta_imag);

  typename Gemm_3xTF32::Arguments arguments{
      problem_size,
      {A_complex, lda},
      {B_complex, ldb},
      {C_complex, ldc},
      {D_complex, ldd},
      {alpha, beta},
      1  // split_k_slices
  };

  Gemm_3xTF32 gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  size_t workspace_size = Gemm_3xTF32::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference complex GEMM on host.
struct Complex { float re, im; };
Complex cmul(Complex a, Complex b) { return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}; }
Complex cadd(Complex a, Complex b) { return {a.re+b.re, a.im+b.im}; }

void reference_complex_gemm_host(
    int M, int N, int K, Complex alpha,
    std::vector<float> const &A, int lda,
    std::vector<float> const &B, int ldb,
    Complex beta,
    std::vector<float> const &C, int ldc,
    std::vector<float> &D, int ldd) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      Complex acc = {0, 0};
      for (int k = 0; k < K; ++k) {
        int ai = (i + k * lda) * 2;       // column-major A
        int bi = (k * ldb + j) * 2;       // row-major B
        Complex a = {A[ai], A[ai+1]};
        Complex b = {B[bi], B[bi+1]};
        acc = cadd(acc, cmul(a, b));
      }
      int ci = (i * ldc + j) * 2;  // row-major C
      int di = (i * ldd + j) * 2;  // row-major D
      Complex c = {C[ci], C[ci+1]};
      Complex result = cadd(cmul(alpha, acc), cmul(beta, c));
      D[di] = result.re;
      D[di+1] = result.im;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, int K,
    float alpha_real, float alpha_imag,
    float beta_real, float beta_imag) {

  int lda = M, ldb = N, ldc = N, ldd = N;

  int count_A = lda * K * 2;
  int count_B = K * ldb * 2;
  int count_C = M * ldc * 2;

  std::vector<float> h_A(count_A), h_B(count_B);
  std::vector<float> h_C(count_C, 0), h_D_cutlass(count_C, 0), h_ref(count_C, 0);

  for (int i = 0; i < count_A; ++i)
    h_A[i] = float((i * 16807 % 16) - 8) * 0.05f;
  for (int i = 0; i < count_B; ++i)
    h_B[i] = float((i * 48271 % 16) - 8) * 0.05f;

  // Run naive reference on host
  reference_complex_gemm_host(M, N, K, {alpha_real, alpha_imag}, h_A, lda, h_B, ldb,
    {beta_real, beta_imag}, h_C, ldc, h_ref, ldd);

  // Run CUTLASS reference on device
  float *d_A, *d_B, *d_C, *d_D;
  CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D, count_C * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, count_C * sizeof(float)));

  cudaError_t result = Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag,
    d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS complex GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    return result;
  }
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(h_D_cutlass.data(), d_D, count_C * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify CUTLASS vs naive (3xTF32 has some tolerance)
  float max_diff = 0;
  for (int i = 0; i < count_C; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_D_cutlass[i] - h_ref[i]));
  if (max_diff > 1e-1f) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  float *d_D_solution;
  CUDA_CHECK(cudaMalloc(&d_D_solution, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_D_solution, 0, count_C * sizeof(float)));

  // Run solution
  result = ComplexGemm(M, N, K, alpha_real, alpha_imag,
    d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D_solution, ldd);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_D_solution); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> h_D_solution(count_C);
  CUDA_CHECK(cudaMemcpy(h_D_solution.data(), d_D_solution, count_C * sizeof(float), cudaMemcpyDeviceToHost));

  float sol_max_diff = 0;
  for (int i = 0; i < count_C; ++i)
    sol_max_diff = fmaxf(sol_max_diff, fabsf(h_D_solution[i] - h_D_cutlass[i]));
  if (sol_max_diff > 1e-1f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
    cudaFree(d_D_solution); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    return cudaErrorUnknown;
  }

  cudaFree(d_D_solution);
#endif

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K,
    float alpha_real, float alpha_imag,
    float beta_real, float beta_imag, int iterations) {

  int lda = M, ldb = N, ldc = N, ldd = N;
  int count_A = lda * K * 2;
  int count_B = K * ldb * 2;
  int count_C = M * ldc * 2;

  float *d_A, *d_B, *d_C, *d_D;
  CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D, count_C * sizeof(float)));

  // Initialize
  std::vector<float> h_A(count_A), h_B(count_B);
  for (int i = 0; i < count_A; ++i) h_A[i] = float((i * 16807 % 16) - 8) * 0.05f;
  for (int i = 0; i < count_B; ++i) h_B[i] = float((i * 48271 % 16) - 8) * 0.05f;
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_D, 0, count_C * sizeof(float)));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++)
    Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
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
  cudaMemset(d_D, 0, count_C * sizeof(float));
  for (int i = 0; i < 3; i++)
    ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
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
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Check for SM >= 80
  bool notSupported = false;
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80." << std::endl;
    notSupported = true;
  }
  if (notSupported) return 0;

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha_real = 1.0f, alpha_imag = 0.0f;
  float beta_real = 0.0f, beta_imag = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 256, N = 256, K = 128;
    std::stringstream ss1(arg[1]); ss1 >> M;
    std::stringstream ss2(arg[2]); ss2 >> N;
    std::stringstream ss3(arg[3]); ss3 >> K;
    if (argc > 4) { std::stringstream ss(arg[4]); ss >> alpha_real; }
    if (argc > 5) { std::stringstream ss(arg[5]); ss >> alpha_imag; }
    if (argc > 6) { std::stringstream ss(arg[6]); ss >> beta_real; }
    if (argc > 7) { std::stringstream ss(arg[7]); ss >> beta_imag; }
    if (argc > 8) { std::stringstream ss(arg[8]); ss >> iterations; }
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

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k,
      alpha_real, alpha_imag, beta_real, beta_imag);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, alpha_real, alpha_imag,
              beta_real, beta_imag, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
