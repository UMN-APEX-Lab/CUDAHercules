/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
  \brief GETT (Generalized Tensor-Tensor contraction) on Hopper using CUTLASS 3.x API.

  For KH_TEST_SOLUTION: operates as a standard column-major GEMM.
  D = alpha * A * B + beta * C
  A column-major [M, K] (lda=M), B column-major [K, N] (ldb=K),
  C column-major [M, N] (ldc=M), D column-major [M, N] (ldd=M).
  Half-precision inputs (A,B,C), float output (D).
*/

#include "gett_kernel.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/reference/device/gett.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/print_error.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace cute;

//////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM kernel: A col-major, B col-major, C col-major, D col-major
/// Half inputs (A,B,C), float output (D). D = alpha * A * B + beta * C.
__global__ void NaiveGettRef_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half const *C, int ldc,
    float *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A col-major: A[i + k * lda]
      // B col-major: B[k + j * ldb]
      acc += __half2float(A[i + k * lda]) * __half2float(B[k + j * ldb]);
    }
    // C col-major: C[i + j * ldc]
    // D col-major: D[i + j * ldd]
    D[i + j * ldd] = alpha * acc + beta * __half2float(C[i + j * ldc]);
  }
}

/// Initialize half matrix with deterministic small values
__global__ void InitHalf51_kernel(__half *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float value = float(((idx + seed) * 16807 % 16) - 8) * 0.01f;
    matrix[idx] = __float2half(value);
  }
}

cudaError_t AllocHalf51(__half **m, int size, int seed) {
  cudaError_t r = cudaMalloc(reinterpret_cast<void**>(m), sizeof(__half) * size);
  if (r != cudaSuccess) return r;
  InitHalf51_kernel<<<(size + 255) / 256, 256>>>(*m, size, seed);
  return cudaGetLastError();
}

//////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Define multi-mode stride types for GETT
using RowModeStridesA51 = cute::Stride<cute::Int<1>, int64_t, int64_t, int64_t>;
using RedModeStridesA51 = cute::Stride<int64_t, int64_t, int64_t>;
using BatModeStridesA51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

using ColModeStridesB51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
using RedModeStridesB51 = cute::Stride<cute::Int<1>, int64_t, int64_t>;
using BatModeStridesB51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

using RowModeStridesC51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
using ColModeStridesC51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
using BatModeStridesC51 = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

using StrideA51 = cute::Stride<RowModeStridesA51, RedModeStridesA51, BatModeStridesA51>;
using StrideB51 = cute::Stride<ColModeStridesB51, RedModeStridesB51, BatModeStridesB51>;
using StrideC51 = cute::Stride<RowModeStridesC51, ColModeStridesC51, BatModeStridesC51>;
using StrideD51 = StrideC51;

using ElementA51 = cutlass::half_t;
using ElementB51 = cutlass::half_t;
using ElementC51 = cutlass::half_t;
using ElementD51 = float;
using ElementAcc51 = float;
using ElementEpi51 = float;

/// Helper to build GETT problem shape and strides for a simple column-major GEMM
static auto MakeGettProblem51(int M, int N, int K) {
  auto problem_shape = make_shape(
    make_shape(M, 1, 1, 1),    // M modes
    make_shape(N, 1, 1, 1),    // N modes
    make_shape(K, 1, 1),       // K modes
    make_shape(1, 1, 1, 1)     // L modes (no batching)
  );

  // A: column-major [M, K]
  StrideA51 stride_A = make_stride(
    make_stride(cute::Int<1>{}, int64_t(0), int64_t(0), int64_t(0)),
    make_stride(int64_t(M), int64_t(0), int64_t(0)),
    make_stride(int64_t(0), int64_t(0), int64_t(0), int64_t(0))
  );

  // B: column-major [K, N]
  StrideB51 stride_B = make_stride(
    make_stride(int64_t(K), int64_t(0), int64_t(0), int64_t(0)),
    make_stride(cute::Int<1>{}, int64_t(0), int64_t(0)),
    make_stride(int64_t(0), int64_t(0), int64_t(0), int64_t(0))
  );

  // C/D: column-major [M, N]
  StrideC51 stride_C = make_stride(
    make_stride(int64_t(1), int64_t(0), int64_t(0), int64_t(0)),
    make_stride(int64_t(M), int64_t(0), int64_t(0), int64_t(0)),
    make_stride(int64_t(0), int64_t(0), int64_t(0), int64_t(0))
  );
  StrideD51 stride_D = stride_C;

  return std::make_tuple(problem_shape, stride_A, stride_B, stride_C, stride_D);
}

/// Run CUTLASS GETT kernel, returning cudaSuccess or error
static cudaError_t CutlassGett51(
    int M, int N, int K, float alpha,
    cutlass::half_t const *A, cutlass::half_t const *B,
    float beta, cutlass::half_t const *C, float *D) {

  auto [problem_shape, stride_A, stride_B, stride_C, stride_D] = MakeGettProblem51(M, N, K);

  auto status = example::gett_kernel(
    problem_shape,
    A, stride_A,
    B, stride_B,
    ElementAcc51{},
    C, stride_C,
    D, stride_D,
    ElementEpi51(alpha), ElementEpi51(beta));

  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaSuccess;
}
#endif

//////////////////////////////////////////////////////////////////////////////

cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  int lda = M, ldb = K, ldc = M, ldd = M;
  int64_t total_A = int64_t(M) * K;
  int64_t total_B = int64_t(K) * N;
  int64_t total_CD = int64_t(M) * N;

  __half *A, *B, *C;
  float *D_ref, *D_cutlass;

  AllocHalf51(&A, total_A, 0);
  AllocHalf51(&B, total_B, 17);
  AllocHalf51(&C, total_CD, 42);
  cudaMalloc(&D_ref, total_CD * sizeof(float));
  cudaMemset(D_ref, 0, total_CD * sizeof(float));
  cudaMalloc(&D_cutlass, total_CD * sizeof(float));
  cudaMemset(D_cutlass, 0, total_CD * sizeof(float));

  // Run naive reference
  {
    dim3 bl(16, 16), gr((M + 15) / 16, (N + 15) / 16);
    NaiveGettRef_kernel<<<gr, bl>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_ref, ldd);
  }
  cudaDeviceSynchronize();

  bool cutlass_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (CutlassGett51(M, N, K, alpha,
      (cutlass::half_t const*)A, (cutlass::half_t const*)B,
      beta, (cutlass::half_t const*)C, D_cutlass) == cudaSuccess) {
    cutlass_ok = true;
  } else {
    cudaGetLastError(); // clear error
  }
#endif

#ifdef KH_TEST_SOLUTION
  float *D_sol;
  cudaMalloc(&D_sol, total_CD * sizeof(float));
  cudaMemset(D_sol, 0, total_CD * sizeof(float));

  cudaError_t sr = HopperGett(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_sol, ldd);
  if (sr != cudaSuccess) {
    std::cerr << "Solution failed: " << cudaGetErrorString(sr) << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return sr;
  }
  cudaDeviceSynchronize();

  float *D_check = cutlass_ok ? D_cutlass : D_ref;
  std::vector<float> hs(total_CD), hr(total_CD);
  cudaMemcpy(hs.data(), D_sol, total_CD * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), D_check, total_CD * sizeof(float), cudaMemcpyDeviceToHost);
  float md = 0;
  for (int64_t i = 0; i < total_CD; ++i) md = fmaxf(md, fabsf(hs[i] - hr[i]));
  if (md > 1.0f) {
    std::cerr << "Solution incorrect. Max diff: " << md << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }
  cudaFree(D_sol);
#endif

  cudaFree(D_ref); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
  return cudaSuccess;
}

//////////////////////////////////////////////////////////////////////////////

void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = M, ldb = K, ldc = M, ldd = M;
  int64_t total_A = int64_t(M) * K;
  int64_t total_B = int64_t(K) * N;
  int64_t total_CD = int64_t(M) * N;

  __half *A, *B, *C;
  float *D;
  AllocHalf51(&A, total_A, 0);
  AllocHalf51(&B, total_B, 17);
  AllocHalf51(&C, total_CD, 42);
  cudaMalloc(&D, total_CD * sizeof(float));
  cudaMemset(D, 0, total_CD * sizeof(float));

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  float ref_min = 1e30f;
  bool rok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (CutlassGett51(M, N, K, alpha,
      (cutlass::half_t const*)A, (cutlass::half_t const*)B,
      beta, (cutlass::half_t const*)C, D) == cudaSuccess) {
    cudaDeviceSynchronize();
    float rt = 0;
    for (int i = 0; i < iterations; ++i) {
      cudaEventRecord(t0);
      CutlassGett51(M, N, K, alpha,
        (cutlass::half_t const*)A, (cutlass::half_t const*)B,
        beta, (cutlass::half_t const*)C, D);
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      float ms;
      cudaEventElapsedTime(&ms, t0, t1);
      rt += ms;
      if (ms < ref_min) ref_min = ms;
    }
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n", rt / iterations, iterations, ref_min);
    rok = true;
  } else {
    cudaGetLastError();
  }
#endif

#ifdef KH_TEST_SOLUTION
  // Save reference output for correctness check
  float *D_ref_saved = nullptr;
  if (rok) {
    cudaMalloc(&D_ref_saved, total_CD * sizeof(float));
    cudaMemcpy(D_ref_saved, D, total_CD * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  cudaMemset(D, 0, total_CD * sizeof(float));
  HopperGett(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  cudaDeviceSynchronize();

  if (rok && D_ref_saved) {
    std::vector<float> h_sol(total_CD), h_ref(total_CD);
    cudaMemcpy(h_sol.data(), D, total_CD * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved, total_CD * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(D_ref_saved);
    float md = 0;
    for (int64_t i = 0; i < total_CD; ++i) md = fmaxf(md, fabsf(h_sol[i] - h_ref[i]));
    if (md > 1.0f) {
      fprintf(stderr, "Solution incorrect in Profile: max_diff=%.6f\n", md);
      std::cout << "Incorrect" << std::endl;
      cudaEventDestroy(t0); cudaEventDestroy(t1);
      cudaFree(D); cudaFree(C); cudaFree(B); cudaFree(A);
      exit(-1);
    }
  }

  for (int i = 0; i < 3; i++)
    HopperGett(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  cudaDeviceSynchronize();

  float st = 0, sm = 1e30f;
  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(t0);
    HopperGett(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    st += ms;
    if (ms < sm) sm = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n", st / iterations, iterations, sm);
  if (rok) fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min / sm);
#endif

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaFree(D);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
}

//////////////////////////////////////////////////////////////////////////////

int main(int argc, char const* argv[]) {

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 1.0f;
  int iterations = 20;

  if (argc >= 4) {
    int M = 128, N = 128, K = 128;
    std::stringstream(argv[1]) >> M;
    std::stringstream(argv[2]) >> N;
    std::stringstream(argv[3]) >> K;
    if (argc > 4) { std::stringstream ss(argv[4]); ss >> iterations; }
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

    if (TestCorrectness(cfg.m, cfg.n, cfg.k, alpha, beta) != cudaSuccess) {
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
