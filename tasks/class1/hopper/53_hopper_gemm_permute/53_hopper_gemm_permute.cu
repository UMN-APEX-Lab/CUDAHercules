/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Hopper GEMM+permute example.

    For KH_TEST_SOLUTION mode, this tests a strided batched GEMM (no permutation fused).
    A row-major [batch, M, K] (lda = K), B row-major [batch, K, N] (ldb = N),
    D row-major [batch, M, N] (ldd = N).
    batch_stride_A = M*K, batch_stride_B = K*N, batch_stride_D = M*N.
    Half-precision inputs/outputs, float accumulation.

    D = alpha * A * B + beta * D
*/

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "helper.h"
#include "permute_kernel.cuh"
#include "permute_traits.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

///////////////////////////////////////////////////////////////////////////////////////////////////
// Naive reference kernel for batched row-major GEMM

__global__ void InitHalf53_kernel(__half *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float value = float(((idx + seed) * 16807 % 16) - 8) * 0.01f;
    matrix[idx] = __float2half(value);
  }
}

cudaError_t AllocHalf53(__half **m, int size, int seed) {
  cudaError_t r = cudaMalloc(reinterpret_cast<void**>(m), sizeof(__half) * size);
  if (r != cudaSuccess) return r;
  InitHalf53_kernel<<<(size + 255) / 256, 256>>>(*m, size, seed);
  return cudaGetLastError();
}

/// Naive batched GEMM: A row-major, B row-major, D row-major
/// D = alpha * A * B + beta * D
__global__ void NaiveBatchedGemmRef_kernel(
    int M, int N, int K, int batch, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half *D, int ldd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (i >= M || j >= N || b >= batch) return;

  int64_t offA = int64_t(b) * M * K;
  int64_t offB = int64_t(b) * K * N;
  int64_t offD = int64_t(b) * M * N;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += __half2float(A[offA + i * lda + k]) * __half2float(B[offB + k * ldb + j]);
  }
  float old_val = beta * __half2float(D[offD + i * ldd + j]);
  D[offD + i * ldd + j] = __float2half(alpha * acc + old_val);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

namespace example {

using ElementA53 = cutlass::half_t;
using ElementB53 = cutlass::half_t;
using ElementC53 = cutlass::half_t;
using ElementD53 = cutlass::half_t;
using ElementAcc53 = float;
using ElementEpi53 = float;

using LayoutA53 = cutlass::layout::RowMajor;
using LayoutB53 = cutlass::layout::RowMajor;
using LayoutC53 = cutlass::layout::RowMajor;
using LayoutD53 = cutlass::layout::RowMajor;

using TileShape53 = Shape<_128, _128, _64>;
using ClusterShape53 = Shape<_2, _2, _1>;

using ProblemShape53 = Shape<int, int, int, int>;

using StrideA53 = cutlass::gemm::TagToStrideA_t<LayoutA53>;
using StrideB53 = cutlass::gemm::TagToStrideB_t<LayoutB53>;
using StrideC53 = cutlass::gemm::TagToStrideC_t<LayoutC53>;
using StrideD53 = cutlass::gemm::TagToStrideC_t<LayoutD53>;

using CollectiveEpilogue53 = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  TileShape53, ClusterShape53, cutlass::epilogue::collective::EpilogueTileAuto,
  ElementAcc53, ElementEpi53,
  ElementC53, LayoutC53, 128 / cutlass::sizeof_bits<ElementC53>::value,
  ElementD53, LayoutD53, 128 / cutlass::sizeof_bits<ElementD53>::value,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop53 = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  ElementA53, LayoutA53, 128 / cutlass::sizeof_bits<ElementA53>::value,
  ElementB53, LayoutB53, 128 / cutlass::sizeof_bits<ElementB53>::value,
  ElementAcc53,
  TileShape53, ClusterShape53,
  cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue53::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel53 = cutlass::gemm::kernel::GemmUniversal<
  ProblemShape53, CollectiveMainloop53, CollectiveEpilogue53>;

using Gemm53 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel53>;

/// Run CUTLASS batched GEMM
static cudaError_t CutlassBatchedGemm53(
    int M, int N, int K, int batch, float alpha,
    cutlass::half_t const *A, cutlass::half_t const *B,
    float beta, cutlass::half_t *D) {

  auto shape_A = make_shape(M, K, batch);
  auto shape_B = make_shape(N, K, batch);
  auto shape_D = make_shape(M, N, batch);

  auto stride_A = cutlass::make_cute_packed_stride(StrideA53{}, shape_A);
  auto stride_B = cutlass::make_cute_packed_stride(StrideB53{}, shape_B);
  auto stride_C = cutlass::make_cute_packed_stride(StrideC53{}, shape_D);
  auto stride_D = cutlass::make_cute_packed_stride(StrideD53{}, shape_D);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  auto mode = batch > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm;

  typename Gemm53::Arguments args{
    mode,
    {M, N, K, batch},
    {A, stride_A, B, stride_B},
    {{alpha, beta}, nullptr, stride_C, D, stride_D},
    hw_info
  };

  Gemm53 gemm;
  size_t ws = Gemm53::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm.run() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaSuccess;
}

} // namespace example

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestCorrectness(int M, int N, int K, int batch, float alpha, float beta) {
  int lda = K, ldb = N, ldd = N;
  int64_t total_A = int64_t(batch) * M * K;
  int64_t total_B = int64_t(batch) * K * N;
  int64_t total_D = int64_t(batch) * M * N;
  size_t szD = total_D * sizeof(__half);

  __half *A, *B, *D_ref, *D_cut;
  AllocHalf53(&A, total_A, 0);
  AllocHalf53(&B, total_B, 17);
  cudaMalloc(&D_ref, szD); cudaMemset(D_ref, 0, szD);
  cudaMalloc(&D_cut, szD); cudaMemset(D_cut, 0, szD);

  // Run naive reference
  {
    dim3 bl(16, 16), gr((M + 15) / 16, (N + 15) / 16, batch);
    NaiveBatchedGemmRef_kernel<<<gr, bl>>>(M, N, K, batch, alpha, A, lda, B, ldb, beta, D_ref, ldd);
  }
  cudaDeviceSynchronize();

  bool cutlass_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (example::CutlassBatchedGemm53(M, N, K, batch, alpha,
      (cutlass::half_t const*)A, (cutlass::half_t const*)B,
      beta, (cutlass::half_t*)D_cut) == cudaSuccess) {
    cudaDeviceSynchronize();
    cutlass_ok = true;
  } else {
    cudaGetLastError(); // clear error
  }
#endif

#ifdef KH_TEST_SOLUTION
  __half *D_sol;
  cudaMalloc(&D_sol, szD); cudaMemset(D_sol, 0, szD);

  cudaError_t sr = HopperGemmPermute(M, N, K, batch, alpha, A, lda, B, ldb, beta, D_sol, ldd);
  if (sr != cudaSuccess) {
    std::cerr << "Solution failed: " << cudaGetErrorString(sr) << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
    return sr;
  }
  cudaDeviceSynchronize();

  __half *D_check = cutlass_ok ? D_cut : D_ref;
  std::vector<__half> hs(total_D), hr(total_D);
  cudaMemcpy(hs.data(), D_sol, szD, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), D_check, szD, cudaMemcpyDeviceToHost);
  float md = 0;
  for (int64_t i = 0; i < total_D; ++i) md = fmaxf(md, fabsf(__half2float(hs[i]) - __half2float(hr[i])));
  if (md > 1.0f) {
    std::cerr << "Solution incorrect. Max diff: " << md << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }
  cudaFree(D_sol);
#endif

  cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void Profile(int M, int N, int K, int batch, float alpha, float beta, int iterations) {
  int lda = K, ldb = N, ldd = N;
  int64_t total_A = int64_t(batch) * M * K;
  int64_t total_B = int64_t(batch) * K * N;
  int64_t total_D = int64_t(batch) * M * N;

  __half *A, *B, *D;
  AllocHalf53(&A, total_A, 0);
  AllocHalf53(&B, total_B, 17);
  cudaMalloc(&D, total_D * sizeof(__half));
  cudaMemset(D, 0, total_D * sizeof(__half));

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  float ref_min = 1e30f;
  bool rok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (example::CutlassBatchedGemm53(M, N, K, batch, alpha,
      (cutlass::half_t const*)A, (cutlass::half_t const*)B,
      beta, (cutlass::half_t*)D) == cudaSuccess) {
    cudaDeviceSynchronize();
    float rt = 0;
    for (int i = 0; i < iterations; ++i) {
      cudaEventRecord(t0);
      example::CutlassBatchedGemm53(M, N, K, batch, alpha,
        (cutlass::half_t const*)A, (cutlass::half_t const*)B,
        beta, (cutlass::half_t*)D);
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
  __half *D_ref_saved = nullptr;
  if (rok) {
    cudaMalloc(&D_ref_saved, total_D * sizeof(__half));
    cudaMemcpy(D_ref_saved, D, total_D * sizeof(__half), cudaMemcpyDeviceToDevice);
  }

  cudaMemset(D, 0, total_D * sizeof(__half));
  HopperGemmPermute(M, N, K, batch, alpha, A, lda, B, ldb, beta, D, ldd);
  cudaDeviceSynchronize();

  if (rok && D_ref_saved) {
    std::vector<__half> h_sol(total_D), h_ref(total_D);
    cudaMemcpy(h_sol.data(), D, total_D * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved, total_D * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(D_ref_saved);
    float md = 0;
    for (int64_t i = 0; i < total_D; ++i) md = fmaxf(md, fabsf(__half2float(h_sol[i]) - __half2float(h_ref[i])));
    if (md > 1.0f) {
      fprintf(stderr, "Solution incorrect in Profile: max_diff=%.6f\n", md);
      std::cout << "Incorrect" << std::endl;
      cudaEventDestroy(t0); cudaEventDestroy(t1);
      cudaFree(D); cudaFree(B); cudaFree(A);
      exit(-1);
    }
  }

  for (int i = 0; i < 3; i++)
    HopperGemmPermute(M, N, K, batch, alpha, A, lda, B, ldb, beta, D, ldd);
  cudaDeviceSynchronize();

  float st = 0, sm = 1e30f;
  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(t0);
    HopperGemmPermute(M, N, K, batch, alpha, A, lda, B, ldb, beta, D, ldd);
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
  cudaFree(B);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **argv) {

  struct TestConfig {
    const char* label;
    int m, n, k, batch;
  };

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (argc >= 5) {
    int M = 128, N = 128, K = 64, B = 8;
    std::stringstream(argv[1]) >> M;
    std::stringstream(argv[2]) >> N;
    std::stringstream(argv[3]) >> K;
    std::stringstream(argv[4]) >> B;
    if (argc > 5) std::stringstream(argv[5]) >> iterations;
    configs.push_back({"custom", M, N, K, B});
  } else {
    configs = {
      {"small",    128,  128,   64,  4},
      {"medium",   512,  512,  256, 16},
      {"large",   2048, 2048, 1024,  4},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch);

    if (TestCorrectness(cfg.m, cfg.n, cfg.k, cfg.batch, alpha, beta) != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, cfg.batch, alpha, beta, iterations);
    }
  }

  return EXIT_SUCCESS;
}
