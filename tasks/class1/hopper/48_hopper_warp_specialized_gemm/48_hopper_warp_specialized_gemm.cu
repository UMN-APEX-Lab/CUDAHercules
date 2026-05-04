/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    TF32 GEMM using Warp Specialized kernel design.
    A is row-major, B is column-major, C/D is column-major. All float.
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

using         ElementA    = float;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using         ElementB    = float;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

using         ElementC    = float;
using         LayoutC     = cutlass::layout::ColumnMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using TileShape           = Shape<_128,_128,_32>;
using ClusterShape        = Shape<_4,_2,_1>;
using StageCountType = cutlass::gemm::collective::StageCountAuto;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Run CUTLASS GEMM: A row-major, B col-major, C col-major
cudaError_t CutlassGemm48(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  int device_id = 0;
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {A, stride_A, B, stride_B},
    {{alpha, beta}, C, stride_C, C, stride_D},
    hw_info
  };

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed" << std::endl;
    return cudaErrorUnknown;
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "initialize failed" << std::endl;
    return cudaErrorUnknown;
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "run failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    int const k = 16807;
    int const m = 16;
    float value = float(((idx + seed) * k % m) - m / 2);
    matrix[idx] = value;
  }
}

cudaError_t InitializeMatrix(float *matrix, int size, int seed) {
  int blocks = (size + 255) / 256;
  InitializeMatrix_kernel<<<blocks, 256>>>(matrix, size, seed);
  return cudaGetLastError();
}

cudaError_t AllocateMatrix(float **matrix, int size, int seed) {
  cudaError_t result;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof(float) * size);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sizeof(float) * size);
  if (result != cudaSuccess) return result;
  result = InitializeMatrix(*matrix, size, seed);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: A row-major, B col-major, C col-major
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
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[i * lda + k] * B[k + j * ldb];
    }
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
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

cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  int lda = K, ldb = K, ldc = M;
  size_t sizeof_C = sizeof(float) * M * N;

  float *A, *B, *C_cutlass, *C_reference;

  AllocateMatrix(&A, M * K, 0);
  AllocateMatrix(&B, K * N, 17);
  AllocateMatrix(&C_cutlass, M * N, 101);
  AllocateMatrix(&C_reference, M * N, 101);

  // Run naive reference
  ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  cudaDeviceSynchronize();

  bool cutlass_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Reset C_cutlass to same initial values
  InitializeMatrix(C_cutlass, M * N, 101);
  cudaError_t result = CutlassGemm48(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result == cudaSuccess) {
    cudaDeviceSynchronize();
    // Verify CUTLASS vs naive (TF32 tolerance)
    std::vector<float> host_cutlass(M * N);
    std::vector<float> host_reference(M * N);
    cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    float max_rel_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
      float ref_val = host_reference[i];
      float diff = std::fabs(host_cutlass[i] - ref_val);
      float rel = diff / (std::fabs(ref_val) + 1e-6f);
      max_rel_diff = std::fmax(max_rel_diff, rel);
    }
    if (max_rel_diff <= 0.01f) {
      cutlass_ok = true;
    } else {
      std::cerr << "CUTLASS vs naive mismatch (rel diff " << max_rel_diff << "), using naive as reference." << std::endl;
    }
  } else {
    std::cerr << "CUTLASS GEMM failed (non-SM90 GPU?), using naive reference." << std::endl;
    cudaGetLastError(); // clear error
  }
#endif

#ifdef KH_TEST_SOLUTION
  float *C_solution;
  AllocateMatrix(&C_solution, M * N, 101);

  cudaError_t sol_result = HopperGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_solution, ldc);
  if (sol_result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_result) << std::endl;
    cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
    return sol_result;
  }
  cudaDeviceSynchronize();

  // Compare solution vs the best available reference (CUTLASS if available, else naive)
  std::vector<float> host_solution(M * N);
  std::vector<float> host_ref(M * N);
  cudaMemcpy(host_solution.data(), C_solution, sizeof_C, cudaMemcpyDeviceToHost);
  if (cutlass_ok) {
    cudaMemcpy(host_ref.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(host_ref.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
  }

  float sol_max_diff = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    sol_max_diff = fmaxf(sol_max_diff, fabsf(host_solution[i] - host_ref[i]));
  }
  // Use absolute tolerance for exact match against naive, or relative for CUTLASS/TF32
  float tol = cutlass_ok ? 0.01f : 1e-3f;
  bool check_rel = cutlass_ok;
  if (check_rel) {
    float sol_max_rel = 0.0f;
    for (int i = 0; i < M * N; ++i) {
      float ref_val = host_ref[i];
      float diff = std::fabs(host_solution[i] - ref_val);
      float rel = diff / (std::fabs(ref_val) + 1e-6f);
      sol_max_rel = std::fmax(sol_max_rel, rel);
    }
    if (sol_max_rel > tol) {
      std::cerr << "Solution incorrect. Max rel diff vs reference: " << sol_max_rel << std::endl;
      cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
      return cudaErrorUnknown;
    }
  } else {
    if (sol_max_diff > tol) {
      std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
      cudaFree(C_solution); cudaFree(C_reference); cudaFree(C_cutlass); cudaFree(B); cudaFree(A);
      return cudaErrorUnknown;
    }
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

void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = K, ldc = M;

  float *A, *B, *C;
  AllocateMatrix(&A, M * K, 0);
  AllocateMatrix(&B, K * N, 17);
  AllocateMatrix(&C, M * N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float ref_min_ms = 1e30f;
  bool ref_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Try warmup -- may fail on non-SM90
  if (CutlassGemm48(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) == cudaSuccess) {
    cudaDeviceSynchronize();
    for (int i = 0; i < 2; i++) {
      CutlassGemm48(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    cudaDeviceSynchronize();

    float ref_total_ms = 0;
    for (int iter = 0; iter < iterations; ++iter) {
      cudaEventRecord(start);
      CutlassGemm48(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      ref_total_ms += ms;
      if (ms < ref_min_ms) ref_min_ms = ms;
    }
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            ref_total_ms / iterations, iterations, ref_min_ms);
    ref_ok = true;
  } else {
    cudaGetLastError(); // clear
  }
#endif

#ifdef KH_TEST_SOLUTION
  // Save reference output for correctness check
  float *C_ref_saved = nullptr;
  if (ref_ok) {
    cudaMalloc(&C_ref_saved, sizeof(float) * M * N);
    cudaMemcpy(C_ref_saved, C, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);
  }

  InitializeMatrix(C, M * N, 101);
  // Run solution once for correctness check
  HopperGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  cudaDeviceSynchronize();

  if (ref_ok && C_ref_saved) {
    std::vector<float> h_sol(M * N), h_ref(M * N);
    cudaMemcpy(h_sol.data(), C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), C_ref_saved, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaFree(C_ref_saved);
    float max_rel = 0;
    for (int i = 0; i < M * N; ++i) {
      float diff = std::fabs(h_sol[i] - h_ref[i]);
      float rel = diff / (std::fabs(h_ref[i]) + 1e-6f);
      max_rel = std::fmax(max_rel, rel);
    }
    if (max_rel > 0.01f) {
      fprintf(stderr, "Solution incorrect in Profile: max_rel_diff=%.6f\n", max_rel);
      std::cout << "Incorrect" << std::endl;
      cudaEventDestroy(start); cudaEventDestroy(stop);
      cudaFree(C); cudaFree(B); cudaFree(A);
      exit(-1);
    }
  }

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    HopperGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    HopperGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sol_total_ms += ms;
    if (ms < sol_min_ms) sol_min_ms = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol_total_ms / iterations, iterations, sol_min_ms);
  if (ref_ok) {
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
  }
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
