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
    \brief Hopper GEMM with custom Collectives and Epilogue Swizzle.

    INT8 GEMM: A row-major (int8), B col-major (int8), D col-major (int8).
    INT32 accumulation with INT32 alpha/beta scalars.
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Problem configuration
using ElementA50 = int8_t;
using ElementB50 = int8_t;
using ElementAcc50 = int32_t;
using ElementOutput50 = int8_t;

using LayoutA50 = cutlass::layout::RowMajor;
using LayoutB50 = cutlass::layout::ColumnMajor;
using LayoutC50 = cutlass::layout::ColumnMajor;
using LayoutD50 = cutlass::layout::ColumnMajor;

using TileShape50 = Shape<_128,_64,_128>;
using ClusterShape50 = Shape<_1,_2,_1>;

constexpr int PipelineStages50 = 8;

using DispatchPolicy50 = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<PipelineStages50, ClusterShape50,
                           cutlass::gemm::KernelTmaWarpSpecialized>;

static constexpr cute::GMMA::Major GmmaMajorA50 = cute::GMMA::Major::K;
static constexpr cute::GMMA::Major GmmaMajorB50 = cute::GMMA::Major::K;

using TiledMma50 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
    ElementA50, ElementB50, ElementAcc50, TileShape50, GmmaMajorA50, GmmaMajorB50>()));

using GmemTiledCopyA50 = std::conditional< cute::size(shape<1>(ClusterShape50{})) == 1,
                           cute::SM90_TMA_LOAD,
                           cute::SM90_TMA_LOAD_MULTICAST>::type;

using GmemTiledCopyB50 = std::conditional< cute::size(shape<0>(ClusterShape50{})) == 1,
                           cute::SM90_TMA_LOAD,
                           cute::SM90_TMA_LOAD_MULTICAST>::type;

using SmemLayoutAtomA50 = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorA50, ElementA50, decltype(cute::get<0>(TileShape50{})), decltype(cute::get<2>(TileShape50{}))
  >());

using SmemLayoutAtomB50 = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorB50, ElementB50, decltype(cute::get<1>(TileShape50{})), decltype(cute::get<2>(TileShape50{}))
  >());

using CollectiveMainloop50 = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy50,
    TileShape50,
    ElementA50,
    cutlass::gemm::TagToStrideA_t<LayoutA50>,
    ElementB50,
    cutlass::gemm::TagToStrideB_t<LayoutB50>,
    TiledMma50,
    GmemTiledCopyA50,
    SmemLayoutAtomA50,
    void,
    cute::identity,
    GmemTiledCopyB50,
    SmemLayoutAtomB50,
    void,
    cute::identity
  >;

// Epilogue with swizzled shared memory layout
using PreSwizzleLayout50 = Layout< Shape< Shape <_32,_4   >,_64>,
                                   Stride<Stride< _1,_2048>,_32>>;

using TileShapeS2R50 = Shape<_128,_16>;

using SmemLayout50 = ComposedLayout<
                       Swizzle<3,4,3>,
                       smem_ptr_flag_bits<sizeof_bits<ElementAcc50>::value>,
                       PreSwizzleLayout50>;

using TiledCopyS2R50 = TiledCopy<
                         Copy_Atom<DefaultCopy, ElementAcc50>,
                         Layout< Shape<_128,_16>,
                                 Stride<_16,_1>>,
                         TileShapeS2R50>;

using Epilogue50 = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
  cutlass::epilogue::collective::Epilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC50>,
    cutlass::gemm::TagToStrideC_t<LayoutD50>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    SmemLayout50,
    Copy_Atom<DefaultCopy, ElementAcc50>,
    TiledCopyS2R50,
    Copy_Atom<DefaultCopy, ElementOutput50>>>;

using GemmKernel50 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop50,
    Epilogue50
>;

using Gemm50 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel50>;

/// Run CUTLASS GEMM: A row-major (int8), B col-major (int8), D col-major (int32)
/// The epilogue LinearCombination outputs int32_t.
cudaError_t CutlassGemm50(
  int M, int N, int K,
  int32_t alpha,
  int8_t const *A, int lda,
  int8_t const *B, int ldb,
  int32_t beta,
  int32_t *D, int ldd) {

  using StrideA_t = typename Gemm50::GemmKernel::StrideA;
  using StrideB_t = typename Gemm50::GemmKernel::StrideB;
  using StrideC_t = typename Gemm50::GemmKernel::StrideC;
  using StrideD_t = typename Gemm50::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA_t{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB_t{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC_t{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD_t{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Construct arguments matching the sm90_gemm_tma_warpspecialized kernel's Arguments struct
  typename Gemm50::GemmKernel::ProblemShape problem_shape{M, N, K, 1};
  typename Gemm50::GemmKernel::MainloopArguments mainloop_args{A, stride_A, B, stride_B};
  typename Gemm50::GemmKernel::EpilogueArguments epilogue_args{
    {alpha, beta}, nullptr, stride_C, D, stride_D
  };

  typename Gemm50::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    mainloop_args,
    epilogue_args,
    hw_info
  };

  Gemm50 gemm;
  size_t workspace_size = Gemm50::get_workspace_size(arguments);
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

/// Kernel to initialize a matrix with small integers (int8).
__global__ void InitializeMatrix_i8_kernel(
  int8_t *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    int const k = 16807;
    int const m = 16;
    int value = ((idx + seed) * k % m) - m / 2;
    matrix[idx] = static_cast<int8_t>(value);
  }
}

cudaError_t AllocateMatrixI8(int8_t **matrix, int size, int seed) {
  cudaError_t result;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof(int8_t) * size);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sizeof(int8_t) * size);
  if (result != cudaSuccess) return result;
  InitializeMatrix_i8_kernel<<<(size + 255) / 256, 256>>>(*matrix, size, seed);
  return cudaGetLastError();
}

/// Kernel to initialize a matrix with small integers (int32).
__global__ void InitializeMatrix_i32_kernel(
  int32_t *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    int const k = 16807;
    int const m = 16;
    int value = ((idx + seed) * k % m) - m / 2;
    matrix[idx] = static_cast<int32_t>(value);
  }
}

cudaError_t AllocateMatrixI32(int32_t **matrix, int size, int seed) {
  cudaError_t result;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof(int32_t) * size);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sizeof(int32_t) * size);
  if (result != cudaSuccess) return result;
  InitializeMatrix_i32_kernel<<<(size + 255) / 256, 256>>>(*matrix, size, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: A row-major (int8), B col-major (int8), D col-major (int32)
/// Accumulates in int32, alpha/beta are int32. No clamping (output is int32).
__global__ void ReferenceGemm_kernel(
  int M, int N, int K,
  int32_t alpha,
  int8_t const *A, int lda,
  int8_t const *B, int ldb,
  int32_t beta,
  int32_t *D, int ldd) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += static_cast<int32_t>(A[i * lda + k]) * static_cast<int32_t>(B[k + j * ldb]);
    }
    D[i + j * ldd] = alpha * acc + beta * D[i + j * ldd];
  }
}

cudaError_t ReferenceGemm(
  int M, int N, int K,
  int32_t alpha,
  int8_t const *A, int lda,
  int8_t const *B, int ldb,
  int32_t beta,
  int32_t *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestCorrectness(int M, int N, int K, int32_t alpha, int32_t beta) {
  int lda = K, ldb = K, ldd = M;
  size_t sizeof_D = sizeof(int32_t) * M * N;

  int8_t *A, *B;
  int32_t *D_cutlass, *D_reference;

  AllocateMatrixI8(&A, M * K, 0);
  AllocateMatrixI8(&B, K * N, 17);
  AllocateMatrixI32(&D_cutlass, M * N, 101);
  AllocateMatrixI32(&D_reference, M * N, 101);

  // Run naive reference
  ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, D_reference, ldd);
  cudaDeviceSynchronize();

  bool cutlass_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Reset D_cutlass to same initial values
  InitializeMatrix_i32_kernel<<<(M * N + 255) / 256, 256>>>(D_cutlass, M * N, 101);
  cudaDeviceSynchronize();
  cudaError_t result = CutlassGemm50(M, N, K, alpha, A, lda, B, ldb, beta, D_cutlass, ldd);
  if (result == cudaSuccess) {
    cudaDeviceSynchronize();
    // Verify CUTLASS vs naive
    std::vector<int32_t> host_cutlass(M * N);
    std::vector<int32_t> host_reference(M * N);
    cudaMemcpy(host_cutlass.data(), D_cutlass, sizeof_D, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_reference.data(), D_reference, sizeof_D, cudaMemcpyDeviceToHost);

    int max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
      int diff = std::abs(host_cutlass[i] - host_reference[i]);
      if (diff > max_diff) max_diff = diff;
    }
    if (max_diff <= 1) {
      cutlass_ok = true;
    } else {
      std::cerr << "CUTLASS vs naive mismatch (max diff " << max_diff << "), using naive as reference." << std::endl;
    }
  } else {
    std::cerr << "CUTLASS GEMM failed (non-SM90 GPU?), using naive reference." << std::endl;
    cudaGetLastError(); // clear error
  }
#endif

#ifdef KH_TEST_SOLUTION
  int32_t *D_solution;
  AllocateMatrixI32(&D_solution, M * N, 101);

  cudaError_t sol_result = HopperEpilogueSwizzleGemm(M, N, K, alpha, A, lda, B, ldb, beta, D_solution, ldd);
  if (sol_result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_result) << std::endl;
    cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
    return sol_result;
  }
  cudaDeviceSynchronize();

  // Compare solution vs the best available reference (CUTLASS if available, else naive)
  std::vector<int32_t> host_solution(M * N);
  std::vector<int32_t> host_ref(M * N);
  cudaMemcpy(host_solution.data(), D_solution, sizeof_D, cudaMemcpyDeviceToHost);
  if (cutlass_ok) {
    cudaMemcpy(host_ref.data(), D_cutlass, sizeof_D, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(host_ref.data(), D_reference, sizeof_D, cudaMemcpyDeviceToHost);
  }

  int sol_max_diff = 0;
  for (int i = 0; i < M * N; ++i) {
    int diff = std::abs(host_solution[i] - host_ref[i]);
    if (diff > sol_max_diff) sol_max_diff = diff;
  }
  int tol = cutlass_ok ? 1 : 0;
  if (sol_max_diff > tol) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
    cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

  cudaFree(D_solution);
#endif

  cudaFree(D_reference);
  cudaFree(D_cutlass);
  cudaFree(B);
  cudaFree(A);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void Profile(int M, int N, int K, int32_t alpha, int32_t beta, int iterations) {
  int lda = K, ldb = K, ldd = M;

  int8_t *A, *B;
  int32_t *D;
  AllocateMatrixI8(&A, M * K, 0);
  AllocateMatrixI8(&B, K * N, 17);
  AllocateMatrixI32(&D, M * N, 101);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float ref_min_ms = 1e30f;
  bool ref_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Try warmup -- may fail on non-SM90
  if (CutlassGemm50(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd) == cudaSuccess) {
    cudaDeviceSynchronize();
    for (int i = 0; i < 2; i++) {
      CutlassGemm50(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    }
    cudaDeviceSynchronize();

    float ref_total_ms = 0;
    for (int iter = 0; iter < iterations; ++iter) {
      cudaEventRecord(start);
      CutlassGemm50(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
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
  int32_t *D_ref_saved = nullptr;
  if (ref_ok) {
    cudaMalloc(&D_ref_saved, sizeof(int32_t) * M * N);
    cudaMemcpy(D_ref_saved, D, sizeof(int32_t) * M * N, cudaMemcpyDeviceToDevice);
  }

  InitializeMatrix_i32_kernel<<<(M * N + 255) / 256, 256>>>(D, M * N, 101);
  cudaDeviceSynchronize();
  HopperEpilogueSwizzleGemm(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  cudaDeviceSynchronize();

  if (ref_ok && D_ref_saved) {
    std::vector<int32_t> h_sol(M * N), h_ref(M * N);
    cudaMemcpy(h_sol.data(), D, sizeof(int32_t) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved, sizeof(int32_t) * M * N, cudaMemcpyDeviceToHost);
    cudaFree(D_ref_saved);
    int max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
      int diff = std::abs(h_sol[i] - h_ref[i]);
      if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1) {
      fprintf(stderr, "Solution incorrect in Profile: max_diff=%d\n", max_diff);
      std::cout << "Incorrect" << std::endl;
      cudaEventDestroy(start); cudaEventDestroy(stop);
      cudaFree(D); cudaFree(B); cudaFree(A);
      exit(-1);
    }
  }

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    HopperEpilogueSwizzleGemm(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  }
  cudaDeviceSynchronize();

  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    HopperEpilogueSwizzleGemm(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
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
  int32_t alpha = 1, beta = 0;
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

/////////////////////////////////////////////////////////////////////////////////////////////////
