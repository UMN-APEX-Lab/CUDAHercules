/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
  \brief Hopper gather+GEMM+scatter kernel fusion example.

  Fuses gather before GEMM and scatter after GEMM into the same kernel.
  N-mode gather/scatter: gather B columns, gather C columns, scatter D columns.

  For KH_TEST_SOLUTION:
  A row-major [M, K] (lda=K), B column-major [K, N_orig] (ldb=K),
  D column-major [M, N_orig] (ldd=M).
  gather_B selects index_size columns from B, scatter_D writes to selected columns of D.
*/

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "gather_gemm.hpp"
#include "gather_kernel.cuh"
#include "scatter_epilogue.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Naive reference kernels

__global__ void InitHalf52_kernel(__half *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float value = float(((idx + seed) * 16807 % 16) - 8) * 0.01f;
    matrix[idx] = __float2half(value);
  }
}

cudaError_t AllocHalf52(__half **m, int size, int seed) {
  cudaError_t r = cudaMalloc(reinterpret_cast<void**>(m), sizeof(__half) * size);
  if (r != cudaSuccess) return r;
  InitHalf52_kernel<<<(size + 255) / 256, 256>>>(*m, size, seed);
  return cudaGetLastError();
}

__global__ void InitIndices_kernel(int *indices, int size, int range, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    // Simple deterministic index generation, values in [0, range)
    indices[idx] = ((idx + seed) * 16807) % range;
    if (indices[idx] < 0) indices[idx] += range;
  }
}

/// Naive gather/scatter GEMM reference kernel
/// A row-major [M, K] (lda=K), B col-major [K, N_orig] (ldb=K)
/// gather_indices[j] selects column from B for logical column j (j in [0, index_size))
/// scatter_indices[j] selects column in D to write for logical column j
/// D col-major [M, N_orig] (ldd=M)
__global__ void NaiveGatherScatterRef_kernel(
    int M, int index_size, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    int const *gather_indices,
    float beta,
    __half *D, int ldd,
    int const *scatter_indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= index_size) return;

  int j_b = gather_indices[j];
  int j_d = scatter_indices[j];

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += __half2float(A[i * lda + k]) * __half2float(B[k + j_b * ldb]);
  }
  float old_val = beta * __half2float(D[i + j_d * ldd]);
  D[i + j_d * ldd] = __float2half(alpha * acc + old_val);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace example {

template<class ElementA, class LayoutA, class GatherA,
         class ElementB, class LayoutB, class GatherB,
         class ElementC, class LayoutC, class GatherC,
         class ElementD, class LayoutD, class ScatterD,
         class ElementAccumulator, class ElementComputeEpilogue>
struct ExampleRunner
{
  using ProblemShape = Shape<int,int,int,int>;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  using Epilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueGatherScatter<
      StrideC, StrideD,
      cutlass::epilogue::thread::LinearCombination<
        ElementD, 1,
        ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default,
        cutlass::FloatRoundStyle::round_to_nearest, ElementC
      >,
      cutlass::gemm::EpilogueDefault,
      GatherC,
      ScatterD
    >
  >;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    Shape<_128,_128,_64>,
    Shape<_1,_1,_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelCpAsyncWarpSpecialized
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmGather<
    ProblemShape,
    Mainloop,
    Epilogue,
    void,
    GatherA,
    GatherB
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static constexpr bool DoGatherA  = not cutlass::platform::is_same<GatherA,  NoGather>::value;
  static constexpr bool DoGatherB  = not cutlass::platform::is_same<GatherB,  NoGather>::value;
  static constexpr bool DoGatherC  = not cutlass::platform::is_same<GatherC,  NoGather>::value;
  static constexpr bool DoScatterD = not cutlass::platform::is_same<ScatterD, NoGather>::value;

  static constexpr bool GatherAonM  = DoGatherA  && cutlass::platform::is_same<LayoutA,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherAonK  = DoGatherA  && cutlass::platform::is_same<LayoutA,cutlass::layout::ColumnMajor>::value;
  static constexpr bool GatherBonN  = DoGatherB  && cutlass::platform::is_same<LayoutB,cutlass::layout::ColumnMajor>::value;
  static constexpr bool GatherBonK  = DoGatherB  && cutlass::platform::is_same<LayoutB,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherConM  = DoGatherC  && cutlass::platform::is_same<LayoutC,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherConN  = DoGatherC  && cutlass::platform::is_same<LayoutC,cutlass::layout::ColumnMajor>::value;
  static constexpr bool ScatterDonM = DoScatterD && cutlass::platform::is_same<LayoutD,cutlass::layout::RowMajor>::value;
  static constexpr bool ScatterDonN = DoScatterD && cutlass::platform::is_same<LayoutD,cutlass::layout::ColumnMajor>::value;

  static constexpr bool GatherModeM = GatherAonM || GatherConM || ScatterDonM;
  static constexpr bool GatherModeN = GatherBonN || GatherConN || ScatterDonN;
  static constexpr bool GatherModeK = GatherAonK || GatherBonK;

  static_assert( GatherModeM && !GatherModeN && !GatherModeK ||
                !GatherModeM &&  GatherModeN && !GatherModeK ||
                !GatherModeM && !GatherModeN &&  GatherModeK,
                "Only one gather mode (M, N or K) is supported by example runner");

  using MainloopRef = Mainloop;

  using EpilogueRef = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC, StrideC, StrideD,
      typename Epilogue::ThreadEpilogueOp,
      typename Epilogue::EpilogueSchedule
    >
  >;

  using KernelRef = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    MainloopRef,
    EpilogueRef,
    void
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<KernelRef>;

  cutlass::gemm::BatchedGemmCoord problem_size_orig;
  cutlass::gemm::BatchedGemmCoord problem_size;
  ProblemShape problem_shape_orig;
  ProblemShape problem_shape;
  cutlass::KernelHardwareInfo hw_info;

  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  StrideA stride_A_orig, stride_A;
  StrideB stride_B_orig, stride_B;
  StrideC stride_C_orig, stride_C;
  StrideD stride_D_orig, stride_D;

  cutlass::device_memory::allocation<ElementA> tensor_a;
  cutlass::device_memory::allocation<ElementB> tensor_b;
  cutlass::device_memory::allocation<ElementC> tensor_c;
  cutlass::device_memory::allocation<ElementD> tensor_d;

  cutlass::device_memory::allocation<int> gather_indices;

  cutlass::device_memory::allocation<ElementA> tensor_a_gathered;
  cutlass::device_memory::allocation<ElementB> tensor_b_gathered;
  cutlass::device_memory::allocation<ElementC> tensor_c_gathered;
  cutlass::device_memory::allocation<ElementD> tensor_d_gathered;
  cutlass::device_memory::allocation<ElementD> tensor_d_reference;

  cutlass::gemm::GemmUniversalMode gemm_mode;

  Gemm gemm;
  typename Gemm::Arguments arguments;
  cutlass::device_memory::allocation<uint8_t> workspace;

  GemmRef gemm_ref;
  typename GemmRef::Arguments arguments_ref;
  cutlass::device_memory::allocation<uint8_t> workspace_ref;

  bool init_ok;

  ExampleRunner(int M_orig, int N_orig, int K_val, int index_size,
                ElementComputeEpilogue alpha_, ElementComputeEpilogue beta_,
                cutlass::KernelHardwareInfo const &hw_info_)
  : problem_size_orig(M_orig, N_orig, K_val, 1),
    problem_size(GatherModeM ? index_size : problem_size_orig.m(),
                 GatherModeN ? index_size : problem_size_orig.n(),
                 GatherModeK ? index_size : problem_size_orig.k(),
                 problem_size_orig.batch()),
    problem_shape_orig(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.k(), problem_size_orig.batch()),
    problem_shape(problem_size.m(), problem_size.n(), problem_size.k(), problem_size.batch()),
    hw_info(hw_info_),
    alpha(alpha_),
    beta(beta_),
    stride_A_orig(cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(problem_size_orig.m(), problem_size_orig.k(), problem_size_orig.batch()))),
    stride_B_orig(cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(problem_size_orig.n(), problem_size_orig.k(), problem_size_orig.batch()))),
    stride_C_orig(cutlass::make_cute_packed_stride(
        StrideC{}, make_shape(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.batch()))),
    stride_D_orig(cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.batch()))),
    stride_A(cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(problem_size.m(), problem_size.k(), problem_size.batch()))),
    stride_B(cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(problem_size.n(), problem_size.k(), problem_size.batch()))),
    stride_C(cutlass::make_cute_packed_stride(
        StrideC{}, make_shape(problem_size.m(), problem_size.n(), problem_size.batch()))),
    stride_D(cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(problem_size.m(), problem_size.n(), problem_size.batch()))),
    tensor_a(problem_size_orig.m() * problem_size_orig.k()),
    tensor_b(problem_size_orig.k() * problem_size_orig.n()),
    tensor_c(problem_size_orig.m() * problem_size_orig.n()),
    tensor_d(problem_size_orig.m() * problem_size_orig.n()),
    gather_indices(index_size),
    tensor_a_gathered(problem_size.m() * problem_size.k()),
    tensor_b_gathered(problem_size.k() * problem_size.n()),
    tensor_c_gathered(problem_size.m() * problem_size.n()),
    tensor_d_gathered(problem_size.m() * problem_size.n()),
    tensor_d_reference(problem_size_orig.m() * problem_size_orig.n()),
    gemm_mode(cutlass::gemm::GemmUniversalMode::kGemm),
    gemm(),
    arguments{
      gemm_mode,
      problem_shape,
      {
        tensor_a.get(),
        stride_A_orig,
        tensor_b.get(),
        stride_B_orig
      },
      {
        { alpha, beta },
        tensor_c.get(), stride_C_orig,
        tensor_d.get(), stride_D_orig,
        typename Epilogue::GatherC {gather_indices.get()},
        typename Epilogue::ScatterD{gather_indices.get()}
      },
      hw_info,
      {},
      typename Kernel::GatherA{gather_indices.get()},
      typename Kernel::GatherB{gather_indices.get()}
    },
    workspace(Gemm::get_workspace_size(arguments)),
    gemm_ref(),
    arguments_ref{
      gemm_mode,
      problem_shape,
      {
        DoGatherA ? tensor_a_gathered.get() : tensor_a.get(),
        stride_A,
        DoGatherB ? tensor_b_gathered.get() : tensor_b.get(),
        stride_B
      },
      {
        { alpha, beta },
        DoGatherC  ? tensor_c_gathered.get() : tensor_c.get(),
        stride_C,
        DoScatterD ? tensor_d_gathered.get() : tensor_d_reference.get(),
        stride_D
      },
      hw_info
    },
    workspace_ref(GemmRef::get_workspace_size(arguments_ref)),
    init_ok(false)
  {
    cutlass::reference::device::BlockFillRandomUniform(tensor_a.get(), tensor_a.size(), 1, ElementA(7), ElementA(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_b.get(), tensor_b.size(), 1, ElementB(7), ElementB(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_c.get(), tensor_c.size(), 1, ElementC(7), ElementC(-8), 0);
    cutlass::reference::device::BlockFillSequential(tensor_d.get(), tensor_d.size(), ElementD(0), ElementD(0));

    int index_range = GatherModeM ? problem_size_orig.m() : (GatherModeN ? problem_size_orig.n() : problem_size_orig.k());
    std::vector<int> indices(index_range);
    std::iota(indices.begin(), indices.end(), 0);
    {
      std::random_device make_seed;
      std::mt19937 source_of_randomness(make_seed());
      std::shuffle(indices.begin(), indices.end(), source_of_randomness);
    }
    gather_indices.copy_from_host(indices.data());

    auto const try_init = [](auto & g, auto const & args, auto & ws) -> bool {
      cutlass::Status status = g.can_implement(args);
      if (status != cutlass::Status::kSuccess) return false;
      status = g.initialize(args, ws.get());
      if (status != cutlass::Status::kSuccess) return false;
      return true;
    };

    init_ok = try_init(gemm, arguments, workspace) && try_init(gemm_ref, arguments_ref, workspace_ref);
  }

  template<class Gemm2>
  static bool run_gemm(Gemm2 &gemm) {
    cutlass::Status status = gemm.run();
    return status == cutlass::Status::kSuccess;
  }

  template<class Gemm2>
  void run_reference(Gemm2 &g) {
    auto run_gather = [this](auto call, auto const & input, auto & output, auto gather_func, auto batch_size, auto stride) {
      [[maybe_unused]] auto idx = find_if(stride, [](auto x){ return not is_constant<1, decltype(x)>{}; });
      constexpr int I = decltype(idx)::value;
      call(input.get(), output.get(), gather_func, batch_size,
           static_cast<int>(input.size() / batch_size),
           static_cast<int>(output.size() / batch_size),
           static_cast<int>(get<I>(stride)), hw_info);
    };

    auto gather_call = [](auto&&... args){ gather(static_cast<decltype(args)&&>(args)...); };
    [[maybe_unused]] auto scatter_call = [](auto&&... args){ scatter(static_cast<decltype(args)&&>(args)...); };

    if constexpr (DoGatherA) {
      run_gather(gather_call, tensor_a, tensor_a_gathered, arguments.gather_A, problem_size.batch(), stride_A);
    }
    if constexpr (DoGatherB) {
      run_gather(gather_call, tensor_b, tensor_b_gathered, arguments.gather_B, problem_size.batch(), stride_B);
    }
    if constexpr (DoGatherC) {
      if (beta != ElementComputeEpilogue(0)) {
        run_gather(gather_call, tensor_c, tensor_c_gathered, arguments.epilogue.gather_C, problem_size.batch(), stride_C);
      }
    }

    run_gemm(g);

    if constexpr (DoScatterD) {
      run_gather(scatter_call, tensor_d_gathered, tensor_d_reference, arguments.epilogue.scatter_D, problem_size.batch(), stride_D);
    }
  }

  bool verify() {
    if (!init_ok) return false;
    if (!run_gemm(gemm)) return false;
    run_reference(gemm_ref);
    cudaDeviceSynchronize();
    return cutlass::reference::device::BlockCompareEqual(tensor_d.get(), tensor_d_reference.get(), tensor_d.size());
  }

  bool run_fused() {
    return run_gemm(gemm);
  }
};

} // namespace example

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestCorrectness(int M, int N_orig, int K, int index_size, float alpha, float beta) {
  int lda = K, ldb = K, ldd = M;
  int64_t total_A = int64_t(M) * K;
  int64_t total_B = int64_t(K) * N_orig;
  int64_t total_D = int64_t(M) * N_orig;

  __half *A, *B, *D_ref;
  int *gather_idx, *scatter_idx;

  AllocHalf52(&A, total_A, 0);
  AllocHalf52(&B, total_B, 17);
  cudaMalloc(&D_ref, total_D * sizeof(__half));
  cudaMemset(D_ref, 0, total_D * sizeof(__half));

  // Create deterministic indices
  cudaMalloc(&gather_idx, index_size * sizeof(int));
  cudaMalloc(&scatter_idx, index_size * sizeof(int));
  InitIndices_kernel<<<(index_size + 255) / 256, 256>>>(gather_idx, index_size, N_orig, 42);
  InitIndices_kernel<<<(index_size + 255) / 256, 256>>>(scatter_idx, index_size, N_orig, 99);
  cudaDeviceSynchronize();

  // Run naive reference
  {
    dim3 bl(16, 16), gr((M + 15) / 16, (index_size + 15) / 16);
    NaiveGatherScatterRef_kernel<<<gr, bl>>>(M, index_size, K, alpha, A, lda, B, ldb,
                                              gather_idx, beta, D_ref, ldd, scatter_idx);
  }
  cudaDeviceSynchronize();

  // CUTLASS reference not used for correctness checking here (SM90-only),
  // naive reference is sufficient for verifying solution.

#ifdef KH_TEST_SOLUTION
  __half *D_sol;
  cudaMalloc(&D_sol, total_D * sizeof(__half));
  cudaMemset(D_sol, 0, total_D * sizeof(__half));

  cudaError_t sr = HopperGatherScatterGemm(M, index_size, K, alpha, A, lda, B, ldb,
                                            gather_idx, beta, D_sol, ldd, scatter_idx);
  if (sr != cudaSuccess) {
    std::cerr << "Solution failed: " << cudaGetErrorString(sr) << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(scatter_idx); cudaFree(gather_idx); cudaFree(B); cudaFree(A);
    return sr;
  }
  cudaDeviceSynchronize();

  std::vector<__half> hs(total_D), hr(total_D);
  cudaMemcpy(hs.data(), D_sol, total_D * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), D_ref, total_D * sizeof(__half), cudaMemcpyDeviceToHost);
  float md = 0;
  for (int64_t i = 0; i < total_D; ++i) md = fmaxf(md, fabsf(__half2float(hs[i]) - __half2float(hr[i])));
  if (md > 1e-1f) {
    std::cerr << "Solution incorrect. Max diff: " << md << std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(scatter_idx); cudaFree(gather_idx); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }
  cudaFree(D_sol);
#endif

  cudaFree(D_ref); cudaFree(scatter_idx); cudaFree(gather_idx); cudaFree(B); cudaFree(A);
  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void Profile(int M, int N_orig, int K, int index_size, float alpha, float beta, int iterations) {
  int lda = K, ldb = K, ldd = M;
  int64_t total_A = int64_t(M) * K;
  int64_t total_B = int64_t(K) * N_orig;
  int64_t total_D = int64_t(M) * N_orig;

  __half *A, *B, *D;
  int *gather_idx, *scatter_idx;

  AllocHalf52(&A, total_A, 0);
  AllocHalf52(&B, total_B, 17);
  cudaMalloc(&D, total_D * sizeof(__half));
  cudaMemset(D, 0, total_D * sizeof(__half));

  cudaMalloc(&gather_idx, index_size * sizeof(int));
  cudaMalloc(&scatter_idx, index_size * sizeof(int));
  InitIndices_kernel<<<(index_size + 255) / 256, 256>>>(gather_idx, index_size, N_orig, 42);
  InitIndices_kernel<<<(index_size + 255) / 256, 256>>>(scatter_idx, index_size, N_orig, 99);
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  // CUTLASS reference profiling skipped (SM90-only)

#ifdef KH_TEST_SOLUTION
  // Warmup
  for (int i = 0; i < 3; i++)
    HopperGatherScatterGemm(M, index_size, K, alpha, A, lda, B, ldb,
                             gather_idx, beta, D, ldd, scatter_idx);
  cudaDeviceSynchronize();

  float st = 0, sm = 1e30f;
  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(t0);
    HopperGatherScatterGemm(M, index_size, K, alpha, A, lda, B, ldb,
                             gather_idx, beta, D, ldd, scatter_idx);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    st += ms;
    if (ms < sm) sm = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n", st / iterations, iterations, sm);
#endif

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaFree(D);
  cudaFree(scatter_idx);
  cudaFree(gather_idx);
  cudaFree(B);
  cudaFree(A);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char ** argv) {

  struct TestConfig {
    const char* label;
    int m, n_orig, k, index_size;
  };

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (argc >= 5) {
    int M = 2048, N = 2048, K = 2048, idx = 1024;
    std::stringstream(argv[1]) >> M;
    std::stringstream(argv[2]) >> N;
    std::stringstream(argv[3]) >> K;
    std::stringstream(argv[4]) >> idx;
    if (argc > 5) std::stringstream(argv[5]) >> iterations;
    configs.push_back({"custom", M, N, K, idx});
  } else {
    configs = {
      {"small",   1024, 1024,  512,  512},
      {"medium",  4096, 4096, 2048, 2048},
      {"large",   8192, 8192, 4096, 4096},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N_orig=%d, K=%d, index_size=%d ===\n",
            cfg.label, cfg.m, cfg.n_orig, cfg.k, cfg.index_size);

    if (TestCorrectness(cfg.m, cfg.n_orig, cfg.k, cfg.index_size, alpha, beta) != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n_orig, cfg.k, cfg.index_size, alpha, beta, iterations);
    }
  }

  return EXIT_SUCCESS;
}
