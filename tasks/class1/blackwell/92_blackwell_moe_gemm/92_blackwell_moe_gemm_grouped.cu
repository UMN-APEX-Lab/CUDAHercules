/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Example of Blackwell MoE-style grouped GEMM implementation using TMA to load A and CPASYNC to load B.

  This example demonstrates an implementation of GEMM using mixed TMA+CPASYNC to load input matrices.
  In the decoding stage of Mixture of Experts (MoE) models, the number of tokens in different experts 
  can varies a lot, which requires frequently updates of TMA descriptors in TMA-based implementation.
  This examples uses CPASYNC to load activation (B) matrix to avoid the overhead of updating TMA descriptors.

  Usage:
  $ ./examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped
  --m=7168 --n=128 --k=512 --groups=8 --benchmark=benchmark.txt

*/

#include <iostream>
#include <fstream>

#ifdef KH_TEST_SOLUTION
#include <cuda_fp8.h>
#include "solution.h"
#endif

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"


using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool verification;

  int m, n, k, groups;

  int iterations;

  std::string benchmark_path;

  Options():
    help(false),
    error(false),
    verification(true),
    m(2048), n(2048), k(2048), groups(1),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("groups", groups, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 10);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);


    if (cmd.check_cmd_line_flag("no_verif")) {
      verification = false;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "92_blackwell_moe_gemm_grouped\n\n"
      << "  Blackwell MoE-style grouped GEMM implementation using TMA to load A and CPASYNC to load B\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --groups=<int>              Sets the groups extent (batch count) of the GEMM\n"
      << "  --iterations=<int>          Set the number of profiling iterations to perform\n"
      << "  --benchmark=<file>          Executes a benchmark problem size\n"
      << "  --no_verif                  Do not run verification kernels\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = static_cast<Element>(2);
    scope_min = static_cast<Element>(0);
  }
  else if (bits_input <= 8) {
    scope_max = static_cast<Element>(2);
    scope_min = static_cast<Element>(-2);
  }
  else {
    scope_max = static_cast<Element>(8);
    scope_min = static_cast<Element>(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleRunner {

  // Type of kernel schedule to generate
  using MainloopScheduleType = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmSm100;
  // Type of epilogue schedule to generate
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  using ClusterShapeMNK = Shape<_1,_1,_1>;
  using MmaTileMNK    = Shape<_128,_16,Int<128 / sizeof(ElementA)>>;  // use tile size of N=16 to match real use cases (N is typically very small in decoding stage)

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileMNK, ClusterShapeMNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType
    >::CollectiveOp;

  using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int,int,int>>; // <M,N,K> per group

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
      //, cutlass::gemm::MoEScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

  //
  // Data members
  //

  /// Initialization
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;

  cutlass::DeviceAllocation<int32_t> tokens_per_expert;

  std::vector<ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<int32_t> tokens_per_expert_host;


  //
  // Methods
  //

  bool verify(ProblemShape const& problem_size, float alpha, float beta) {
    auto [maxM, maxN, maxK] = problem_size.get_host_problem_shape(0); //gets max problem shape
    for (int i = 0; i < problem_size.num_groups; i++) {
      auto problem = problem_sizes_host.at(i);
      auto [M, N, K] = problem;
      printf("group [%d] : M = %d, N = %d, K = %d\n", i, M, N, K);

      cutlass::TensorRef ref_A(block_A.get() + size_t(1) * i * maxM * maxK, Gemm::LayoutA(maxK));
      cutlass::TensorRef ref_B(block_B.get() + size_t(1) * i * maxN * maxK, Gemm::LayoutB(maxK));
      cutlass::TensorRef ref_C(block_C.get() + size_t(1) * i * maxN * maxM, Gemm::LayoutC(maxM));
      cutlass::TensorRef ref_D(block_ref_D.get() + size_t(1) * i * maxN * maxM, Gemm::LayoutD(maxM));

      using DeviceGemmReference = cutlass::reference::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementScalar,
        ElementAccumulator>;

      DeviceGemmReference gemm_reference;

      gemm_reference(
        {M, N, K},
        ElementScalar(alpha),
        ref_A,
        ref_B,
        ElementScalar(beta),
        ref_C,
        ref_D);

      cudaError_t result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        std::cerr << "Reference kernel failed. Last CUDA error: "
                  << cudaGetErrorString(result) << std::endl;
        return false;
      }

      // Check if output from CUTLASS kernel and reference kernel are equal or not
      // assume all M == maxM
      bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get() + size_t(1) * i * maxN * maxM, block_D.get() + size_t(1) * i * maxN * maxM, M * N);
      if (!passed) {
        return false;
      }
    }

    return true;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(ProblemShape const& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size.get_host_problem_shape(0), problem_size.groups());
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(size_t(1) * M * K * L);
    block_B.reset(size_t(1) * K * N * L);
    block_C.reset(size_t(1) * M * N * L);
    block_D.reset(size_t(1) * M * N * L);
    block_ref_D.reset(size_t(1) * M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  /// Load a benchmark
  void benchmark_problems(std::string const& benchmark_path) {

    std::ifstream file(benchmark_path);
    if (!file.good()) {
      std::cout << "Issues with benchmark file." << std::endl;
      return;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        extent.at(i) = std::atoi(tokens.at(i).c_str());
      }
      problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
    }

    return;
  }

  void benchmark_tokens_per_expert(std::string const& benchmark_path) {

    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        extent.at(i) = std::atoi(tokens.at(i).c_str());
      }
      tokens_per_expert_host.push_back(extent.n());
    }

    return;
  }

  bool run(Options const& options, cutlass::KernelHardwareInfo const& hw_info) {

    // If problems not already set up, load from benchmark file
    if (problem_sizes_host.empty()) {
      benchmark_problems(options.benchmark_path);
      if (problem_sizes_host.empty()) {
        return false;
      }
    }

    if (tokens_per_expert_host.empty()) {
      benchmark_tokens_per_expert(options.benchmark_path);
      if (tokens_per_expert_host.empty()) {
        return false;
      }
    }

    tokens_per_expert.reset(tokens_per_expert_host.size());
    tokens_per_expert.copy_from_host(tokens_per_expert_host.data());

    ProblemShape problem_size {options.m, options.n, options.k, options.groups, tokens_per_expert.get()};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem_size,
      {block_A.get(), block_B.get()},
      {{}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    // arguments.scheduler.max_swizzle_size = options.swizzle;
    
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    if (options.verification) {
      // Verify that the result is correct
      bool passed = verify(problem_size, 1.0f, 0.0f);

      std::cout << "  Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) {
        exit(-1);
        return false;
      }
    }

    // Run profiling loop
    if (options.iterations > 0)
    {
      // Per-iteration timing for reference
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);

      // Warmup
      for (int i = 0; i < 3; i++) {
        CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
        CUTLASS_CHECK(gemm_op.run());
      }
      cudaDeviceSynchronize();

      float ref_total = 0, ref_min = 1e30f;
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
        cudaEventRecord(ev_start);
        CUTLASS_CHECK(gemm_op.run());
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        ref_total += ms;
        if (ms < ref_min) ref_min = ms;
      }
      fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
              ref_total / options.iterations, options.iterations, ref_min);

#ifdef KH_TEST_SOLUTION
      // Run solution kernel and verify
      {
        int M = options.m, N = options.n, K = options.k;
        int num_experts = options.groups;
        float alpha = 1.0f, beta = 0.0f;

        auto [maxM, maxN, maxK] = problem_size.get_host_problem_shape(0);

        // Prepare FP8 data for solution from CUTLASS allocations
        std::vector<__nv_fp8_e4m3*> h_A_ptrs(num_experts), h_B_ptrs(num_experts);
        std::vector<__half*> h_D_sol(num_experts);
        std::vector<int> Ms(num_experts, M), Ns(num_experts, N), Ks(num_experts, K);
        std::vector<int> ldas(num_experts, K), ldbs(num_experts, K), ldds(num_experts, M);

        for (int e = 0; e < num_experts; ++e) {
          h_A_ptrs[e] = reinterpret_cast<__nv_fp8_e4m3*>(block_A.get() + size_t(1) * e * maxM * maxK);
          h_B_ptrs[e] = reinterpret_cast<__nv_fp8_e4m3*>(block_B.get() + size_t(1) * e * maxN * maxK);
          cudaMalloc(&h_D_sol[e], sizeof(__half) * M * N);
          cudaMemset(h_D_sol[e], 0, sizeof(__half) * M * N);
        }

        int *d_Ms, *d_Ns, *d_Ks, *d_ldas, *d_ldbs, *d_ldds;
        __nv_fp8_e4m3 **d_A_ptrs, **d_B_ptrs;
        __half **d_D_ptrs;

        cudaMalloc(&d_Ms, sizeof(int)*num_experts);
        cudaMalloc(&d_Ns, sizeof(int)*num_experts);
        cudaMalloc(&d_Ks, sizeof(int)*num_experts);
        cudaMalloc(&d_ldas, sizeof(int)*num_experts);
        cudaMalloc(&d_ldbs, sizeof(int)*num_experts);
        cudaMalloc(&d_ldds, sizeof(int)*num_experts);
        cudaMalloc(&d_A_ptrs, sizeof(void*)*num_experts);
        cudaMalloc(&d_B_ptrs, sizeof(void*)*num_experts);
        cudaMalloc(&d_D_ptrs, sizeof(void*)*num_experts);

        cudaMemcpy(d_Ms, Ms.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ns, Ns.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ks, Ks.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ldas, ldas.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ldbs, ldbs.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ldds, ldds.data(), sizeof(int)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_ptrs, h_A_ptrs.data(), sizeof(void*)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_ptrs, h_B_ptrs.data(), sizeof(void*)*num_experts, cudaMemcpyHostToDevice);
        cudaMemcpy(d_D_ptrs, h_D_sol.data(), sizeof(void*)*num_experts, cudaMemcpyHostToDevice);

        cudaError_t sol_err = BlackwellMoeGemm(
          num_experts, d_Ms, d_Ns, d_Ks, alpha, beta,
          (__nv_fp8_e4m3 const*const*)d_A_ptrs, d_ldas,
          (__nv_fp8_e4m3 const*const*)d_B_ptrs, d_ldbs,
          d_D_ptrs, d_ldds);

        if (sol_err != cudaSuccess) {
          std::cerr << "Solution failed: " << cudaGetErrorString(sol_err) << std::endl;
          return false;
        }
        cudaDeviceSynchronize();

        // Verify solution against CUTLASS reference (block_D)
        bool sol_ok = true;
        for (int e = 0; e < num_experts && sol_ok; ++e) {
          // CUTLASS output: col-major [M,N] at block_D + e*maxM*maxN, element type half_t
          std::vector<__half> sol_out(M * N);
          cudaMemcpy(sol_out.data(), h_D_sol[e], sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

          // CUTLASS ref output: col-major half_t
          std::vector<cutlass::half_t> ref_out(maxM * maxN);
          cudaMemcpy(ref_out.data(), block_D.get() + size_t(1) * e * maxM * maxN,
                     sizeof(cutlass::half_t) * maxM * maxN, cudaMemcpyDeviceToHost);

          int mismatches = 0;
          for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
              // col-major index
              float r = float(ref_out[i + j * maxM]);
              float s = __half2float(sol_out[i + j * M]);
              float diff = fabsf(r - s);
              float denom = fmaxf(fabsf(r), 1e-6f);
              if (diff / denom > 0.3f) mismatches++;
            }
          }
          if (float(mismatches) / float(M * N) > 0.01f) {
            std::cerr << "Solution incorrect at expert " << e << std::endl;
            sol_ok = false;
          }
        }

        if (!sol_ok) return false;

        // Profile solution
        for (int i = 0; i < 3; i++) {
          BlackwellMoeGemm(num_experts, d_Ms, d_Ns, d_Ks, alpha, beta,
            (__nv_fp8_e4m3 const*const*)d_A_ptrs, d_ldas,
            (__nv_fp8_e4m3 const*const*)d_B_ptrs, d_ldbs,
            d_D_ptrs, d_ldds);
        }
        cudaDeviceSynchronize();

        float sol_total = 0, sol_min = 1e30f;
        for (int iter = 0; iter < options.iterations; ++iter) {
          cudaEventRecord(ev_start);
          BlackwellMoeGemm(num_experts, d_Ms, d_Ns, d_Ks, alpha, beta,
            (__nv_fp8_e4m3 const*const*)d_A_ptrs, d_ldas,
            (__nv_fp8_e4m3 const*const*)d_B_ptrs, d_ldbs,
            d_D_ptrs, d_ldds);
          cudaEventRecord(ev_stop);
          cudaEventSynchronize(ev_stop);
          float ms;
          cudaEventElapsedTime(&ms, ev_start, ev_stop);
          sol_total += ms;
          if (ms < sol_min) sol_min = ms;
        }
        fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
                sol_total / options.iterations, options.iterations, sol_min);
        fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min / sol_min);

        // Cleanup
        cudaFree(d_Ms); cudaFree(d_Ns); cudaFree(d_Ks);
        cudaFree(d_ldas); cudaFree(d_ldbs); cudaFree(d_ldds);
        cudaFree(d_A_ptrs); cudaFree(d_B_ptrs); cudaFree(d_D_ptrs);
        for (int e = 0; e < num_experts; ++e) cudaFree(h_D_sol[e]);
      }
#endif

      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);
    }

    return true;
  }

  /// Set up problem sizes programmatically (without benchmark file)
  void setup_problems(int M, int N, int K, int groups) {
    problem_sizes_host.clear();
    tokens_per_expert_host.clear();
    for (int g = 0; g < groups; ++g) {
      problem_sizes_host.push_back({M, N, K});
      tokens_per_expert_host.push_back(N);
    }
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  if (!(props.major == 10 && props.minor == 0)) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100)." << std::endl;
    return 0;
  }

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  struct TestConfig {
    const char* label;
    int m, n, k, groups;
  };

  bool explicit_size = !options.benchmark_path.empty();
  int iterations = options.iterations > 0 ? options.iterations : 20;

  std::vector<TestConfig> configs;
  if (explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k, options.groups});
  } else {
    configs = {
      {"small",   128,  16,  256,  4},
      {"medium",  512,  16,  512, 16},
      {"large",  2048,  16, 2048,  4},
    };
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, groups=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.groups);

    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;
    options.groups = cfg.groups;
    options.iterations = iterations;

    ExampleRunner runner;
    if (explicit_size) {
      runner.benchmark_problems(options.benchmark_path);
      runner.benchmark_tokens_per_expert(options.benchmark_path);
    } else {
      runner.setup_problems(cfg.m, cfg.n, cfg.k, cfg.groups);
    }

    bool ok = runner.run(options, hw_info);
    if (!ok) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;
  }

#endif

  return 0;
}
