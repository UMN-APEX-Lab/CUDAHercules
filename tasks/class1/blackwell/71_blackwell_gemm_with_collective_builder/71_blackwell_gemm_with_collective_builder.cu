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
    \brief Blackwell SM100 GEMM example demonstrating compatible mainloop+epilogue builder schedules
    and epilogue visitor tree (EVT) construction

    Example usage:
      $ ./examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder \
            --m=2048 --n=2048 --k=2048 --l=2
*/

#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif
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
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  float alpha, beta;
  int swizzle;

  Options():
    help(false),
    error(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f),
    swizzle(0)
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
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("swizzle", swizzle);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "71_blackwell_gemm_with_collective_builder\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  performant kernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n\n";

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
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Wrapper to construct, run, and verify a GEMM. This example showcases CUTLASS's collective
// operation builders by specializing the GEMM on the kernel+epilogue schedule it will use and the
// number of pipeline stages.
template <
  // Type of kernel schedule to generate
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  // Type of epilogue schedule to generate
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  // Number of pipeline stages to use
  class StageCountType = cutlass::gemm::collective::StageCountAuto,
  // Do we use custom epilogue visitor tree (EVT) fusion
  bool UseCustomEVT = false
>
struct ExampleRunner {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  using ClusterShapeMNK = Shape<_2,_2,_1>;
  static constexpr bool Use2SmMma =
      // Manually specified 2sm cluster MMA schedule, will error if cluster M is not a multiple of 2
      std::is_same_v<MainloopScheduleType, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100> ||
      // Auto schedule will try to select 2sm cluster MMA based on cluster M
      std::is_same_v<MainloopScheduleType, cutlass::gemm::collective::KernelScheduleAuto> && size<0>(ClusterShapeMNK{}) % 2 == 0;
  // The MMA tile used by the mainloop collective. Blackwell 1sm MMA supports up to MMA tile M = 128, 2sm MMA supports up to MMA tile M = 256
  using MmaTileMNK    = std::conditional_t<Use2SmMma, Shape<_256,_128,_64>, Shape<_128,_128,_64>>;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // Blackwell fusions for the most part use the same EVT nodes used in Hopper. Most Blackwell EVTs will alias to their Hopper counterparts.
  // EVT nodes new to Blackwell mainly relate to narrow precision scale factor generation and are contained in include/cutlass/epilogue/fusion/sm100_visitor_*.hpp
  // See include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp for EVT construction using these new nodes
  // Fusions relating to narrow-precision scale factor generation are demonstrated in example 72b and can only be used in blackwell kernels
  using CustomEVT =  // alpha * acc + beta * C
    cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // beta
      cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>, // C
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
        cutlass::epilogue::fusion::Sm90AccFetch // acc
      >
    >;

  // As in Hopper, a predefined set of fusion operations are provided in include/cutlass/epilogue/fusion/operations.hpp and can be passed to the epilogue builder
  // Fusions operations supported by the Hopper TMA epilogue will also be supported by the Blackwell TMA epilogue
  // Fusions relating to narrow-precision scale factor generation are demonstrated in example 72b and can only be used in blackwell kernels
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileMNK, ClusterShapeMNK,
      cute::conditional_t<cute::is_same_v<StageCountType, cutlass::gemm::collective::StageCountAuto>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCountType>,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

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
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, float alpha, float beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementScalar(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementScalar(beta),
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    arguments.scheduler.max_swizzle_size = options.swizzle;

    // See example 48 for details on custom EVT construction
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {    // ternary op : beta * C + (alpha * acc)
          {{options.beta}}, // leaf op+args : beta
          {},               // leaf op+args : C
          {                 // binary op : alpha * acc
            {{options.alpha}}, // leaf op+args : alpha
            {},                // leaf op+args : acc
            {}              // binary args : multiplies
          },                // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = options.alpha;
      arguments.epilogue.thread.beta = options.beta;
    }

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

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    return passed;
  }

  /// Run with KH_TEST_SOLUTION pattern: correctness + profiling
  int run_kh(const Options& options, const cutlass::KernelHardwareInfo& hw_info, int iterations) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    arguments.scheduler.max_swizzle_size = options.swizzle;

    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {
          {{options.beta}},
          {},
          {
            {{options.alpha}},
            {},
            {}
          },
          {}
        };
    }
    else {
      arguments.epilogue.thread.alpha = options.alpha;
      arguments.epilogue.thread.beta = options.beta;
    }

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported." << std::endl;
      std::cout << "Incorrect" << std::endl;
      return -1;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel." << std::endl;
      std::cout << "Incorrect" << std::endl;
      return -1;
    }

    // Run CUTLASS reference
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel." << std::endl;
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    cudaDeviceSynchronize();

    // Verify CUTLASS vs device reference
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "CUTLASS reference check failed" << std::endl;
      std::cout << "Incorrect" << std::endl;
      return -1;
    }

#ifdef KH_TEST_SOLUTION
    // Verify solution against CUTLASS reference
    {
      int lda = options.k, ldb = options.k, ldd = options.m;
      size_t num_D = (size_t)options.m * options.n;

      // Allocate solution output
      cutlass::DeviceAllocation<cutlass::half_t> block_sol_D;
      block_sol_D.reset(num_D);

      cudaError_t sol_err = BlackwellCollectiveGemm(
        options.m, options.n, options.k,
        options.alpha,
        reinterpret_cast<const __half*>(block_A.get()), lda,
        reinterpret_cast<const __half*>(block_B.get()), ldb,
        options.beta,
        reinterpret_cast<__half*>(block_sol_D.get()), ldd);

      if (sol_err != cudaSuccess) {
        std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
        std::cout << "Incorrect" << std::endl;
        return -1;
      }
      cudaDeviceSynchronize();

      // Compare solution vs CUTLASS reference (block_D)
      std::vector<cutlass::half_t> h_sol(num_D), h_ref(num_D);
      cudaMemcpy(h_sol.data(), block_sol_D.get(), num_D * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_ref.data(), block_D.get(), num_D * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);

      float max_diff = 0.0f;
      for (size_t i = 0; i < num_D; ++i) {
        max_diff = fmaxf(max_diff, fabsf(float(h_sol[i]) - float(h_ref[i])));
      }
      if (max_diff > 1e-1f) {
        std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
        std::cout << "Incorrect" << std::endl;
        return -1;
      }
    }
#endif

    std::cout << "Passed" << std::endl;

    // Profiling
    if (iterations > 0) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      // Warmup reference
      for (int i = 0; i < 3; i++) {
        gemm_op.initialize(arguments, workspace.get());
        gemm_op.run();
      }
      cudaDeviceSynchronize();

      // Profile reference
      float ref_total_ms = 0, ref_min_ms = 1e30f;
      for (int iter = 0; iter < iterations; ++iter) {
        cudaEventRecord(start);
        gemm_op.initialize(arguments, workspace.get());
        gemm_op.run();
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
      {
        int lda = options.k, ldb = options.k, ldd = options.m;
        size_t num_D = (size_t)options.m * options.n;
        cutlass::DeviceAllocation<cutlass::half_t> prof_D;
        prof_D.reset(num_D);

        // Warmup solution
        for (int i = 0; i < 3; i++) {
          BlackwellCollectiveGemm(options.m, options.n, options.k, options.alpha,
            reinterpret_cast<const __half*>(block_A.get()), lda,
            reinterpret_cast<const __half*>(block_B.get()), ldb,
            options.beta,
            reinterpret_cast<__half*>(prof_D.get()), ldd);
        }
        cudaDeviceSynchronize();

        // Profile solution
        float sol_total_ms = 0, sol_min_ms = 1e30f;
        for (int iter = 0; iter < iterations; ++iter) {
          cudaEventRecord(start);
          BlackwellCollectiveGemm(options.m, options.n, options.k, options.alpha,
            reinterpret_cast<const __half*>(block_A.get()), lda,
            reinterpret_cast<const __half*>(block_B.get()), ldb,
            options.beta,
            reinterpret_cast<__half*>(prof_D.get()), ldd);
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
      }
#endif

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    return 0;
  }

};

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
}

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

  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100a)." << std::endl;
    return 0;
  }

  //
  // Parse options
  //

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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  // Check if user explicitly passed size
  bool explicit_size = false;
  {
    cutlass::CommandLine cmd(argc, args);
    explicit_size = cmd.check_cmd_line_flag("m") ||
                    cmd.check_cmd_line_flag("n") ||
                    cmd.check_cmd_line_flag("k");
  }

  std::vector<TestConfig> configs;
  int iterations = 20;

  if (explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k});
  } else {
    configs = {
      {"small",   1024, 1024, 1024},
      {"medium",  4096, 4096, 4096},
      {"large",   8192, 8192, 8192},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    Options test_options = options;
    test_options.m = cfg.m;
    test_options.n = cfg.n;
    test_options.k = cfg.k;

    // Use auto-schedule runner (default ExampleRunner)
    ExampleRunner<> runner;
    int ret = runner.run_kh(test_options, hw_info, iterations);
    if (ret != 0) return ret;
  }

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
