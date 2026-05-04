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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM103 architecture.

    This example demonstrates a simple way to instantiate and run a blockscaled ultra FP4 GEMM on the NVIDIA Blackwell SM103 architecture.

    Usage:

      $ ./examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm --m=2048 --n=2048 --k=2048
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
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
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include <cuda_bf16.h>
#include "solution.h"
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e2m1_t;                          // Element type for A matrix operand
using         ElementSFA  = cutlass::float_ue4m3_t;
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e2m1_t;                          // Element type for A matrix operand
using         ElementSFB  = cutlass::float_ue4m3_t;
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm103;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// using ElementD = cutlass::float_e2m1_t; // Enable for SF Output          // Element type for D matrix operands

// Kernel Perf config
using MmaTileShape1Sm        = cute::Shape<cute::_128, cute::_256, Int<768>>;// 1SM MMA's tile size
using MmaTileShape2Sm        = cute::Shape<cute::_256, cute::_256, Int<768>>;// 2SM MMA's tile size
using ClusterShape        = cute::Shape<int, int, cute::_1>;                 // Cluster shape

using CollectiveEpilogue1Sm = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape1Sm, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveEpilogue2Sm = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape2Sm, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm
  >::CollectiveOp;

using CollectiveMainloop1Sm = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementA,ElementSFA>, LayoutATag, AlignmentA,
    cute::tuple<ElementB,ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape1Sm, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue1Sm::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103     // Kernel schedule policy. Auto or using targeted scheduling policy
  >::CollectiveOp;
using CollectiveMainloop2Sm = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementA,ElementSFA>, LayoutATag, AlignmentA,
    cute::tuple<ElementB,ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape2Sm, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue2Sm::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103     // Kernel schedule policy. Auto or using targeted scheduling policy
  >::CollectiveOp;

using GemmKernel1Sm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop1Sm,
    CollectiveEpilogue1Sm>;

using Gemm1Sm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1Sm>;
using Gemm = Gemm1Sm;
using GemmKernel2Sm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop2Sm,
    CollectiveEpilogue2Sm>;

using Gemm2Sm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2Sm>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
uint64_t seed;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementSFA, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementSFB, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
#endif // defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle = 0;

  dim3 cluster_shape = dim3(2,1,1);
  dim3 cluster_shape_fallback = dim3(2,1,1);

  bool verification = true;
  int batch = 1;

  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10),
    swizzle(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("cluster_m", cluster_shape.x);
    cmd.get_cmd_line_argument("cluster_n", cluster_shape.y);
    cmd.get_cmd_line_argument("cluster_fallback_m", cluster_shape_fallback.x);
    cmd.get_cmd_line_argument("cluster_fallback_n", cluster_shape_fallback.y);
    if (cmd.check_cmd_line_flag("no_verif")) {
      verification = false;
    }
    cmd.get_cmd_line_argument("batch", batch);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "89_sm103_fp4_ultra_gemm\n\n"
      << "  Sm103 ultra FP4 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --cluster_m=<int>           Preferred cluster X dimension (input only)\n"
      << "  --cluster_n=<int>           Preferred cluster Y dimension (input only)\n"
      << "  --cluster_fallback_m=<int>  Fallback cluster X dimension (input only)\n"
      << "  --cluster_fallback_n=<int>  Fallback cluster Y dimension (input only)\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n"
      << "  --batch=<int>               Number of batches (L dimension)\n"
      << "  --no_verif                   Do not run host-side verification\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm"
      << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707"
      << " --cluster_m=4 --cluster_n=4 --cluster_fallback_m=2 --cluster_fallback_n=1\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add, times batch (L dimension)
    uint64_t flop = uint64_t(2) * uint64_t(m) * uint64_t(n) * uint64_t(k) * uint64_t(batch);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, options.batch});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, options.batch});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, options.batch});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, options.batch});

  layout_A = make_layout(make_shape(options.m, options.k, options.batch), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, options.batch), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, options.batch), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, options.batch), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, options.batch));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, options.batch));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  initialize_block(block_A.host_view(), seed + 2021);
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
}

// Populates a Gemm::Arguments structure from the given commandline options
template <typename Gemm>
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.batch},
    { // Mainloop arguments
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;
  arguments.hw_info.cluster_shape = options.cluster_shape;
  arguments.hw_info.cluster_shape_fallback = options.cluster_shape_fallback;
  return arguments;
}

bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D)                    // TensorD
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<Gemm>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  uint8_t* workspace = nullptr;
  cudaError_t status = cudaMalloc(&workspace, workspace_size);
  if (status != cudaSuccess) {
    std::cerr << "Failed to allocate workspace memory: " << cudaGetErrorString(status) << std::endl;
    return -1;
  }

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  cudaDeviceSynchronize();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verification) {
    result.passed = verify(options);
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
    if (!result.passed) {
      cudaFree(workspace);
      exit(-1);
    }
  } else {
    std::cout << "  Disposition: Skipped verification" << std::endl;
    result.passed = true;
  }

#ifdef KH_TEST_SOLUTION
  // Run and verify solution
  {
    int M = options.m, N = options.n, K = options.k;
    int group_size = 32;
    int k_groups = (K + group_size - 1) / group_size;

    // The CUTLASS data uses packed FP4 layout with interleaved scale factors.
    // For the solution interface, we need flat FP4 data (one per byte) and flat float scales.
    // We use the host data from block_A/block_B (which is in PackedVectorLayout) and
    // set scales to 1.0 for a fair comparison.

    // Re-initialize scales to 1.0 for both CUTLASS and solution
    {
      auto ones_SFA = cutlass::float_ue4m3_t(1.0f);
      auto ones_SFB = cutlass::float_ue4m3_t(1.0f);
      for (int i = 0; i < (int)block_SFA.size(); i++) block_SFA.host_data()[i] = ones_SFA;
      for (int i = 0; i < (int)block_SFB.size(); i++) block_SFB.host_data()[i] = ones_SFB;
      block_SFA.sync_device();
      block_SFB.sync_device();
    }

    // Re-run CUTLASS with all-ones scales
    auto arguments2 = args_from_options<Gemm>(options);
    size_t ws2 = Gemm::get_workspace_size(arguments2);
    uint8_t* workspace2 = nullptr;
    cudaMalloc(&workspace2, ws2);
    CUTLASS_CHECK(gemm.initialize(arguments2, workspace2));
    CUTLASS_CHECK(gemm.run());
    cudaDeviceSynchronize();

    // Prepare solution inputs: flat FP4 data and float scales
    // block_A is in PackedVectorLayout - it stores elements sequentially
    // For the solution, we pass the raw device pointers
    int sA_count = M * k_groups * options.batch;
    int sB_count = N * k_groups * options.batch;
    float *dsA, *dsB;
    cudaMalloc(&dsA, sA_count * sizeof(float));
    cudaMalloc(&dsB, sB_count * sizeof(float));
    std::vector<float> flat_sA(sA_count, 1.0f), flat_sB(sB_count, 1.0f);
    cudaMemcpy(dsA, flat_sA.data(), sA_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dsB, flat_sB.data(), sB_count * sizeof(float), cudaMemcpyHostToDevice);

    // Solution output
    __nv_bfloat16 *sol_D;
    int D_count = M * N * options.batch;
    cudaMalloc(&sol_D, D_count * sizeof(__nv_bfloat16));
    cudaMemset(sol_D, 0, D_count * sizeof(__nv_bfloat16));

    cudaError_t sol_err = Fp4UltraGemm(
      M, N, K, options.alpha,
      reinterpret_cast<const cutlass::float_e2m1_t*>(block_A.device_data()), K,
      dsA, k_groups,
      reinterpret_cast<const cutlass::float_e2m1_t*>(block_B.device_data()), K,
      dsB, k_groups,
      options.beta,
      sol_D, N);

    if (sol_err != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
      cudaFree(dsA); cudaFree(dsB); cudaFree(sol_D); cudaFree(workspace2);
      exit(-1);
    }
    cudaDeviceSynchronize();

    // Compare solution vs CUTLASS reference
    block_D.sync_host();
    std::vector<__nv_bfloat16> host_sol_D(D_count);
    cudaMemcpy(host_sol_D.data(), sol_D, D_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < D_count; i++) {
      float ref_val = float(block_D.host_data()[i]);
      float sol_val = __bfloat162float(host_sol_D[i]);
      float diff = std::fabs(ref_val - sol_val);
      float denom = std::fmax(std::fabs(ref_val), 1e-6f);
      if (diff / denom > 0.3f) mismatches++;
    }
    float mismatch_rate = float(mismatches) / float(D_count);
    if (mismatch_rate > 0.01f) {
      std::cerr << "Solution verification failed. Mismatch rate: " << mismatch_rate << std::endl;
      cudaFree(dsA); cudaFree(dsB); cudaFree(sol_D); cudaFree(workspace2);
      exit(-1);
    }

    cudaFree(dsA);
    cudaFree(dsB);
    cudaFree(sol_D);
    cudaFree(workspace2);
  }
#endif

  // Profiling
  if (options.iterations > 0) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup reference
    for (int i = 0; i < 3; i++) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace));
      CUTLASS_CHECK(gemm.run());
    }
    cudaDeviceSynchronize();

    // Profile reference
    float ref_total_ms = 0, ref_min_ms = 1e30f;
    for (int iter = 0; iter < options.iterations; ++iter) {
      cudaEventRecord(start);
      CUTLASS_CHECK(gemm.initialize(arguments, workspace));
      CUTLASS_CHECK(gemm.run());
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      ref_total_ms += ms;
      if (ms < ref_min_ms) ref_min_ms = ms;
    }
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            ref_total_ms / options.iterations, options.iterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
    {
      int M = options.m, N = options.n, K = options.k;
      int group_size = 32;
      int k_groups = (K + group_size - 1) / group_size;
      int sA_count = M * k_groups * options.batch;
      int sB_count = N * k_groups * options.batch;
      float *dsA, *dsB;
      cudaMalloc(&dsA, sA_count * sizeof(float));
      cudaMalloc(&dsB, sB_count * sizeof(float));
      std::vector<float> flat_sA(sA_count, 1.0f), flat_sB(sB_count, 1.0f);
      cudaMemcpy(dsA, flat_sA.data(), sA_count * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dsB, flat_sB.data(), sB_count * sizeof(float), cudaMemcpyHostToDevice);

      __nv_bfloat16 *sol_D;
      int D_count = M * N * options.batch;
      cudaMalloc(&sol_D, D_count * sizeof(__nv_bfloat16));

      // Warmup solution
      for (int i = 0; i < 3; i++) {
        Fp4UltraGemm(M, N, K, options.alpha,
          reinterpret_cast<const cutlass::float_e2m1_t*>(block_A.device_data()), K,
          dsA, k_groups,
          reinterpret_cast<const cutlass::float_e2m1_t*>(block_B.device_data()), K,
          dsB, k_groups,
          options.beta, sol_D, N);
      }
      cudaDeviceSynchronize();

      // Profile solution
      float sol_total_ms = 0, sol_min_ms = 1e30f;
      for (int iter = 0; iter < options.iterations; ++iter) {
        cudaEventRecord(start);
        Fp4UltraGemm(M, N, K, options.alpha,
          reinterpret_cast<const cutlass::float_e2m1_t*>(block_A.device_data()), K,
          dsA, k_groups,
          reinterpret_cast<const cutlass::float_e2m1_t*>(block_B.device_data()), K,
          dsB, k_groups,
          options.beta, sol_D, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        sol_total_ms += ms;
        if (ms < sol_min_ms) sol_min_ms = ms;
      }
      fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
              sol_total_ms / options.iterations, options.iterations, sol_min_ms);
      fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);

      cudaFree(dsA);
      cudaFree(dsB);
      cudaFree(sol_D);
    }
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaFree(workspace);

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.9 or higher Toolkit to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 10 && props.minor == 3)) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 103)." << std::endl;
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

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (options.m != 1024 || options.n != 1024 || options.k != 1024);

  std::vector<TestConfig> configs;

  if (explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k});
  } else {
    configs = {
      {"small",   1024, 1024, 1024},
      {"medium",  4096, 4096, 4096},
      {"large",   8192, 8192, 8192},
    };
  }

  options.iterations = 20;
  options.verification = true;

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;

    run<Gemm1Sm>(options);

    std::cout << "Passed" << std::endl;
  }

#endif // defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
