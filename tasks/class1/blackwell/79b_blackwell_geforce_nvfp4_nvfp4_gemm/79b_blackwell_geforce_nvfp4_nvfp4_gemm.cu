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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM120 architecture.

    This example demonstrates a simple way to instantiate and run a blockscaled NVFP4 GEMM on the NVIDIA Blackwell SM120 architecture.
    The kernel outputs quantized fp4 values with scale factors that will be the input of another GEMM.
    This kernel is optimized for the GeForce RTX 50 series GPUs.

    Similar to 79a_blackwell_geforce_nvfp4_bf16_gemm, this kernel leverages:

    1. Warp-Specialized persistent kernel design that supports both cooperative and ping-pong kernel schedule introduced in Hopper.
    2. The new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).
    3. Block Scaled Tensor Core MMA Instructions
    4. Epilogue Optimization

    Note that GeForce RTX 50 series GPUs do not support:
    1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
    2. Dynamic datatypes.

    Usage:

      $ ./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm --m=2048 --n=2048 --k=2048
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

#include "cutlass/cutlass.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

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

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::float_e2m1_t;                          // Element type for D matrix operand
using         ElementSFD  = cutlass::float_ue8m0_t;                         // Element type for SFD matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
using         LayoutSFDTag = LayoutDTag;                                    // Layout type for SFD should be same as D matrix operand

constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

constexpr int InputSFVectorSize  = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// D = alpha * acc + beta * C
//      With BlockScaleFactor generation.
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,                      // Epilogue schedule policy
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                           // Ping-pong kernel schedule policy.
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

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

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

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
LayoutSFD layout_SFD;

uint64_t seed;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_SFD;

// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_reference_SFD;
// Matrix-wide normalization constant
cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> block_Normconst;

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

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

  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10)
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
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "79b_blackwell_geforce_nvfp4_nvfp4_gemm\n\n"
      << "  Blackwell NVFP4 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
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

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

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
  // For SFD tensor layout
  using Sm1xxBlockScaledOutputConfig=  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(options.m, options.n, options.k, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  block_Normconst.reset(cutlass::make_Coord(1));

  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

  initialize_block(block_A.host_view(), seed + 2021);
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);
  block_Normconst.at(cutlass::make_Coord(0)) = 2;

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_SFD.sync_device();
  block_Normconst.sync_device();
}

// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
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

  if constexpr (IsBlockScaleSupported) {
    arguments.epilogue.thread.block_scale_factor_ptr = block_SFD.device_data();
    arguments.epilogue.thread.norm_constant_ptr      = block_Normconst.device_data();
  }

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
  auto tensor_SFD = make_tensor(block_reference_SFD.host_data(), layout_SFD);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D),                   // TensorD
      decltype(tensor_SFD),                 // TensorSfD
      cute::Int<OutputSFVectorSize>,
      cutlass::reference::host::SfStrategy::SfDGen
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D, tensor_SFD, block_Normconst.at(cutlass::make_Coord(0))};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}

/// Run CUTLASS GEMM
cutlass::Status run_cutlass_gemm(Options &options) {
  Gemm gemm;
  auto arguments = args_from_options(options);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  return cutlass::Status::kSuccess;
}

/// Test correctness and profile for a given size
int test_and_profile(Options &options) {
  initialize(options);

  // Run CUTLASS reference
  cutlass::Status status = run_cutlass_gemm(options);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM failed." << std::endl;
    return -1;
  }

  // Verify CUTLASS reference against host reference
  if (!verify(options)) {
    std::cerr << "CUTLASS reference incorrect vs host reference." << std::endl;
    return -1;
  }

#ifdef KH_TEST_SOLUTION
  // Generate simple test data independently for solution verification
  // A: row-major [M,K], B: col-major [K,N], D: row-major [M,N] in FP4
  int M = options.m, N = options.n, K = options.k;
  int lda = K, ldb = K, ldd = N;
  static constexpr int kGroupSize_ = 32;
  static constexpr int kOutputSFVec_ = 16;
  int k_groups = (K + kGroupSize_ - 1) / kGroupSize_;
  int n_sf_groups = (N + kOutputSFVec_ - 1) / kOutputSFVec_;

  // Generate host data
  std::vector<cutlass::float_e2m1_t> hA(M * K), hB(K * N);
  std::vector<float> fA(M * K), fB(K * N);
  std::vector<float> sA(M * k_groups, 1.0f), sB(N * k_groups, 1.0f);

  for (int i = 0; i < M * K; ++i) {
    hA[i] = cutlass::float_e2m1_t(float((i * 16807 % 6) - 3) * 0.5f);
    fA[i] = float(hA[i]);
  }
  for (int i = 0; i < K * N; ++i) {
    hB[i] = cutlass::float_e2m1_t(float((i * 48271 % 6) - 3) * 0.5f);
    fB[i] = float(hB[i]);
  }

  // Allocate device memory
  cutlass::float_e2m1_t *dA, *dB, *dD;
  float *dsA, *dsB, *dsD;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(cutlass::float_e2m1_t)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(cutlass::float_e2m1_t)));
  CUDA_CHECK(cudaMalloc(&dsA, M * k_groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dsB, N * k_groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, M * N * sizeof(cutlass::float_e2m1_t)));
  CUDA_CHECK(cudaMalloc(&dsD, M * n_sf_groups * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(cutlass::float_e2m1_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(cutlass::float_e2m1_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsA, sA.data(), M * k_groups * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsB, sB.data(), N * k_groups * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD, 0, M * N * sizeof(cutlass::float_e2m1_t)));
  CUDA_CHECK(cudaMemset(dsD, 0, M * n_sf_groups * sizeof(float)));

  cudaError_t sol_err = Nvfp4Nvfp4Gemm(
    M, N, K, options.alpha, dA, lda, dsA, k_groups, dB, ldb, dsB, k_groups,
    options.beta, dD, ldd, dsD, n_sf_groups);

  if (sol_err != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
    cudaFree(dA); cudaFree(dB); cudaFree(dsA); cudaFree(dsB); cudaFree(dD); cudaFree(dsD);
    return -1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // CPU reference
  std::vector<float> fRef(M * N, 0);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) {
        int kg = k / kGroupSize_;
        float a_val = fA[i * lda + k] * sA[i * k_groups + kg];
        float b_val = fB[k + j * ldb] * sB[j * k_groups + kg];
        acc += a_val * b_val;
      }
      fRef[i * ldd + j] = options.alpha * acc;
    }

  // Dequantize output: D_float[i,j] = D_fp4[i,j] * D_scales[i, j/kOutputSFVec]
  std::vector<cutlass::float_e2m1_t> hD(M * N);
  std::vector<float> hDscales(M * n_sf_groups);
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, M * N * sizeof(cutlass::float_e2m1_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hDscales.data(), dsD, M * n_sf_groups * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> fD(M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      int sg = j / kOutputSFVec_;
      fD[i * ldd + j] = float(hD[i * ldd + j]) * hDscales[i * n_sf_groups + sg];
    }

  float max_rel_diff = 0; int mismatches = 0;
  for (int idx = 0; idx < M * N; ++idx) {
    float diff = std::fabs(fD[idx] - fRef[idx]);
    float denom = std::fmax(std::fabs(fRef[idx]), 1e-6f);
    float rel = diff / denom;
    max_rel_diff = std::fmax(max_rel_diff, rel);
    if (rel > 0.5f) mismatches++;  // More relaxed for double quantization
  }
  float mismatch_rate = float(mismatches) / float(M * N);
  if (mismatch_rate > 0.05f) {
    std::cerr << "Solution incorrect. Max relative diff: " << max_rel_diff
              << ", mismatch rate: " << mismatch_rate << std::endl;
    cudaFree(dA); cudaFree(dB); cudaFree(dsA); cudaFree(dsB); cudaFree(dD); cudaFree(dsD);
    return -1;
  }
#endif

  std::cout << "Passed" << std::endl;

  // Profile
  if (options.iterations > 0) {
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warmup reference
    for (int i = 0; i < 3; i++) run_cutlass_gemm(options);

    // Profile reference (CUTLASS)
    float ref_total_ms = 0, ref_min_ms = 1e30f;
    for (int iter = 0; iter < options.iterations; ++iter) {
      cudaEventRecord(ev_start);
      run_cutlass_gemm(options);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);
      float ms;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      ref_total_ms += ms;
      if (ms < ref_min_ms) ref_min_ms = ms;
    }
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            ref_total_ms / options.iterations, options.iterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
    // Warmup solution
    for (int i = 0; i < 3; i++) {
      Nvfp4Nvfp4Gemm(M, N, K, options.alpha,
        dA, lda, dsA, k_groups, dB, ldb, dsB, k_groups,
        options.beta, dD, ldd, dsD, n_sf_groups);
      cudaDeviceSynchronize();
    }

    // Profile solution
    float sol_total_ms = 0, sol_min_ms = 1e30f;
    for (int iter = 0; iter < options.iterations; ++iter) {
      cudaEventRecord(ev_start);
      Nvfp4Nvfp4Gemm(M, N, K, options.alpha,
        dA, lda, dsA, k_groups, dB, ldb, dsB, k_groups,
        options.beta, dD, ldd, dsD, n_sf_groups);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);
      float ms;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      sol_total_ms += ms;
      if (ms < sol_min_ms) sol_min_ms = ms;
    }
    fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            sol_total_ms / options.iterations, options.iterations, sol_min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dsA); cudaFree(dsB); cudaFree(dD); cudaFree(dsD);
#endif

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
  }
#ifdef KH_TEST_SOLUTION
  else {
    cudaFree(dA); cudaFree(dB); cudaFree(dsA); cudaFree(dsB); cudaFree(dD); cudaFree(dsD);
  }
#endif

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support." << std::endl;
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support." << std::endl;
    return 0;
  }
#endif

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 120 or 121)." << std::endl;
    return 0;
  }

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = false;
  {
    cutlass::CommandLine cmd(argc, args);
    explicit_size = cmd.check_cmd_line_flag("m") ||
                    cmd.check_cmd_line_flag("n") ||
                    cmd.check_cmd_line_flag("k");
  }

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

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);
    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;

    int ret = test_and_profile(options);
    if (ret != 0) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
