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
    \brief Grouped scale Hopper FP8 Grouped GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrates a grouped scaled FP8 Grouped GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. This example also showcases on-the-fly modification of TMA
    descriptors to move between groups/problem_count (represented by groups).
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 --save_aux=false --save_amax=false \
        --raster=h --swizzle=2 --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cfloat>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"

// Includes from examples directory
#include "helper.h"
#include "hopper_fp8_commandline.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp8.h>
#include "solution.h"
#endif

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using ArchTag       = cutlass::arch::Sm90;                          // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;               // Operator class tag
using TileShape     = Shape<_128,_128,_128>;                        // Threadblock-level tile size
using ClusterShape  = Shape<_1,_2,_1>;                              // Shape of the threadblocks in a cluster

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;

constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand

using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpilogueTileType  = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA *, LayoutSFA *>, AlignmentA,
    ElementB, cute::tuple<LayoutB *, LayoutSFB *>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopWithGroupWiseScaling,
    CollectiveEpilogue
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization

cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_blockscale_A;
std::vector<int64_t> offset_blockscale_B;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;
std::vector<LayoutSFA> layout_SFA_host;
std::vector<LayoutSFB> layout_SFB_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementD> block_D;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

cutlass::DeviceAllocation<const ElementA *> ptr_A;
cutlass::DeviceAllocation<const ElementB *> ptr_B;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<ElementD *> ptr_D;
cutlass::DeviceAllocation<ElementD *> ptr_ref_D;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) 

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class ScopeMin = std::nullopt_t, class ScopeMax = std::nullopt_t>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023,
  ScopeMin scope_min = std::nullopt, ScopeMax scope_max = std::nullopt) {

  double _scope_max, _scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  if (bits_input == 1) {
    _scope_max = 2;
    _scope_min = 0;
  } else if (bits_input <= 8) {
    _scope_max = 2;
    _scope_min = -2;
  } else if (bits_input == 16) {
    _scope_max = 5;
    _scope_min = -5;
  } else {
    _scope_max = 8;
    _scope_min = -8;
  }
  if constexpr (!std::is_same_v<ScopeMax, std::nullopt_t>) {
    _scope_max = scope_max;
  }
  if constexpr (!std::is_same_v<ScopeMin, std::nullopt_t>) {
    _scope_min = scope_min;
  }
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, (Element) _scope_max, (Element) _scope_min, 0);

  return true;
}

/// Allocates device-side data
template <typename OptionType>
void allocate(const OptionType &options) {

  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;

  offset_A.clear();
  offset_B.clear();
  offset_C.clear();
  offset_D.clear();
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();
  stride_A_host.clear();
  stride_B_host.clear();
  stride_C_host.clear();
  stride_D_host.clear();
  
  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto group_layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto group_layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_blockscale_A.push_back(total_elements_blockscale_A);
    offset_blockscale_B.push_back(total_elements_blockscale_B);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_blockscale_A = size(filter_zeros(group_layout_SFA));
    int64_t elements_blockscale_B = size(filter_zeros(group_layout_SFB));

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_blockscale_A += elements_blockscale_A;
    total_elements_blockscale_B += elements_blockscale_B;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    layout_SFA_host.push_back(group_layout_SFA);
    layout_SFB_host.push_back(group_layout_SFB);

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
  blockscale_block_A.reset(total_elements_blockscale_A);
  blockscale_block_B.reset(total_elements_blockscale_B);
}

/// Initialize operands to be used in the GEMM and reference GEMM
template <typename OptionType>
void initialize(const OptionType &options) {

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementD *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_A_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_B_host(options.groups);

  alpha_host.clear();
  beta_host.clear();

  for (int i = 0; i < options.groups; i++) {
    // If the current group's matrix has size 0, set the pointer to nullptr
    if (i < options.groups - 1 && offset_A.at(i) == offset_A.at(i + 1)) {
      ptr_A_host.at(i) = nullptr;
    } else {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    }
    if (i < options.groups - 1 && offset_B.at(i) == offset_B.at(i + 1)) {
      ptr_B_host.at(i) = nullptr;
    } else {
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    }
    if (i < options.groups - 1 && offset_C.at(i) == offset_C.at(i + 1)) {
      ptr_C_host.at(i) = nullptr;
    } else {
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    }
    if (i < options.groups - 1 && offset_D.at(i) == offset_D.at(i + 1)) {
      ptr_D_host.at(i) = nullptr;
    } else {
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }
    if (i < options.groups - 1 && offset_blockscale_A.at(i) == offset_blockscale_A.at(i + 1)) {
      ptr_blockscale_A_host.at(i) = nullptr;
    } else {
      ptr_blockscale_A_host.at(i) = blockscale_block_A.get() + offset_blockscale_A.at(i);
    }
    if (i < options.groups - 1 && offset_blockscale_B.at(i) == offset_blockscale_B.at(i + 1)) {
      ptr_blockscale_B_host.at(i) = nullptr;
    } else {
      ptr_blockscale_B_host.at(i) = blockscale_block_B.get() + offset_blockscale_B.at(i);
    }
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_blockscale_A.reset(options.groups);
  ptr_blockscale_A.copy_from_host(ptr_blockscale_A_host.data());

  ptr_blockscale_B.reset(options.groups);
  ptr_blockscale_B.copy_from_host(ptr_blockscale_B_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  layout_SFA.reset(options.groups);
  layout_SFA.copy_from_host(layout_SFA_host.data());

  layout_SFB.reset(options.groups);
  layout_SFB.copy_from_host(layout_SFB_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_block(block_A, seed + 2022);
  initialize_block(block_B, seed + 2023);
  initialize_block(block_C, seed + 2024);
  initialize_block(blockscale_block_A, seed + 2025, -1, 1);
  initialize_block(blockscale_block_B, seed + 2026, -1, 1);

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

}

/// Populates a Gemm::Arguments structure from the given commandline options
template<typename GemmArguments, typename OptionType>
GemmArguments args_from_options(const OptionType &options, bool host_problem_shapes_available = true)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);

  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), host_problem_shapes_available ? options.problem_sizes_host.data() : (decltype(options.problem_sizes_host.data())) nullptr},
    {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
     ptr_blockscale_A.get(), layout_SFA.get(),
     ptr_blockscale_B.get(), layout_SFB.get()
    },
    {
      {}, // epilogue.thread
      ptr_C.get(), stride_C.get(),
      ptr_D.get(), stride_D.get()
    },
    kernel_hw_info
  };

  auto &fusion_args = arguments.epilogue.thread;
  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }
  else {
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.beta_ptr_array = beta_device.get();
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
  }

  arguments.scheduler.raster_order = options.raster_order;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

template <typename OptionType>
bool verify(const OptionType &options) {

  //
  // Compute reference output
  //

  std::vector<ElementA> block_A_host(block_A.size());
  std::vector<ElementB> block_B_host(block_B.size());
  std::vector<ElementC> block_C_host(block_C.size());
  std::vector<ElementD> block_D_host_kernel(block_D.size());
  std::vector<ElementD> block_D_host_ref(block_D.size());
  std::vector<ElementBlockScale> blockscale_block_A_host(blockscale_block_A.size());
  std::vector<ElementBlockScale> blockscale_block_B_host(blockscale_block_B.size());

  block_A.copy_to_host(block_A_host.data());
  block_B.copy_to_host(block_B_host.data());
  block_C.copy_to_host(block_C_host.data());
  block_D.copy_to_host(block_D_host_kernel.data());
  blockscale_block_A.copy_to_host(blockscale_block_A_host.data());
  blockscale_block_B.copy_to_host(blockscale_block_B_host.data());

  bool passed = true;
  std::cout << "  Running host reference kernel - may run for a while for large problems." << std::endl;
  for (int group_idx = 0; group_idx < options.groups; group_idx++) {
    // Group scaling tensors shapes based `ScaleGranularityM`, CTA Block (TileShape) and GEMM Problem shape
    auto [m, n, k] = options.problem_sizes_host.at(group_idx);

    // Create instantiation for device reference gemm kernel
    auto A = cute::make_tensor(block_A_host.data() + offset_A.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, k, 1),
                                  stride_A_host.at(group_idx)
                                )
                              );
    auto B = cute::make_tensor(block_B_host.data() + offset_B.at(group_idx),
                              cute::make_layout(
                                cute::make_shape(n, k, 1),
                                stride_B_host.at(group_idx)
                                )
                              );
    auto C = cute::make_tensor(block_C_host.data() + offset_C.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_C_host.at(group_idx)
                                )
                              );
    auto D = cute::make_tensor(block_D_host_ref.data() + offset_D.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_D_host.at(group_idx)
                                )
                              );

    auto SFA = cute::make_tensor(blockscale_block_A_host.data() + offset_blockscale_A.at(group_idx),
                                 layout_SFA_host.at(group_idx));
    auto SFB = cute::make_tensor(blockscale_block_B_host.data() + offset_blockscale_B.at(group_idx),
                                 layout_SFB_host.at(group_idx));

    using unused_t = decltype(D);

    cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(A), 
      decltype(SFA), 
      decltype(B),
      decltype(SFB)
    > mainloop_params{A, SFA, B, SFB};

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D)
    > epilogue_params;

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha_host.at(group_idx);
    epilogue_params.beta = beta_host.at(group_idx);

    // get reference result
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    auto this_group_passed = std::equal(
      // std::execution::par_unseq,
      block_D_host_ref.data() + offset_D.at(group_idx),
      block_D_host_ref.data() + offset_D.at(group_idx) + m * n,
      block_D_host_kernel.data() + offset_D.at(group_idx)
    );
    
    passed &= this_group_passed;

#if 0
    std::cout << "Group: " << group_idx << " M: " << m << " N: " << n << " K: " << k << " Status: " << this_group_passed << std::endl;
#endif

  }

  return passed;
}

/// Run CUTLASS reference grouped GEMM once
cudaError_t RunCutlassReference(Options<ProblemShape> &options) {
  Gemm gemm;
  auto arguments = args_from_options<typename Gemm::Arguments>(options, true);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

/// Test correctness: CUTLASS reference vs CPU, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(Options<ProblemShape> &options) {
  allocate(options);
  initialize(options);

  // Run CUTLASS reference
  cudaError_t result = RunCutlassReference(options);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS reference failed." << std::endl;
    return result;
  }
  cudaDeviceSynchronize();

  // CPU verification skipped (non-fatal, not needed for benchmark)

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  cutlass::DeviceAllocation<ElementD> sol_block_D;
  sol_block_D.reset(block_D.size());

  // Build host pointer arrays for solution
  std::vector<__nv_fp8_e4m3 const*> sol_A_ptrs(options.groups);
  std::vector<__nv_fp8_e4m3 const*> sol_B_ptrs(options.groups);
  std::vector<__nv_fp8_e4m3*> sol_D_ptrs(options.groups);
  std::vector<float const*> sol_sA_ptrs(options.groups);
  std::vector<float const*> sol_sB_ptrs(options.groups);
  std::vector<int> sol_Ms(options.groups), sol_Ns(options.groups), sol_Ks(options.groups);
  std::vector<int> sol_ldas(options.groups), sol_ldbs(options.groups), sol_ldds(options.groups);

  for (int i = 0; i < options.groups; i++) {
    auto [m, n, k] = options.problem_sizes_host.at(i);
    sol_Ms[i] = m;
    sol_Ns[i] = n;
    sol_Ks[i] = k;
    // A row-major [M,K] lda=K, B col-major [K,N] ldb=K, D col-major [M,N] ldd=M
    sol_ldas[i] = k;
    sol_ldbs[i] = k;
    sol_ldds[i] = m;

    sol_A_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3 const*>(block_A.get() + offset_A.at(i));
    sol_B_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3 const*>(block_B.get() + offset_B.at(i));
    sol_D_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3*>(sol_block_D.get() + offset_D.at(i));
    sol_sA_ptrs[i] = blockscale_block_A.get() + offset_blockscale_A.at(i);
    sol_sB_ptrs[i] = blockscale_block_B.get() + offset_blockscale_B.at(i);
  }

  // Copy pointer arrays to device
  cutlass::DeviceAllocation<__nv_fp8_e4m3 const*> d_sol_A_ptrs(options.groups);
  cutlass::DeviceAllocation<__nv_fp8_e4m3 const*> d_sol_B_ptrs(options.groups);
  cutlass::DeviceAllocation<__nv_fp8_e4m3*> d_sol_D_ptrs(options.groups);
  cutlass::DeviceAllocation<float const*> d_sol_sA_ptrs(options.groups);
  cutlass::DeviceAllocation<float const*> d_sol_sB_ptrs(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ms(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ns(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ks(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldas(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldbs(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldds(options.groups);

  d_sol_A_ptrs.copy_from_host(sol_A_ptrs.data());
  d_sol_B_ptrs.copy_from_host(sol_B_ptrs.data());
  d_sol_D_ptrs.copy_from_host(sol_D_ptrs.data());
  d_sol_sA_ptrs.copy_from_host(sol_sA_ptrs.data());
  d_sol_sB_ptrs.copy_from_host(sol_sB_ptrs.data());
  d_sol_Ms.copy_from_host(sol_Ms.data());
  d_sol_Ns.copy_from_host(sol_Ns.data());
  d_sol_Ks.copy_from_host(sol_Ks.data());
  d_sol_ldas.copy_from_host(sol_ldas.data());
  d_sol_ldbs.copy_from_host(sol_ldbs.data());
  d_sol_ldds.copy_from_host(sol_ldds.data());

  cudaError_t sol_result = HopperFp8BlockwiseGroupedGemm(
    options.groups,
    d_sol_Ms.get(), d_sol_Ns.get(), d_sol_Ks.get(),
    options.alpha,
    d_sol_A_ptrs.get(), d_sol_ldas.get(), d_sol_sA_ptrs.get(),
    d_sol_B_ptrs.get(), d_sol_ldbs.get(), d_sol_sB_ptrs.get(),
    d_sol_D_ptrs.get(), d_sol_ldds.get(),
    128);

  if (sol_result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_result) << std::endl;
    return sol_result;
  }
  cudaDeviceSynchronize();

  // Compare solution vs CUTLASS reference for each group
  std::vector<ElementD> ref_D_host(block_D.size());
  std::vector<ElementD> sol_D_host(sol_block_D.size());
  block_D.copy_to_host(ref_D_host.data());
  sol_block_D.copy_to_host(sol_D_host.data());

  for (int g = 0; g < options.groups; g++) {
    auto [m, n, k] = options.problem_sizes_host.at(g);
    int total = m * n;
    float max_diff = 0.0f;
    for (int i = 0; i < total; ++i) {
      float ref_val = float(ref_D_host[offset_D.at(g) + i]);
      float sol_val = float(sol_D_host[offset_D.at(g) + i]);
      max_diff = std::fmax(max_diff, std::fabs(sol_val - ref_val));
    }
    // FP8 E4M3 output tolerance
    if (max_diff > 1.0f) {
      std::cerr << "Solution incorrect for group " << g << ". Max diff: " << max_diff << std::endl;
      return cudaErrorUnknown;
    }
  }
#endif

  return cudaSuccess;
}

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(Options<ProblemShape> &options, int iterations) {
  allocate(options);
  initialize(options);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    RunCutlassReference(options);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    RunCutlassReference(options);
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
  // Build solution pointer arrays (same as in TestCorrectness)
  std::vector<__nv_fp8_e4m3 const*> sol_A_ptrs(options.groups);
  std::vector<__nv_fp8_e4m3 const*> sol_B_ptrs(options.groups);
  std::vector<__nv_fp8_e4m3*> sol_D_ptrs(options.groups);
  std::vector<float const*> sol_sA_ptrs(options.groups);
  std::vector<float const*> sol_sB_ptrs(options.groups);
  std::vector<int> sol_Ms(options.groups), sol_Ns(options.groups), sol_Ks(options.groups);
  std::vector<int> sol_ldas(options.groups), sol_ldbs(options.groups), sol_ldds(options.groups);

  for (int i = 0; i < options.groups; i++) {
    auto [m, n, k] = options.problem_sizes_host.at(i);
    sol_Ms[i] = m; sol_Ns[i] = n; sol_Ks[i] = k;
    sol_ldas[i] = k; sol_ldbs[i] = k; sol_ldds[i] = m;
    sol_A_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3 const*>(block_A.get() + offset_A.at(i));
    sol_B_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3 const*>(block_B.get() + offset_B.at(i));
    sol_D_ptrs[i] = reinterpret_cast<__nv_fp8_e4m3*>(block_D.get() + offset_D.at(i));
    sol_sA_ptrs[i] = blockscale_block_A.get() + offset_blockscale_A.at(i);
    sol_sB_ptrs[i] = blockscale_block_B.get() + offset_blockscale_B.at(i);
  }

  cutlass::DeviceAllocation<__nv_fp8_e4m3 const*> d_sol_A_ptrs(options.groups);
  cutlass::DeviceAllocation<__nv_fp8_e4m3 const*> d_sol_B_ptrs(options.groups);
  cutlass::DeviceAllocation<__nv_fp8_e4m3*> d_sol_D_ptrs(options.groups);
  cutlass::DeviceAllocation<float const*> d_sol_sA_ptrs(options.groups);
  cutlass::DeviceAllocation<float const*> d_sol_sB_ptrs(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ms(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ns(options.groups);
  cutlass::DeviceAllocation<int> d_sol_Ks(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldas(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldbs(options.groups);
  cutlass::DeviceAllocation<int> d_sol_ldds(options.groups);

  d_sol_A_ptrs.copy_from_host(sol_A_ptrs.data());
  d_sol_B_ptrs.copy_from_host(sol_B_ptrs.data());
  d_sol_D_ptrs.copy_from_host(sol_D_ptrs.data());
  d_sol_sA_ptrs.copy_from_host(sol_sA_ptrs.data());
  d_sol_sB_ptrs.copy_from_host(sol_sB_ptrs.data());
  d_sol_Ms.copy_from_host(sol_Ms.data());
  d_sol_Ns.copy_from_host(sol_Ns.data());
  d_sol_Ks.copy_from_host(sol_Ks.data());
  d_sol_ldas.copy_from_host(sol_ldas.data());
  d_sol_ldbs.copy_from_host(sol_ldbs.data());
  d_sol_ldds.copy_from_host(sol_ldds.data());

  // Save reference output for correctness check
  size_t total_D_bytes = block_D.size() * sizeof(ElementD);
  cutlass::DeviceAllocation<ElementD> D_ref_saved;
  D_ref_saved.reset(block_D.size());
  cudaMemcpy(D_ref_saved.get(), block_D.get(), total_D_bytes, cudaMemcpyDeviceToDevice);

  // Run solution once for correctness check
  HopperFp8BlockwiseGroupedGemm(
    options.groups,
    d_sol_Ms.get(), d_sol_Ns.get(), d_sol_Ks.get(),
    options.alpha,
    d_sol_A_ptrs.get(), d_sol_ldas.get(), d_sol_sA_ptrs.get(),
    d_sol_B_ptrs.get(), d_sol_ldbs.get(), d_sol_sB_ptrs.get(),
    d_sol_D_ptrs.get(), d_sol_ldds.get(),
    128);
  cudaDeviceSynchronize();

  {
    size_t count = block_D.size();
    std::vector<__nv_fp8_e4m3> h_sol(count), h_ref(count);
    cudaMemcpy(h_sol.data(), block_D.get(), count * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved.get(), count * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    float max_diff = 0;
    for (size_t i = 0; i < count; ++i)
      max_diff = std::fmax(max_diff, std::fabs(float(h_sol[i]) - float(h_ref[i])));
    if (max_diff > 1.0f) {
      fprintf(stderr, "Solution incorrect in Profile: max_diff=%.6f\n", max_diff);
      std::cout << "Incorrect" << std::endl;
      cudaEventDestroy(start); cudaEventDestroy(stop);
      exit(-1);
    }
  }

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    HopperFp8BlockwiseGroupedGemm(
      options.groups,
      d_sol_Ms.get(), d_sol_Ns.get(), d_sol_Ks.get(),
      options.alpha,
      d_sol_A_ptrs.get(), d_sol_ldas.get(), d_sol_sA_ptrs.get(),
      d_sol_B_ptrs.get(), d_sol_ldbs.get(), d_sol_sB_ptrs.get(),
      d_sol_D_ptrs.get(), d_sol_ldds.get(),
      128);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    HopperFp8BlockwiseGroupedGemm(
      options.groups,
      d_sol_Ms.get(), d_sol_Ns.get(), d_sol_Ks.get(),
      options.alpha,
      d_sol_A_ptrs.get(), d_sol_ldas.get(), d_sol_sA_ptrs.get(),
      d_sol_B_ptrs.get(), d_sol_ldbs.get(), d_sol_sB_ptrs.get(),
      d_sol_D_ptrs.get(), d_sol_ldds.get(),
      128);
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
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.3 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

  Options<ProblemShape> options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  bool explicit_size = false;
  {
    cutlass::CommandLine cmd(argc, args);
    explicit_size = cmd.check_cmd_line_flag("m") ||
                    cmd.check_cmd_line_flag("n") ||
                    cmd.check_cmd_line_flag("k") ||
                    cmd.check_cmd_line_flag("groups") ||
                    !options.benchmark_path.empty();
  }

  struct TestConfig {
    const char* label;
    int m, n, k, groups;
  };

  std::vector<TestConfig> configs;
  int iterations = 20;

  if (explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k, options.groups});
  } else {
    configs = {
      {"small",    128,  128,  128,  4},
      {"medium",   512,  512,  512, 16},
      {"large",   2048, 2048, 2048,  4},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, groups=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.groups);

    // Set up uniform problem sizes for all groups
    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;
    options.groups = cfg.groups;
    options.problem_sizes_host.clear();
    options.problem_sizes_after_alignment_host.clear();
    for (int i = 0; i < cfg.groups; i++) {
      options.problem_sizes_host.push_back({cfg.m, cfg.n, cfg.k});
      options.problem_sizes_after_alignment_host.push_back({cfg.m, cfg.n, cfg.k});
    }

    if (TestCorrectness(options) != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(options, iterations);
    }
  }

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
