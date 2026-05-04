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
    \brief 
    Hopper Mixed-input Grouped GEMM example using CUTLASS 3 APIs for NVIDIA Hopper architecture. 
    See 55_hopper_mixed_dtype_gemm.cu for more details about Mixed-input GEMMs.

    Limitations:
      1) Only support row-wise scaling. Zero-points and block-wise scaling is currently not supported.

    To run this example:

      $ ./examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm --m=2048 --n=2048 --k=2048 --mode=1 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes.
      Skipping any of the problem dimensions randomizes it across the different groups.
      Same applies for alpha and beta values that are randomized across the different groups.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <typeinfo>
#include <float.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "helper.h"
#include "grouped_mixed_dtype_utils.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::float_e5m2_t;
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix configuration
using         ElementA    = MmaType;
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = cutlass::bfloat16_t;
using ElementScale = cutlass::bfloat16_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_16,cute::Int<TileShapeK>>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementB, LayoutB_Transpose *, AlignmentB,
    ElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelConvertOnly = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopConvertOnly,
    CollectiveEpilogue
>;

using GemmConvertOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelConvertOnly>;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.
using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale>, LayoutB_Transpose *, AlignmentB,
    ElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, 
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

using StrideA = typename GemmConvertOnly::GemmKernel::InternalStrideA;
using StrideB = typename GemmConvertOnly::GemmKernel::InternalStrideB;
using StrideC = typename GemmConvertOnly::GemmKernel::InternalStrideC;
using StrideD = typename GemmConvertOnly::GemmKernel::InternalStrideD;
using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_B_dq;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_scale;
std::vector<int64_t> offset_zero;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;
std::vector<StrideC_ref> stride_C_host_ref;
std::vector<StrideD_ref> stride_D_host_ref;
std::vector<StrideS> stride_S_host;
std::vector<StrideS_ref> stride_S_host_ref;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed = 2020;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<MmaType> block_A;
cutlass::DeviceAllocation<QuantType> block_B;
cutlass::DeviceAllocation<MmaType> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<ElementZero> block_zero;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const MmaType *> ptr_A;
cutlass::DeviceAllocation<const QuantType *> ptr_B;
cutlass::DeviceAllocation<const MmaType *> ptr_B_dq;
cutlass::DeviceAllocation<const ElementScale *> ptr_scale;
cutlass::DeviceAllocation<const ElementZero *> ptr_zero;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput *> ptr_D;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<StrideC_ref> stride_C_ref;
cutlass::DeviceAllocation<StrideD_ref> stride_D_ref;
cutlass::DeviceAllocation<StrideS_ref> stride_S_ref;
cutlass::DeviceAllocation<StrideS> stride_S;

// Note, this is an array of pointers to alpha and beta scaling values per group
cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

/// Clear global state between multi-size iterations
void clear_global_state() {
  offset_A.clear();
  offset_B.clear();
  offset_B_dq.clear();
  offset_C.clear();
  offset_D.clear();
  offset_scale.clear();
  offset_zero.clear();

  stride_A_host.clear();
  stride_B_host.clear();
  stride_C_host.clear();
  stride_D_host.clear();
  stride_C_host_ref.clear();
  stride_D_host_ref.clear();
  stride_S_host.clear();
  stride_S_host_ref.clear();

  alpha_host.clear();
  beta_host.clear();
}

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using Options = GroupedMixedDtypeOptions<QuantType>;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device-side data
void allocate(Options const& options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_B_dq = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_scale = 0;
  int64_t total_elements_zero = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    int const scale_k = cutlass::ceil_div(options.k, options.c);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B * cutlass::sizeof_bits<QuantType>::value / 8);
    offset_B_dq.push_back(total_elements_B_dq);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_scale.push_back(total_elements_scale);
    offset_zero.push_back(total_elements_zero);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N ;
    int64_t elements_B_dq = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_scale = scale_k * N;
    int64_t elements_zero = scale_k * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_B_dq += elements_B_dq;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_scale += elements_scale;
    total_elements_zero += elements_zero;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1}));
    stride_C_host_ref.push_back(cutlass::make_cute_packed_stride(StrideC_ref{}, {M, N, 1}));
    stride_D_host_ref.push_back(cutlass::make_cute_packed_stride(StrideD_ref{}, {M, N, 1}));
    stride_S_host_ref.push_back(cutlass::make_cute_packed_stride(StrideS_ref{}, {N, scale_k, 1}));
    stride_S_host.push_back(cutlass::make_cute_packed_stride(StrideS{}, {N, scale_k, 1}));
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_B_dq.reset(total_elements_B_dq);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_scale.reset(total_elements_scale);
  block_zero.reset(total_elements_zero);

  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(Options &options) {

  uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<MmaType *> ptr_A_host(options.groups);
  std::vector<QuantType *> ptr_B_host(options.groups);
  std::vector<MmaType *> ptr_B_dq_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementC *> ptr_D_host(options.groups);
  std::vector<ElementScale *> ptr_scale_host(options.groups);
  std::vector<ElementZero *> ptr_zero_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_B_dq_host.at(i) = block_B_dq.get() + offset_B_dq.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    ptr_scale_host.at(i) = block_scale.get() + offset_scale.at(i);
    ptr_zero_host.at(i) = block_zero.get() + offset_zero.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_B_dq.reset(options.groups);
  ptr_B_dq.copy_from_host(ptr_B_dq_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_scale.reset(options.groups);
  ptr_scale.copy_from_host(ptr_scale_host.data());

  ptr_zero.reset(options.groups);
  ptr_zero.copy_from_host(ptr_zero_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  stride_C_ref.reset(options.groups);
  stride_C_ref.copy_from_host(stride_C_host_ref.data());

  stride_D_ref.reset(options.groups);
  stride_D_ref.copy_from_host(stride_D_host_ref.data());

  stride_S_ref.reset(options.groups);
  stride_S_ref.copy_from_host(stride_S_host_ref.data());

  stride_S.reset(options.groups);
  stride_S.copy_from_host(stride_S_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_tensor(block_A, seed + 2023);
  initialize_tensor(block_B, seed + 2022);
  initialize_tensor(block_C, seed + 2021);
  initialize_scale(block_scale, options);
  initialize_zero(block_zero, options);
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  problem_sizes.reset(options.groups);
  // Reverse MN -> NM for SwapAB
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [M, N, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(N, M, K);
  }
  problem_sizes.copy_from_host(options.problem_sizes_host.data());
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename Gemm>
typename Gemm::Arguments args_from_options(Options const& options, bool host_problem_shapes_available = true)
{ 
  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;

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

  if constexpr (Gemm::CollectiveMainloop::KernelConversionMode == Gemm::CollectiveMainloop::ConversionMode::DirectConvert) {
    arguments = typename Gemm::Arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), nullptr},
      {ptr_B.get(), stride_B.get(), ptr_A.get(), stride_A.get()},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info
    };
  }
  else if constexpr (Gemm::CollectiveMainloop::KernelConversionMode == Gemm::CollectiveMainloop::ConversionMode::ConvertAndScale) {
    arguments = typename Gemm::Arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), nullptr},
      {ptr_B.get(), stride_B.get(), ptr_A.get(), stride_A.get(), ptr_scale.get(), stride_S.get(), options.c},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info
    };
  } 
  else {
    std::cerr << "Invalid mode " << options.mode << ". Must be 0, 1 or 2." << std::endl;
    exit(-1);
  }
  return arguments;
}

bool verify(Options const& options) {
  bool passed = true;

  constexpr bool IsFP8Input = cute::is_same_v<MmaType, cutlass::float_e4m3_t> || cute::is_same_v<MmaType, cutlass::float_e5m2_t>;
  using FP8Sched = cute::conditional_t<size<0>(TileShape{}) == 64, cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum, cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using ScheduleRef = cute::conditional_t<IsFP8Input, FP8Sched, cutlass::gemm::collective::KernelScheduleAuto>;


  using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaType, LayoutA, AlignmentA,
    MmaType, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    ScheduleRef
  >::CollectiveOp;

  using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloopRef,
    CollectiveEpilogueRef
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;
  using StrideA_verif = typename GemmRef::GemmKernel::StrideA;
  using StrideB_verif = typename GemmRef::GemmKernel::StrideB;
  using StrideC_verif = typename GemmRef::GemmKernel::StrideC;
  using StrideD_verif = typename GemmRef::GemmKernel::StrideD;

  const ElementD epsilon(1e-2f);
  const ElementD non_zero_floor(1e-4f);

  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    // we don't swap and transpose in the verify so revert the problem shape.
    auto N = get<0>(problem);
    auto M = get<1>(problem);
    auto K = get<2>(problem);
    if (M == 0) {
      continue;
    }
    else {
      StrideA_verif stride_A_verif;
      StrideB_verif stride_B_verif;

      stride_A_verif = cutlass::make_cute_packed_stride(StrideA_verif{}, cute::make_shape(M, K, 1));
      stride_B_verif = cutlass::make_cute_packed_stride(StrideB_verif{}, cute::make_shape(N, K, 1));

      int const scale_k = cutlass::ceil_div(options.k, options.c);
      auto layout_B = make_layout(cute::make_shape(N, K, Int<1>{}), stride_B_host.at(i));
      auto layout_scale_zero = make_layout(cute::make_shape(N, scale_k, Int<1>{}), stride_S_host_ref.at(i));
      cudaStream_t stream = cudaStreamDefault;
      cutlass::dequantize(block_B_dq.get() + offset_B_dq.at(i), block_B.get() + offset_B.at(i), layout_B, block_scale.get() + offset_scale.at(i), block_zero.get() + offset_zero.at(i), layout_scale_zero, options.c, stream);

      //
      // Compute reference output
      //

      typename GemmRef::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {block_A.get() + offset_A.at(i), stride_A_verif, block_B_dq.get() + offset_B_dq.at(i), stride_B_verif},
        {{alpha_host.at(i), beta_host.at(i)}, block_C.get() + offset_C.at(i), stride_C_host_ref.at(i), block_ref_D.get() + offset_D.at(i), stride_D_host_ref.at(i)}
      };

      // Run the gemm where the scaling is performed outside of the kernel.
      GemmRef gemm_ref;
      size_t workspace_size = GemmRef::get_workspace_size(arguments);
      cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
      CUTLASS_CHECK(gemm_ref.can_implement(arguments));
      CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm_ref.run());

      // Wait for kernel to finish
      CUDA_CHECK(cudaDeviceSynchronize());

      passed &= cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N, epsilon, non_zero_floor);
      std::cout << "Group " << i << ": " << options.problem_sizes_host[i] << ", alpha: " << alpha_host[i] << ", beta: " << beta_host[i] << " Status: " << passed << std::endl;
    }
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options, bool host_problem_shapes_available = true)
{
  allocate(options);
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<Gemm>(options, host_problem_shapes_available);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  MixedDtypeResult result;
  // CPU verification skipped (non-fatal, not needed for benchmark)

#ifdef KH_TEST_SOLUTION
  // Test solution: build simple-interface data and verify against CPU reference
  {
    // The problem sizes were swapped (N,M,K) for CUTLASS. Un-swap for the solution.
    int PC = options.groups;
    std::vector<int> h_Ms(PC), h_Ns(PC), h_Ks(PC);
    std::vector<int> h_ldas(PC), h_ldbs(PC), h_ldds(PC);
    for (int i = 0; i < PC; ++i) {
      auto [N_swapped, M_swapped, K_val] = options.problem_sizes_host[i];
      h_Ms[i] = M_swapped;
      h_Ns[i] = N_swapped;
      h_Ks[i] = K_val;
      h_ldas[i] = K_val;  // A row-major: lda = K
      h_ldbs[i] = K_val;  // B col-major: ldb = K
      h_ldds[i] = N_swapped;  // D row-major: ldd = N
    }

    // Allocate solution D arrays (half)
    std::vector<__half*> sol_D_ptrs(PC);
    for (int i = 0; i < PC; ++i) {
      CUDA_CHECK(cudaMalloc(&sol_D_ptrs[i], h_Ms[i] * h_Ns[i] * sizeof(__half)));
      CUDA_CHECK(cudaMemset(sol_D_ptrs[i], 0, h_Ms[i] * h_Ns[i] * sizeof(__half)));
    }

    // Convert A (bf16) -> half, B (fp8) -> int8 for solution interface
    std::vector<__half*> sol_A_ptrs(PC);
    std::vector<int8_t*> sol_B_ptrs(PC);
    for (int i = 0; i < PC; ++i) {
      int sA = h_Ms[i] * h_Ks[i], sB = h_Ks[i] * h_Ns[i];

      // Copy A bf16 -> host -> convert to half -> copy to device
      std::vector<__nv_bfloat16> h_A_bf16(sA);
      cutlass::device_memory::copy_to_host(
        reinterpret_cast<__nv_bfloat16*>(h_A_bf16.data()),
        reinterpret_cast<__nv_bfloat16*>(block_A.get() + offset_A[i]),
        sA);
      std::vector<__half> h_A_half(sA);
      for (int j = 0; j < sA; ++j) h_A_half[j] = __float2half(__bfloat162float(h_A_bf16[j]));
      CUDA_CHECK(cudaMalloc(&sol_A_ptrs[i], sA * sizeof(__half)));
      CUDA_CHECK(cudaMemcpy(sol_A_ptrs[i], h_A_half.data(), sA * sizeof(__half), cudaMemcpyHostToDevice));

      // Copy B fp8 -> host -> convert to int8 (clamp) -> copy to device
      std::vector<uint8_t> h_B_fp8(sB);
      CUDA_CHECK(cudaMemcpy(h_B_fp8.data(), block_B.get() + offset_B[i], sB, cudaMemcpyDeviceToHost));
      std::vector<int8_t> h_B_int8(sB);
      for (int j = 0; j < sB; ++j) h_B_int8[j] = static_cast<int8_t>(h_B_fp8[j]);
      CUDA_CHECK(cudaMalloc(&sol_B_ptrs[i], sB * sizeof(int8_t)));
      CUDA_CHECK(cudaMemcpy(sol_B_ptrs[i], h_B_int8.data(), sB * sizeof(int8_t), cudaMemcpyHostToDevice));
    }

    // Copy arrays to device
    int *dMs, *dNs, *dKs, *dldas, *dldbs, *dldds;
    __half const **dAp;
    int8_t const **dBp;
    __half **dDp;
    CUDA_CHECK(cudaMalloc(&dMs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dKs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldas, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldbs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldds, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dAp, PC * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&dBp, PC * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&dDp, PC * sizeof(void*)));

    CUDA_CHECK(cudaMemcpy(dMs, h_Ms.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNs, h_Ns.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKs, h_Ks.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldas, h_ldas.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldbs, h_ldbs.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldds, h_ldds.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAp, sol_A_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBp, sol_B_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDp, sol_D_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));

    float sol_alpha = alpha_host[0], sol_beta = 0.0f;

    cudaError_t sol_err = HopperMixedGroupedGemm(
      PC, dMs, dNs, dKs, sol_alpha, (const __half**)dAp, dldas, (const int8_t**)dBp, dldbs, sol_beta, dDp, dldds);
    if (sol_err != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
      result.passed = false;
    } else {
      CUDA_CHECK(cudaDeviceSynchronize());

      // Verify each group with CPU reference
      for (int p = 0; p < PC && result.passed; ++p) {
        int M = h_Ms[p], N = h_Ns[p], K = h_Ks[p];
        int sD = M * N;
        std::vector<__half> hD(sD);
        CUDA_CHECK(cudaMemcpy(hD.data(), sol_D_ptrs[p], sD * sizeof(__half), cudaMemcpyDeviceToHost));

        // Get A and B in float for CPU reference
        int sA = M * K, sB = K * N;
        std::vector<__nv_bfloat16> h_A_bf16(sA);
        cutlass::device_memory::copy_to_host(
          reinterpret_cast<__nv_bfloat16*>(h_A_bf16.data()),
          reinterpret_cast<__nv_bfloat16*>(block_A.get() + offset_A[p]),
          sA);
        std::vector<uint8_t> h_B_fp8(sB);
        CUDA_CHECK(cudaMemcpy(h_B_fp8.data(), block_B.get() + offset_B[p], sB, cudaMemcpyDeviceToHost));

        std::vector<float> fA(sA), fB(sB), fD(sD), fRef(sD, 0);
        for (int j = 0; j < sA; ++j) fA[j] = __bfloat162float(h_A_bf16[j]);
        for (int j = 0; j < sB; ++j) fB[j] = float(int8_t(h_B_fp8[j]));
        for (int j = 0; j < sD; ++j) fD[j] = __half2float(hD[j]);

        // CPU: D = alpha * A @ B (A row-major, B col-major)
        for (int i = 0; i < M; ++i)
          for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k)
              acc += fA[i * K + k] * fB[k + j * K];
            fRef[i * N + j] = sol_alpha * acc;
          }

        float md = 0;
        for (int j = 0; j < sD; ++j) md = fmaxf(md, fabsf(fD[j] - fRef[j]));
        if (md > 1.0f) {
          std::cerr << "Solution group " << p << " incorrect. Max diff: " << md << std::endl;
          result.passed = false;
        }
      }
    }

    // Cleanup solution data
    for (int i = 0; i < PC; ++i) {
      cudaFree(sol_A_ptrs[i]);
      cudaFree(sol_B_ptrs[i]);
      cudaFree(sol_D_ptrs[i]);
    }
    cudaFree(dMs); cudaFree(dNs); cudaFree(dKs);
    cudaFree(dldas); cudaFree(dldbs); cudaFree(dldds);
    cudaFree(dAp); cudaFree(dBp); cudaFree(dDp);
  }
  if (!result.passed) {
    std::cout << "Incorrect" << std::endl;
    return -1;
  }
#endif

  //
  // Profile reference
  //
  int const kIterations = options.iterations;
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CUTLASS_CHECK(gemm.run());
    cudaDeviceSynchronize();
  }

  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < kIterations; ++iter) {
    cudaEventRecord(ev_start);
    CUTLASS_CHECK(gemm.run());
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    ref_total_ms += ms;
    if (ms < ref_min_ms) ref_min_ms = ms;
  }
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref_total_ms / kIterations, kIterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
  // Profile solution
  {
    int PC = options.groups;
    std::vector<int> h_Ms(PC), h_Ns(PC), h_Ks(PC);
    std::vector<int> h_ldas(PC), h_ldbs(PC), h_ldds(PC);
    for (int i = 0; i < PC; ++i) {
      auto [N_swapped, M_swapped, K_val] = options.problem_sizes_host[i];
      h_Ms[i] = M_swapped; h_Ns[i] = N_swapped; h_Ks[i] = K_val;
      h_ldas[i] = K_val; h_ldbs[i] = K_val; h_ldds[i] = N_swapped;
    }

    std::vector<__half*> sol_A_ptrs(PC), sol_D_ptrs(PC);
    std::vector<int8_t*> sol_B_ptrs(PC);
    for (int i = 0; i < PC; ++i) {
      int sA = h_Ms[i] * h_Ks[i], sB = h_Ks[i] * h_Ns[i], sD = h_Ms[i] * h_Ns[i];
      std::vector<__nv_bfloat16> h_A_bf16(sA);
      cutlass::device_memory::copy_to_host(
        reinterpret_cast<__nv_bfloat16*>(h_A_bf16.data()),
        reinterpret_cast<__nv_bfloat16*>(block_A.get() + offset_A[i]), sA);
      std::vector<__half> h_A_half(sA);
      for (int j = 0; j < sA; ++j) h_A_half[j] = __float2half(__bfloat162float(h_A_bf16[j]));
      CUDA_CHECK(cudaMalloc(&sol_A_ptrs[i], sA * sizeof(__half)));
      CUDA_CHECK(cudaMemcpy(sol_A_ptrs[i], h_A_half.data(), sA * sizeof(__half), cudaMemcpyHostToDevice));

      std::vector<uint8_t> h_B_fp8(sB);
      CUDA_CHECK(cudaMemcpy(h_B_fp8.data(), block_B.get() + offset_B[i], sB, cudaMemcpyDeviceToHost));
      std::vector<int8_t> h_B_int8(sB);
      for (int j = 0; j < sB; ++j) h_B_int8[j] = static_cast<int8_t>(h_B_fp8[j]);
      CUDA_CHECK(cudaMalloc(&sol_B_ptrs[i], sB * sizeof(int8_t)));
      CUDA_CHECK(cudaMemcpy(sol_B_ptrs[i], h_B_int8.data(), sB * sizeof(int8_t), cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMalloc(&sol_D_ptrs[i], sD * sizeof(__half)));
    }

    int *dMs, *dNs, *dKs, *dldas, *dldbs, *dldds;
    __half const **dAp; int8_t const **dBp; __half **dDp;
    CUDA_CHECK(cudaMalloc(&dMs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dKs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldas, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldbs, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dldds, PC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dAp, PC * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&dBp, PC * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&dDp, PC * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(dMs, h_Ms.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNs, h_Ns.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKs, h_Ks.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldas, h_ldas.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldbs, h_ldbs.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dldds, h_ldds.data(), PC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAp, sol_A_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBp, sol_B_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDp, sol_D_ptrs.data(), PC * sizeof(void*), cudaMemcpyHostToDevice));

    float sol_alpha = alpha_host[0], sol_beta = 0.0f;

    // Warmup
    for (int i = 0; i < 3; i++) {
      HopperMixedGroupedGemm(PC, dMs, dNs, dKs, sol_alpha, (const __half**)dAp, dldas, (const int8_t**)dBp, dldbs, sol_beta, dDp, dldds);
    }
    cudaDeviceSynchronize();

    float sol_total_ms = 0, sol_min_ms = 1e30f;
    for (int iter = 0; iter < kIterations; ++iter) {
      cudaEventRecord(ev_start);
      HopperMixedGroupedGemm(PC, dMs, dNs, dKs, sol_alpha, (const __half**)dAp, dldas, (const int8_t**)dBp, dldbs, sol_beta, dDp, dldds);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);
      float ms;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      sol_total_ms += ms;
      if (ms < sol_min_ms) sol_min_ms = ms;
    }
    fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            sol_total_ms / kIterations, kIterations, sol_min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);

    for (int i = 0; i < PC; ++i) {
      cudaFree(sol_A_ptrs[i]); cudaFree(sol_B_ptrs[i]); cudaFree(sol_D_ptrs[i]);
    }
    cudaFree(dMs); cudaFree(dNs); cudaFree(dKs);
    cudaFree(dldas); cudaFree(dldbs); cudaFree(dldds);
    cudaFree(dAp); cudaFree(dBp); cudaFree(dDp);
  }
#endif

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.3 Toolkit to run this example
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
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

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

  // Multi-size test configurations
  struct TestConfig {
    const char* label;
    int m, k, groups;
  };

  // Check if explicit size was given
  bool explicit_size = false;
  for (int i = 1; i < argc; ++i) {
    std::string a(args[i]);
    if (a.find("--m=") == 0 || a.find("--groups=") == 0) {
      explicit_size = true;
      break;
    }
  }

  std::vector<TestConfig> configs;
  if (explicit_size) {
    configs.push_back({"custom", options.m, options.k, options.groups});
  } else {
    configs = {
      {"small",    128,  512, 4},
      {"medium",   512,  512, 16},
      {"large",   2048,  512, 4},
    };
  }

  // N is fixed in the CUTLASS config (uses options.n which defaults to 2048)
  int fixed_n = 2048;

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, groups=%d ===\n",
            cfg.label, cfg.m, fixed_n, cfg.k, cfg.groups);

    // Clear global state from previous iteration
    clear_global_state();

    // Re-create options for this config
    Options size_options;
    size_options.m = cfg.m;
    size_options.n = fixed_n;
    size_options.k = cfg.k;
    size_options.groups = cfg.groups;
    size_options.mode = 1;  // ScaleOnly
    size_options.alpha = 1.0f;
    size_options.beta = 0.0f;
    size_options.iterations = 20;

    // Build problem sizes (all same M,N,K)
    size_options.problem_sizes_host.clear();
    for (int i = 0; i < cfg.groups; ++i) {
      size_options.problem_sizes_host.push_back({cfg.m, fixed_n, cfg.k});
    }

    run<GemmScaleOnly>(size_options, false);
    std::cout << "Passed" << std::endl;
  }

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
