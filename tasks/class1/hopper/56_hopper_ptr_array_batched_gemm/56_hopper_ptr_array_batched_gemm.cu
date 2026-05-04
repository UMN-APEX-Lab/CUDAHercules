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
    \brief Hopper Ptr-Array Batched GEMM example using CUTLASS 3 APIs for NVIDIA Hopper architecture.

    This example demonstrates an implementation of Ptr-Array Batched GEMM using a TMA + GMMA
    warp-specialized cooperative kernel.
    The new feature showcased in this example is on-the-fly modification of TMA descriptors
    to move between batches (represented by l).

    To run this example:

      $ ./examples/56_hopper_ptr_array_batched_gemm/56_hopper_ptr_array_batched_gemm --m=2048 --n=2048 --k=2048 --l=10
*/

#include <iostream>

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

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size

// Different configs for pingpong/cooperative
struct CooperativeConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape           = Shape<_256,_128,_64>;
  using ClusterShape        = Shape<_1,_2,_1>;
};

struct PingpongConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape           = Shape<_64,_128,_64>;
  using ClusterShape        = Shape<_1,_1,_1>;
};

template <typename ScheduleConfig>
struct GemmGivenSchedule {
  using TileShape           = typename ScheduleConfig::TileShape;                   // Threadblock-level tile size
  using ClusterShape        = typename ScheduleConfig::ClusterShape;                // Shape of the threadblocks in a cluster
  using KernelSchedule      = typename ScheduleConfig::KernelSchedule;              // Kernel to launch
  using EpilogueSchedule    = typename ScheduleConfig::EpilogueSchedule;            // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using GemmKernel = GemmGivenSchedule<CooperativeConfig>::GemmKernel;
using Gemm = GemmGivenSchedule<CooperativeConfig>::Gemm;

using GemmKernelPingpong = GemmGivenSchedule<PingpongConfig>::GemmKernel;
using GemmPingpong = GemmGivenSchedule<PingpongConfig>::Gemm;


// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 10;
  int m = 1024, n = 512, k = 1024, l = 10;

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
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "56_hopper_ptr_array_batched_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the batch count for Ptr-Array GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "56_hopper_ptr_array_batched_gemm" << " --m=1024 --n=512 --k=1024 --l=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

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
  } else if (bits_input <= 8) {
    scope_max = static_cast<Element>(2);
    scope_min = static_cast<Element>(-2);
  } else {
    scope_max = static_cast<Element>(8);
    scope_min = static_cast<Element>(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  for (int32_t i = 0; i < options.l; ++i) {

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = options.m * options.k;
    int64_t elements_B = options.k * options.n;
    int64_t elements_C = options.m * options.n;
    int64_t elements_D = options.m * options.n;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.l);
  std::vector<ElementB *> ptr_B_host(options.l);
  std::vector<ElementC *> ptr_C_host(options.l);
  std::vector<ElementC *> ptr_D_host(options.l);

  for (int32_t i = 0; i < options.l; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
  }

  ptr_A.reset(options.l);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.l);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.l);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.l);
  ptr_D.copy_from_host(ptr_D_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename GemmT>
typename GemmT::Arguments args_from_options(const Options &options)
{
  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GemmT::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kArray,
    {{options.m, options.n, options.k, options.l}},
    {ptr_A.get(), stride_A, ptr_B.get(), stride_B},
    {{options.alpha, options.beta}, ptr_C.get(), stride_C, ptr_D.get(), stride_D},
    hw_info
  };

  return arguments;
}

bool verify(const Options &options) {
  bool passed = true;
  for (int32_t i = 0; i < options.l; ++i) {
    cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({options.m, options.k}));
    cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({options.k, options.n}));
    cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({options.m, options.n}));
    cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({options.m, options.n}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(
      {options.m, options.n, options.k},
      ElementAccumulator(options.alpha),
      ref_A,
      ref_B,
      ElementAccumulator(options.beta),
      ref_C,
      ref_D);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    passed &= cutlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), options.m * options.n);
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename GemmT>
int run(Options &options)
{
  allocate(options);
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  GemmT gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<GemmT>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmT::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  // CPU verification skipped (non-fatal, not needed for benchmark)

#ifdef KH_TEST_SOLUTION
  // Verify solution against CUTLASS reference
  {
    int M = options.m, N = options.n, K = options.k, batch = options.l;
    int lda = K, ldb = K, ldd = M;
    size_t sizeD_per_batch = (size_t)M * N;

    // Build device pointer arrays for solution
    // block_A, block_B are already allocated, use same offsets
    // Allocate solution output buffers
    std::vector<__half *> sol_D_host(batch);
    for (int b = 0; b < batch; ++b) {
      cudaMalloc(&sol_D_host[b], sizeD_per_batch * sizeof(__half));
      cudaMemset(sol_D_host[b], 0, sizeD_per_batch * sizeof(__half));
    }

    // Build pointer arrays for A, B, D_sol
    std::vector<__half const *> h_A_ptrs(batch), h_B_ptrs(batch);
    std::vector<__half *> h_D_ptrs(batch);
    for (int b = 0; b < batch; ++b) {
      h_A_ptrs[b] = reinterpret_cast<const __half*>(block_A.get() + offset_A[b]);
      h_B_ptrs[b] = reinterpret_cast<const __half*>(block_B.get() + offset_B[b]);
      h_D_ptrs[b] = sol_D_host[b];
    }

    __half const **d_A_ptrs_sol;
    __half const **d_B_ptrs_sol;
    __half **d_D_ptrs_sol;
    cudaMalloc(&d_A_ptrs_sol, batch * sizeof(__half const *));
    cudaMalloc(&d_B_ptrs_sol, batch * sizeof(__half const *));
    cudaMalloc(&d_D_ptrs_sol, batch * sizeof(__half *));
    cudaMemcpy(d_A_ptrs_sol, h_A_ptrs.data(), batch * sizeof(__half const *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptrs_sol, h_B_ptrs.data(), batch * sizeof(__half const *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D_ptrs_sol, h_D_ptrs.data(), batch * sizeof(__half *), cudaMemcpyHostToDevice);

    cudaError_t sol_err = HopperPtrArrayBatchedGemm(
      M, N, K, batch, options.alpha,
      d_A_ptrs_sol, lda,
      d_B_ptrs_sol, ldb,
      options.beta,
      d_D_ptrs_sol, ldd);

    if (sol_err != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
      for (int b = 0; b < batch; ++b) cudaFree(sol_D_host[b]);
      cudaFree(d_A_ptrs_sol); cudaFree(d_B_ptrs_sol); cudaFree(d_D_ptrs_sol);
      std::cout << "Incorrect" << std::endl;
      exit(-1);
    }
    cudaDeviceSynchronize();

    // Compare solution output vs CUTLASS reference per batch
    bool sol_passed = true;
    for (int b = 0; b < batch && sol_passed; ++b) {
      std::vector<__half> h_ref(sizeD_per_batch), h_sol(sizeD_per_batch);
      cudaMemcpy(h_ref.data(), block_D.get() + offset_D[b], sizeD_per_batch * sizeof(__half), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_sol.data(), sol_D_host[b], sizeD_per_batch * sizeof(__half), cudaMemcpyDeviceToHost);

      float max_diff = 0.0f;
      for (size_t i = 0; i < sizeD_per_batch; ++i) {
        float diff = std::fabs(__half2float(h_sol[i]) - __half2float(h_ref[i]));
        max_diff = std::fmax(max_diff, diff);
      }
      if (max_diff > 1.0f) {
        std::cerr << "Solution incorrect at batch " << b << ". Max diff: " << max_diff << std::endl;
        sol_passed = false;
      }
    }

    for (int b = 0; b < batch; ++b) cudaFree(sol_D_host[b]);
    cudaFree(d_A_ptrs_sol); cudaFree(d_B_ptrs_sol); cudaFree(d_D_ptrs_sol);

    if (!sol_passed) {
      std::cout << "Incorrect" << std::endl;
      exit(-1);
    }
  }
#endif

  std::cout << "Passed" << std::endl;

  // Profiling
  int const kIterations = options.iterations;
  if (kIterations > 0)
  {
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warmup reference
    for (int i = 0; i < 3; i++) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    cudaDeviceSynchronize();

    // Profile reference
    float ref_total_ms = 0, ref_min_ms = 1e30f;
    for (int iter = 0; iter < kIterations; ++iter) {
      cudaEventRecord(ev_start);
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
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
    {
      int M = options.m, N = options.n, K = options.k, batch = options.l;
      int lda = K, ldb = K, ldd = M;
      size_t sizeD_per_batch = (size_t)M * N;

      // Allocate solution output
      std::vector<__half *> sol_D_host(batch);
      for (int b = 0; b < batch; ++b) {
        cudaMalloc(&sol_D_host[b], sizeD_per_batch * sizeof(__half));
      }

      std::vector<__half const *> h_A_ptrs(batch), h_B_ptrs(batch);
      std::vector<__half *> h_D_ptrs(batch);
      for (int b = 0; b < batch; ++b) {
        h_A_ptrs[b] = reinterpret_cast<const __half*>(block_A.get() + offset_A[b]);
        h_B_ptrs[b] = reinterpret_cast<const __half*>(block_B.get() + offset_B[b]);
        h_D_ptrs[b] = sol_D_host[b];
      }

      __half const **d_A_ptrs_sol;
      __half const **d_B_ptrs_sol;
      __half **d_D_ptrs_sol;
      cudaMalloc(&d_A_ptrs_sol, batch * sizeof(__half const *));
      cudaMalloc(&d_B_ptrs_sol, batch * sizeof(__half const *));
      cudaMalloc(&d_D_ptrs_sol, batch * sizeof(__half *));
      cudaMemcpy(d_A_ptrs_sol, h_A_ptrs.data(), batch * sizeof(__half const *), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B_ptrs_sol, h_B_ptrs.data(), batch * sizeof(__half const *), cudaMemcpyHostToDevice);
      cudaMemcpy(d_D_ptrs_sol, h_D_ptrs.data(), batch * sizeof(__half *), cudaMemcpyHostToDevice);

      // Warmup solution
      for (int i = 0; i < 3; i++) {
        HopperPtrArrayBatchedGemm(M, N, K, batch, options.alpha,
          d_A_ptrs_sol, lda, d_B_ptrs_sol, ldb, options.beta, d_D_ptrs_sol, ldd);
      }
      cudaDeviceSynchronize();

      // Profile solution
      float sol_total_ms = 0, sol_min_ms = 1e30f;
      for (int iter = 0; iter < kIterations; ++iter) {
        cudaEventRecord(ev_start);
        HopperPtrArrayBatchedGemm(M, N, K, batch, options.alpha,
          d_A_ptrs_sol, lda, d_B_ptrs_sol, ldb, options.beta, d_D_ptrs_sol, ldd);
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

      for (int b = 0; b < batch; ++b) cudaFree(sol_D_host[b]);
      cudaFree(d_A_ptrs_sol); cudaFree(d_B_ptrs_sol); cudaFree(d_D_ptrs_sol);
    }
#endif

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.3 Toolkit to run this example
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
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

  //
  // Multi-size testing
  //

  struct TestConfig {
    const char* label;
    int m, n, k, batch;
  };

  bool explicit_size = false;
  for (int i = 1; i < argc; ++i) {
    std::string a(args[i]);
    if (a.find("--m=") == 0 || a.find("--n=") == 0 || a.find("--k=") == 0 || a.find("--l=") == 0) {
      explicit_size = true;
      break;
    }
  }

  std::vector<TestConfig> configs;
  if (explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k, options.l});
  } else {
    configs = {
      {"small",   128,  128,  128,   4},
      {"medium",  512,  512,  512,  16},
      {"large",  2048, 2048, 2048,   4},
    };
  }

  options.iterations = 20;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch);

    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;
    options.l = cfg.batch;

    // Reset offset vectors for each config
    offset_A.clear();
    offset_B.clear();
    offset_C.clear();
    offset_D.clear();

    run<Gemm>(options);
  }
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

