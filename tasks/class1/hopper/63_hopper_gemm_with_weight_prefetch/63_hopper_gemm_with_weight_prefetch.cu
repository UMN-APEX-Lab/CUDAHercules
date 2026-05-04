/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Hopper FP8 GEMM + L2 Weight Prefetch

    This example implements a non-persistent warp-specialized GEMM kernel for the Hopper
    architecture with programmatic dependent launch (PDL) enabling prefetching weights into
    L2 cache.
    
    For more information about dependent launch refer to the CUDA programming guide:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization

    In some cases, PDL can result in a window where a previous kernel is not actively utilizing 
    DRAM, and the next kernel sits idle until the previous finishes. During this window, the next
    kernel can begin loading a non-dependent operand (i.e. weights in a linear projection are
    typically static) and cache it in L2.

    The kernel and collective mainloop assume operand `A` corresponds to weights and operand `B`
    corresponds to activations (so we can have very small batch/token count).
    After initialization, the prefetch warp starts loading K tiles of `A` into an unused portion 
    of shared memory, and loads up to half of all K tiles that the same CTA would eventually load.
    The exact number of K tiles loaded is determined by `args.mainloop.prefetch_ratio` \in 
    [0.0, 1.0]. Smaller values result in less prefetching, and larger values result in more.
    Negative values result in a "best-effort" prefetch, meaning prefetcher will stop issuing weight
    loads as soon as the activation DMA warp starts loading (as soon as it is signaled that the 
    previous kernel has flushed its memory.)

    The DMA warp responsible for loading `A` will also begin loading K tiles until it fills up
    the available shared memory.
    The DMA warp responsible for loading `B` will wait until activations are flushed to global 
    memory by the preceding kernel.

    Another mainloop parameter, `args.mainloop.overlap_ratio` \in [0.0, 1.0] determines how early 
    the next kernel (the one doing the prefetch) is launched. Smaller values result in greater 
    overlap, and larger values result in smaller overlap. Negative values disable PDL completely,
    meaning there will be no overlap. This will make prefetch ineffective.

    These two runtime parameters should be tuned per problem size and GEMM config combination, and
    if feasible, per-operation in an entire layer or model.

    NOTE: you must build this target with the following flag to enable Grid Dependency Control
    instructions (GDC) in CUTLASS:
      - CUTLASS_ENABLE_GDC_FOR_SM90

    To lock persistence mode, power (350W), clocks (1005MHz) for evaluation (assumes device 0 and H100)

      $ sudo nvidia-smi -pm 1 -i 0

      $ sudo nvidia-smi -i 0 -pl 350

      $ sudo nvidia-smi -i 0 -lgc 1005

    Example:

      $ mkdir build && cd build

      $ cmake .. -DCUTLASS_NVCC_ARCHS="90a" -DCUTLASS_ENABLE_GDC_FOR_SM90=1

      $ cd examples/63_hopper_gemm_with_weight_prefetch

      $ make

      $ ./63_hopper_gemm_with_weight_prefetch --p=0.5 --o=0.5
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
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
#include "cutlass/util/reference/host/gett.hpp"


#include "collective/dispatch_policy_extra.hpp"
#include "collective/builder.hpp"
#include "kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp"

#include "helper.h"
#include "gemm_with_weight_prefetch_commandline.hpp"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp8.h>
#include "solution.h"
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e5m2_t;                          // Element type for B matrix operand
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
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_64,_64,_128>;                            // Threadblock-level tile size
// Cluster_N > 1 is not supported yet.
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  double eff_bw;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    double eff_bw = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), eff_bw(eff_bw), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_tensor(tensor_B.host_view(), seed + 2023);
  initialize_tensor(tensor_C.host_view(), seed + 2024);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;
  fusion_args.alpha_ptr = scalar_alpha.device_data();
  fusion_args.beta_ptr = scalar_beta.device_data();

  arguments.mainloop.overlap_ratio = options.overlap_ratio;
  arguments.mainloop.prefetch_ratio = options.prefetch_ratio;

  return arguments;
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto C = cute::make_tensor(tensor_C.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t, // bias
      unused_t, // aux
      unused_t, // valpha
      unused_t  // vbeta
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

  return passed;
}

/// Run CUTLASS reference GEMM once with current tensors/options
cudaError_t RunCutlassReference(Options &options) {
  Gemm gemm;
  auto arguments = args_from_options(options);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = gemm.run(nullptr, nullptr, /* launch_with_pdl = */ options.overlap_ratio >= 0);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(Options &options) {
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
  // Run solution: A row-major E4M3, B col-major E5M2, D col-major E4M3
  // Allocate solution output
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  cutlass::HostTensor<ElementD, LayoutD> tensor_sol_D;
  tensor_sol_D.resize(c_coord);
  tensor_sol_D.sync_device();

  // The solution uses __nv_fp8_e4m3 / __nv_fp8_e5m2 raw pointers
  // A is row-major [M, K] (lda = K), B is col-major [K, N] (ldb = K), D is col-major [M, N] (ldd = M)
  int lda_sol = options.k;
  int ldb_sol = options.k;
  int ldd_sol = options.m;

  cudaError_t sol_result = HopperGemmPrefetch(
    options.m, options.n, options.k,
    options.alpha,
    reinterpret_cast<__nv_fp8_e4m3 const*>(tensor_A.device_data()), lda_sol,
    reinterpret_cast<__nv_fp8_e5m2 const*>(tensor_B.device_data()), ldb_sol,
    options.beta,
    reinterpret_cast<__nv_fp8_e4m3*>(tensor_sol_D.device_data()), ldd_sol);

  if (sol_result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_result) << std::endl;
    return sol_result;
  }
  cudaDeviceSynchronize();

  // Compare solution vs CUTLASS reference
  tensor_sol_D.sync_host();
  tensor_D.sync_host();

  // Use element-wise comparison with FP8 tolerance
  int total = options.m * options.n;
  float max_diff = 0.0f;
  for (int i = 0; i < total; ++i) {
    float ref_val = float(tensor_D.host_data()[i]);
    float sol_val = float(tensor_sol_D.host_data()[i]);
    max_diff = std::fmax(max_diff, std::fabs(sol_val - ref_val));
  }
  // FP8 E4M3 has limited precision
  if (max_diff > 1.0f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
    return cudaErrorUnknown;
  }
#endif

  return cudaSuccess;
}

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(Options &options, int iterations) {
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
  int lda_sol = options.k, ldb_sol = options.k, ldd_sol = options.m;
  int total_D = options.m * options.n;

  // Save reference output for correctness check
  __nv_fp8_e4m3 *D_ref_saved;
  cudaMalloc(&D_ref_saved, total_D * sizeof(__nv_fp8_e4m3));
  cudaMemcpy(D_ref_saved, tensor_D.device_data(), total_D * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToDevice);

  // Run solution once for correctness check
  HopperGemmPrefetch(options.m, options.n, options.k, options.alpha,
    reinterpret_cast<__nv_fp8_e4m3 const*>(tensor_A.device_data()), lda_sol,
    reinterpret_cast<__nv_fp8_e5m2 const*>(tensor_B.device_data()), ldb_sol,
    options.beta,
    reinterpret_cast<__nv_fp8_e4m3*>(tensor_D.device_data()), ldd_sol);
  cudaDeviceSynchronize();

  {
    std::vector<__nv_fp8_e4m3> h_sol(total_D), h_ref(total_D);
    cudaMemcpy(h_sol.data(), tensor_D.device_data(), total_D * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved, total_D * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    cudaFree(D_ref_saved);
    float max_diff = 0;
    for (int i = 0; i < total_D; ++i)
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
    HopperGemmPrefetch(options.m, options.n, options.k, options.alpha,
      reinterpret_cast<__nv_fp8_e4m3 const*>(tensor_A.device_data()), lda_sol,
      reinterpret_cast<__nv_fp8_e5m2 const*>(tensor_B.device_data()), ldb_sol,
      options.beta,
      reinterpret_cast<__nv_fp8_e4m3*>(tensor_D.device_data()), ldd_sol);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    HopperGemmPrefetch(options.m, options.n, options.k, options.alpha,
      reinterpret_cast<__nv_fp8_e4m3 const*>(tensor_A.device_data()), lda_sol,
      reinterpret_cast<__nv_fp8_e5m2 const*>(tensor_B.device_data()), ldb_sol,
      options.beta,
      reinterpret_cast<__nv_fp8_e4m3*>(tensor_D.device_data()), ldd_sol);
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

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  // Check for explicit size on command line
  Options options;
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

    options.m = cfg.m;
    options.n = cfg.n;
    options.k = cfg.k;

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
