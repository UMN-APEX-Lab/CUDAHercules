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
    \brief Example implementation of fused multi-head attention for the NVIDIA Blackwell SM100
    architecture using CUTLASS 3.

    MQA/GQA
    -------

    The head dimension can be represented as a tuple, where the K/V strides in the
    first dimension is zero. This has the effect of MQA or GQA.
    * MHA is (head_size:head_stride).
    * MQA is (head_size:head_stride) in Q and (head_size:_0) in K and V.
    * GQA is (grouped_heads,heads_kv):(head_stride,grouped_heads*head_stride) in Q
      and (grouped_heads,heads_kv):(0,head_stride) in K and V

    Output Scale
    ------------

    The output scale gets passed to the collective mainloop, and is applied
    using FP32 compute pre-quantization

    Variable Sequence Length
    ------------------------

    For variable sequence length, pass in VariableLength objects
    (max_seqlen, cumulative_seqlen_ptr) in the problem shape for
    seqlen Q and KV.

    Support
    ---------

    Right now e4m3 with fp32 compute is using a 256x256 tiling and a head dimension
    of 128 is supported.


    Example usage:
      $ ./examples/77_blackell_fmha/77_blackell_fmha_fp8 \
            --b=2048 --h=2048 --d=2048 --q=2048 --k=2048
*/

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_fwd_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  std::vector<int> varlen_q;
  std::vector<int> varlen_k;
  int d = 128;
  int warmup_iterations = 1;
  int iterations = 3;
  int tensor_ring_buffers = 1;
  bool verify = false;
  bool verbose = false;

  bool causal = false;
  bool causal_q_begin = true;
  bool residual = false;
  bool varlen = false;
  bool persistent = false;
  int sm_count = 0;
  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_k = InitStyle::kRandom;
  InitStyle init_style_v = InitStyle::kRandom;

  static void get_init_style_argument(cutlass::CommandLine& cmd, const char* name, InitStyle& dst, InitStyle const& src) {
    std::string s;
    cmd.get_cmd_line_argument(name, s, s);
    if (s.empty()) {
      dst = src;
    }
    else {
      if (s == "r") {
        dst = InitStyle::kRandom;
      }
      else if (s == "1") {
        dst = InitStyle::kOne;
      }
      else if (s == "d") {
        dst = InitStyle::kLinearStride1;
      }
      else if (s == "s") {
        dst = InitStyle::kLinearStride128;
      }
      else if (s == "n") {
        dst = InitStyle::kNone;
      }
      else {
        std::cout << "Error: " << s << " is not a valid input type.\n";
        std::exit(-1);
      }
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    Options defaults;

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("d", d, defaults.d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

    cmd.get_cmd_line_argument("h_k", h_k, -1);
    if (h_k == -1) h_k = h;

    varlen = cmd.check_cmd_line_flag("varlen");

    cmd.get_cmd_line_argument("q", q, -1);
    cmd.get_cmd_line_argument("k", k, -1);
    cmd.get_cmd_line_argument("b", b, -1);

    std::string varlen_q_str;
    cmd.get_cmd_line_argument("varlen-q", varlen_q_str);
    std::string varlen_k_str;
    cmd.get_cmd_line_argument("varlen-k", varlen_k_str);

    if (varlen && ! varlen_q_str.empty()) {
      varlen_q.clear();
      while (! varlen_q_str.empty()) {
        size_t pos = varlen_q_str.find(':');
        varlen_q.push_back(std::stoi(varlen_q_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_q_str = varlen_q_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_q.size());
      }
      if (b != static_cast<int>(varlen_q.size())) {
        std::cout << "Error: Invalid --varlen-q length\n";
        std::exit(-1);
      }
      int new_q = 0;
      for (auto elem : varlen_q) {
        new_q += elem;
      }
      if (q != -1) {
        std::cout << "Error: Can't provide --q and --varlen-q\n";
        std::exit(-1);
      }
      q = new_q;
    }

    if (varlen && ! varlen_k_str.empty()) {
      varlen_k.clear();
      while (! varlen_k_str.empty()) {
        size_t pos = varlen_k_str.find(':');
        varlen_k.push_back(std::stoi(varlen_k_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_k_str = varlen_k_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_k.size());
      }
      if (b != static_cast<int>(varlen_k.size())) {
        std::cout << " Error: Invalid --varlen-k length\n";
        std::exit(-1);
      }
      int new_k = 0;
      for (auto elem : varlen_k) {
        new_k += elem;
      }
      if (k != -1) {
        std::cout << "Error: Can't provide --k and --varlen-k\n";
        std::exit(-1);
      }
      k = new_k;
    }

    if (q == -1) q = k;
    if (k == -1) k = q;
    if (q == -1 && k == -1) q = k = defaults.q;
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, defaults.warmup_iterations);
    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    cmd.get_cmd_line_argument("tensor_ring_buffers", tensor_ring_buffers, defaults.tensor_ring_buffers);

    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    persistent = cmd.check_cmd_line_flag("persistent");

    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    std::string causal_type;
    cmd.get_cmd_line_argument<std::string>("causal-type", causal_type, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
      if (varlen) {
        residual = true;
      }
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
      if(causal_type == "qend") {
        causal_q_begin = false;
      } else {
        causal_q_begin = true;
      }
    }
    else if (mask == "residual") {
      residual = true;
      causal = false;
    }
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);
    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_k, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_v, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-k", init_style_k, init_style_k);
    get_init_style_argument(cmd, "init-style-v", init_style_v, init_style_v);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_fmha\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention forward-passkernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --h_k=<int>                 Sets the H_K/V extent (for GQA/MQA)\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --varlen-q=<int>:<int...>   Sets the variable Q extent per batch (colon separated)\n"
      << "  --varlen-k=<int>:<int...>   Sets the variable K extent per batch (colon separated)\n"
      << "  --d=<int>                   Sets the D extent\n"
      << "  --tensor_ring_buffers=<int> Sets the number of tensor ring buffers\n"
      << "  --warmup_iterations=<int>   Sets the warmup iterations\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
      << "  --causal-type=<qbegin|qend> Causal mask type\n"
      << "  --persistent                Enables persistent scheduler\n"
      << "  --varlen                    Enables variable sequence length\n"
      << "                              B*Q and B*K become the total sequence length\n"
      << "                              and are split B-ways, alternatingly +10% and -10%\n"
      << "                              with the last batch sized to make it fit\n"
      << "                              implies at least residual masking for correctness\n"
      << "  --sm-count                  Sets SM count rather than querying it\n"
      << "  --kernel-filter=<filter>    Sets regexp to match kernel against\n"
      << "\n";

    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(
    DeviceAllocation<Element>& block,
    uint64_t seed=2023, InitStyle init_style = InitStyle::kRandom) {

  switch (init_style) {
    case InitStyle::kOne: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 1, (Element) 1);
      break;
    }
    case InitStyle::kRandom: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) 0, (Element) 1);
      break;
    }
    case InitStyle::kLinearStride1: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (j % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kLinearStride128: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (i % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kNone: {
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleResult {
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  double tops_exp2_s = 0;
  double tbytes_s = 0;
  size_t smem_size = 0;
#ifdef KH_TEST_SOLUTION
  float sol_total_ms = 0;
  float sol_min_ms = 1e30f;
  int sol_iterations = 0;
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  bool kIsVarlen,
  class TileShape,
  class DispatchPolicy,
  class ActiveMask,
  class... KernelOptions
>
struct FwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#else
  using Element = cutlass::half_t;
#endif

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  // Q K D ((H_R, H_K) B)
  using ProblemShapeRegular = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType = std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;
  
  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D ((H_R, H_K), B)
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;  // K D ((H_R, H_K), B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;     // Q ((H_R, H_K), B)

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;
  using TileScheduler = std::conditional_t<kIsPersistent, cutlass::fmha::kernel::PersistentTileScheduler, cutlass::fmha::kernel::IndividualTileScheduler>;

  using Mainloop = 
    cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      ActiveMask
    >;
  using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType,
      Mainloop,
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
      >,
      TileScheduler
    >>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  struct DeviceBuffer {
    DeviceAllocation<Element> block_Q;
    DeviceAllocation<Element> block_K;
    DeviceAllocation<Element> block_V;
    DeviceAllocation<ElementOut> block_O;
    DeviceAllocation<ElementAccumulatorPV> block_LSE;
    DeviceAllocation<ElementOut> block_ref_O;
    DeviceAllocation<ElementAccumulatorPV> block_ref_LSE;
    DeviceAllocation<int> device_cumulative_seqlen_q;
    DeviceAllocation<int> device_cumulative_seqlen_kv;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    size_t get_storage_size() const {
      return block_Q.get_storage_size() + block_K.get_storage_size() + block_V.get_storage_size()
          + block_O.get_storage_size() + block_LSE.get_storage_size() + block_ref_O.get_storage_size()
          + block_ref_LSE.get_storage_size() + device_cumulative_seqlen_q.get_storage_size()
          + device_cumulative_seqlen_kv.get_storage_size();
    }
  };

  std::vector<std::unique_ptr<DeviceBuffer>> buffers;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;

  //
  // Methods
  //
  bool verify(const ProblemShapeType& problem_shape, DeviceBuffer& buffer) {
    Tensor mQ = make_tensor(make_gmem_ptr(buffer.block_Q.get()),
      select<0,2,3>(problem_shape),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(buffer.block_K.get()),
      select<1,2,3>(problem_shape),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(buffer.block_V.get()),
      select<1,2,3>(problem_shape),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(buffer.block_ref_O.get()),
      select<0,2,3>(problem_shape),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(buffer.block_ref_LSE.get()),
      select<0,3>(problem_shape),
      stride_LSE);
    
    auto [Q, K, D, HB] = problem_shape;

    auto problem_shape_ref = cute::make_tuple(Q, K, D, D, HB);

    fmha_reference(problem_shape_ref, mQ, mK, mV, mO, mLSE, ActiveMask{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(buffer.block_O, buffer.block_ref_O, max_diff, mean_diff);

    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(buffer.block_LSE, buffer.block_ref_LSE, max_diff, mean_diff);

    bool passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_LSE;
  }

  template<class ProblemShape>
  auto initialize_varlen(
      const Options& options, const ProblemShape& problem_size,
      const bool kVarlenSame = true) {

    int num_batches = get<3,1>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative 
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<0>(problem_size), get<0>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<1>(problem_size), get<1>(problem_size) / 2);
    std::cout << "N: " << num_batches << ", Q: " << get<0>(problem_size) << ", KV: " << get<1>(problem_size) << std::endl;

    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;

    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = (! options.varlen_q.empty()) ? options.varlen_q.at(i) : 
              kVarlenSame ? get<0>(problem_size) :
              generate_positive_int(dist_q, rng);
      int seqlen_kv = (! options.varlen_k.empty()) ? options.varlen_k.at(i) :
              kVarlenSame ? get<1>(problem_size) :
              generate_positive_int(dist_kv, rng);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
    }
    std::cout << "Q max: " << max_seqlen_q << " total: " << total_seqlen_q << " vs even " << num_batches * get<0>(problem_size) << std::endl;
    std::cout << "KV max: " << max_seqlen_kv << " total: " << total_seqlen_kv << " vs even " << num_batches * get<1>(problem_size) << std::endl;

    ProblemShape problem_size_for_init = problem_size;
    get<3,1>(problem_size_for_init) = 1;
    get<0>(problem_size_for_init) = total_seqlen_q;
    get<1>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = VariableLength{max_seqlen_q, nullptr, total_seqlen_q};
    get<1>(problem_size_for_launch) = VariableLength{max_seqlen_kv, nullptr, total_seqlen_kv};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }


  /// Initialize operands to be used in the GEMM and reference GEMM

  ProblemShapeType initialize(const Options& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);
    auto problem_shape_in = cute::make_tuple(options.q, options.k, options.d, cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    
    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(options, problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    get<2>(problem_size) = cutlass::round_up(get<2>(problem_size), 8);  // alignment

    auto shape_QO = select<0,2,3>(problem_size);
    auto shape_KV = select<1,2,3>(problem_size);
    auto shape_LSE = select<0,3>(problem_size);

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B = size<3,1>(problem_size);

    stride_Q = make_stride(H*D , _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    stride_O = stride_Q;
    stride_K = make_stride(H_K*D , _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    stride_V = stride_K;
    stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    if (kIsVarlen) {
      get<2,1>(stride_Q) = 0;
      get<2,1>(stride_K) = 0;
      get<2,1>(stride_V) = 0;
      get<2,1>(stride_O) = 0;
      get<1,1>(stride_LSE) = 0;
    }

    auto buffer_init_fn = [&](auto& buffer) {
      buffer.block_Q.reset(size(shape_QO));
      buffer.block_K.reset(size(shape_KV));
      buffer.block_V.reset(size(shape_KV));
      buffer.block_O.reset(size(shape_QO), kIsVarlen ? D*SQ*H : 0);
      buffer.block_LSE.reset(size(shape_LSE));
      buffer.block_ref_O.reset(size(shape_QO), kIsVarlen ? D*SQ*H : 0);
      buffer.block_ref_LSE.reset(size(shape_LSE));

      initialize_block(buffer.block_Q, seed + 2023, options.init_style_q);
      initialize_block(buffer.block_K, seed + 2022, options.init_style_k);
      initialize_block(buffer.block_V, seed + 2021, options.init_style_v);

      if ( ! cumulative_seqlen_q.empty()) {
        buffer.device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
        buffer.device_cumulative_seqlen_q.copy_from_host(
          cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      }
      if ( ! cumulative_seqlen_kv.empty()) {
        buffer.device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
        buffer.device_cumulative_seqlen_kv.copy_from_host(
          cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
      }   
    };

    buffers.push_back(std::make_unique<DeviceBuffer>());
    buffer_init_fn(*buffers.back());

    int tensor_ring_buffers = options.tensor_ring_buffers;
    for (int i = 1; i < tensor_ring_buffers; i++) {
      buffers.push_back(std::make_unique<DeviceBuffer>());
      buffer_init_fn(*buffers.back());
    }

    if constexpr (kIsVarlen) {
      get<0>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_kv.get();
    }

    return problem_shape;
  }

  auto get_arguments(const ProblemShapeType& problem_shape, const cutlass::KernelHardwareInfo& hw_info, int buffer_index) {
    auto problem_shape_ = problem_shape;
    if constexpr (kIsVarlen) {
      get<0>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_kv.get();
    }
    typename Operation::Arguments arguments{
      problem_shape_,
      { buffers[buffer_index]->block_Q.get(), stride_Q,
        buffers[buffer_index]->block_K.get(), stride_K,
        buffers[buffer_index]->block_V.get(), stride_V },
      { buffers[buffer_index]->block_O.get(), stride_O,
        buffers[buffer_index]->block_LSE.get(), stride_LSE },
      hw_info
    };
    return arguments;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {

    ProblemShapeType problem_shape = initialize(options);

    int buffer_index = 0;
    typename Operation::Arguments arguments = get_arguments(problem_shape, hw_info, buffer_index);

    Operation op;

    ExampleResult example_result;

    example_result.smem_size = Operation::Kernel::SharedStorageSize;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    status = op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    // Run
    for (int i = 0; i < options.warmup_iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops;
    if (kIsVarlen) {
      flops = 0.0;
      for (int i = 0; i < size<3,1>(problem_shape); i++) {
        flops += (cumulative_seqlen_q[i+1] - cumulative_seqlen_q[i])
               * 1.0
               * (cumulative_seqlen_kv[i+1] - cumulative_seqlen_kv[i]);
      }
    }
    else {
      flops = 1.0;
      flops *= static_cast<double>(size<0>(problem_shape));
      flops *= static_cast<double>(size<1>(problem_shape));
      flops *= static_cast<double>(size<3,1>(problem_shape));
    }
    flops *= 4.0 * (std::is_same_v<ActiveMask, CausalMask<true>> || std::is_same_v<ActiveMask, CausalMask<false>> ? 0.5 : 1.0);
    flops *= static_cast<double>(size<2>(problem_shape));
    flops *= static_cast<double>(size<3,0>(problem_shape));
    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_shape, *buffers[0]);
      if (passed) example_result.verified = true;
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

#ifdef KH_TEST_SOLUTION
    //
    // Run and verify LLM solution against CUTLASS reference
    //
    {
      auto [Q_len, K_len, D, HB] = problem_shape;
      int batch = size<3,1>(problem_shape);
      int num_heads = size<3,0>(problem_shape);
      int seq_q = size<0>(problem_shape);
      int seq_k = size<1>(problem_shape);
      int head_dim = size<2>(problem_shape);

      int total_out = batch * num_heads * seq_q * head_dim;
      DeviceAllocation<ElementOut> sol_O;
      sol_O.reset(total_out);
      cudaMemset(sol_O.get(), 0, sizeof(ElementOut) * total_out);

      // The CUTLASS kernel uses a specific memory layout:
      //   Q/K/V/O are stored as [seq, head_dim, ((head_ratio, heads_kv), batch)]
      // with stride_Q = (H*D, 1, ((D, H_Q*D), H*D*SQ))
      // For the solution we need [batch, num_heads, seq, head_dim] row-major.
      // Since the CUTLASS layout packs things differently, we'll convert.

      int H = num_heads;
      int H_K = size<3,0,1>(problem_shape);
      int H_Q = size<3,0,0>(problem_shape);

      // Allocate temporary buffers in [batch, heads, seq, dim] layout for solution
      DeviceAllocation<Element> sol_Q, sol_K, sol_V;
      sol_Q.reset(batch * H * seq_q * head_dim);
      sol_K.reset(batch * H_K * seq_k * head_dim);
      sol_V.reset(batch * H_K * seq_k * head_dim);

      // Copy CUTLASS layout to standard [B,H,S,D] layout
      // CUTLASS Q layout: stride = (H*D, 1, ((D, H_Q*D), H*D*SQ))
      // Linear index in CUTLASS: q*H*D + d + hr*D + hk*H_Q*D + b*H*D*SQ
      // Standard layout: b*H*S*D + (hr + hk*H_Q)*S*D + s*D + d
      {
        std::vector<Element> h_Q_cutlass(batch * H * seq_q * head_dim);
        std::vector<Element> h_K_cutlass(batch * H_K * seq_k * head_dim);
        std::vector<Element> h_V_cutlass(batch * H_K * seq_k * head_dim);

        cudaMemcpy(h_Q_cutlass.data(), buffers[0]->block_Q.get(), h_Q_cutlass.size() * sizeof(Element), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K_cutlass.data(), buffers[0]->block_K.get(), h_K_cutlass.size() * sizeof(Element), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V_cutlass.data(), buffers[0]->block_V.get(), h_V_cutlass.size() * sizeof(Element), cudaMemcpyDeviceToHost);

        std::vector<Element> h_Q_std(batch * H * seq_q * head_dim);
        std::vector<Element> h_K_std(batch * H_K * seq_k * head_dim);
        std::vector<Element> h_V_std(batch * H_K * seq_k * head_dim);

        // Reorder Q: CUTLASS index = q*(H*D) + d + hr*D + hk*(H_Q*D) + b*(H*D*SQ)
        // Standard index = b*(H*seq_q*D) + (hr + hk*H_Q)*(seq_q*D) + q*D + d
        for (int b_i = 0; b_i < batch; ++b_i)
          for (int hk = 0; hk < H_K; ++hk)
            for (int hr = 0; hr < H_Q; ++hr)
              for (int q_i = 0; q_i < seq_q; ++q_i)
                for (int d_i = 0; d_i < head_dim; ++d_i) {
                  int cutlass_idx = q_i * (H * head_dim) + d_i + hr * head_dim + hk * (H_Q * head_dim) + b_i * (H * head_dim * seq_q);
                  int h_idx = hr + hk * H_Q;
                  int std_idx = b_i * (H * seq_q * head_dim) + h_idx * (seq_q * head_dim) + q_i * head_dim + d_i;
                  h_Q_std[std_idx] = h_Q_cutlass[cutlass_idx];
                }

        // Reorder K/V: CUTLASS index = k*(H_K*D) + d + 0*... + hk*D + b*(H_K*D*SK)
        // Standard index = b*(H_K*seq_k*D) + hk*(seq_k*D) + k*D + d
        for (int b_i = 0; b_i < batch; ++b_i)
          for (int hk = 0; hk < H_K; ++hk)
            for (int k_i = 0; k_i < seq_k; ++k_i)
              for (int d_i = 0; d_i < head_dim; ++d_i) {
                int cutlass_idx = k_i * (H_K * head_dim) + d_i + hk * head_dim + b_i * (H_K * head_dim * seq_k);
                int std_idx = b_i * (H_K * seq_k * head_dim) + hk * (seq_k * head_dim) + k_i * head_dim + d_i;
                h_K_std[std_idx] = h_K_cutlass[cutlass_idx];
                h_V_std[std_idx] = h_V_cutlass[cutlass_idx];
              }

        sol_Q.copy_from_host(h_Q_std.data(), h_Q_std.size());
        sol_K.copy_from_host(h_K_std.data(), h_K_std.size());
        sol_V.copy_from_host(h_V_std.data(), h_V_std.size());
      }

      cudaError_t sol_err = BlackwellFmha(
        batch, H, seq_q, seq_k, head_dim,
        reinterpret_cast<__half const*>(sol_Q.get()),
        reinterpret_cast<__half const*>(sol_K.get()),
        reinterpret_cast<__half const*>(sol_V.get()),
        reinterpret_cast<__half*>(sol_O.get()));

      if (sol_err != cudaSuccess) {
        std::cerr << "Solution kernel failed: " << cudaGetErrorString(sol_err) << std::endl;
        return example_result;
      }
      cudaDeviceSynchronize();

      // Convert CUTLASS output to standard layout for comparison
      std::vector<ElementOut> h_ref_O_cutlass(total_out);
      cudaMemcpy(h_ref_O_cutlass.data(), buffers[0]->block_O.get(), total_out * sizeof(ElementOut), cudaMemcpyDeviceToHost);

      std::vector<ElementOut> h_ref_O_std(total_out);
      for (int b_i = 0; b_i < batch; ++b_i)
        for (int hk = 0; hk < H_K; ++hk)
          for (int hr = 0; hr < H_Q; ++hr)
            for (int q_i = 0; q_i < seq_q; ++q_i)
              for (int d_i = 0; d_i < head_dim; ++d_i) {
                int cutlass_idx = q_i * (H * head_dim) + d_i + hr * head_dim + hk * (H_Q * head_dim) + b_i * (H * head_dim * seq_q);
                int h_idx = hr + hk * H_Q;
                int std_idx = b_i * (H * seq_q * head_dim) + h_idx * (seq_q * head_dim) + q_i * head_dim + d_i;
                h_ref_O_std[std_idx] = h_ref_O_cutlass[cutlass_idx];
              }

      std::vector<ElementOut> h_sol_O(total_out);
      cudaMemcpy(h_sol_O.data(), sol_O.get(), total_out * sizeof(ElementOut), cudaMemcpyDeviceToHost);

      int mismatches = 0;
      for (int i = 0; i < total_out; ++i) {
        float s = float(h_sol_O[i]), r = float(h_ref_O_std[i]);
        float diff = std::fabs(s - r);
        float denom = std::fmax(std::fabs(r), 1e-6f);
        if (diff > 0.05f && diff / denom > 0.1f) mismatches++;
      }
      float mismatch_rate = float(mismatches) / float(total_out);
      if (mismatch_rate > 0.02f) {
        std::cerr << "Solution incorrect. Mismatch rate: " << mismatch_rate << std::endl;
        return example_result;
      }

      // Profile solution with per-iteration CUDA event timing
      if (options.iterations > 0) {
        int kIterations = options.iterations;
        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        // Warmup solution
        for (int i = 0; i < 3; i++) {
          BlackwellFmha(
            batch, H, seq_q, seq_k, head_dim,
            reinterpret_cast<__half const*>(sol_Q.get()),
            reinterpret_cast<__half const*>(sol_K.get()),
            reinterpret_cast<__half const*>(sol_V.get()),
            reinterpret_cast<__half*>(sol_O.get()));
        }
        cudaDeviceSynchronize();

        float sol_total_ms_local = 0, sol_min_ms_local = 1e30f;
        for (int iter = 0; iter < kIterations; ++iter) {
          cudaEventRecord(ev_start);
          BlackwellFmha(
            batch, H, seq_q, seq_k, head_dim,
            reinterpret_cast<__half const*>(sol_Q.get()),
            reinterpret_cast<__half const*>(sol_K.get()),
            reinterpret_cast<__half const*>(sol_V.get()),
            reinterpret_cast<__half*>(sol_O.get()));
          cudaEventRecord(ev_stop);
          cudaEventSynchronize(ev_stop);
          float ms;
          cudaEventElapsedTime(&ms, ev_start, ev_stop);
          sol_total_ms_local += ms;
          if (ms < sol_min_ms_local) sol_min_ms_local = ms;
        }
        example_result.sol_total_ms = sol_total_ms_local;
        example_result.sol_min_ms = sol_min_ms_local;
        example_result.sol_iterations = kIterations;

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
      }
    }
#endif // KH_TEST_SOLUTION

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_result = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  if (! result.passed) {
    main_result = -1;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
ExampleResult run_fwd_64_single(Mask, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  using HeadDim = _64;
  FwdRunner<false, Shape<_256, _128, HeadDim>, void, Mask, Option<Tag::kIsPersistent, false_type>> runner;
  return runner.run(options, hw_info);
}

template<class Mask>
ExampleResult run_fwd_128_single(Mask, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  using HeadDim = _128;
  FwdRunner<false, Shape<_256, _128, HeadDim>, void, Mask, Option<Tag::kIsPersistent, false_type>> runner;
  return runner.run(options, hw_info);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

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

  if (props.major != 10) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Blackwell Architecture "
      << "(compute capability 100a)." << std::endl;
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

  // Check if explicit size was provided
  bool explicit_size = false;
  {
    cutlass::CommandLine cmd(argc, args);
    explicit_size = cmd.check_cmd_line_flag("q") ||
                    cmd.check_cmd_line_flag("k") ||
                    cmd.check_cmd_line_flag("b") ||
                    cmd.check_cmd_line_flag("h") ||
                    cmd.check_cmd_line_flag("d");
  }

  struct TestConfig {
    const char* label;
    int seq_q, seq_k, heads, head_dim, batch;
  };

  std::vector<TestConfig> configs;

  if (explicit_size) {
    configs.push_back({"custom", options.q, options.k, options.h, options.d, options.b});
  } else {
    configs = {
      {"small",    256,  256,  8, 64,  4},
      {"medium",   512,  512, 16, 64, 16},
      {"large",   2048, 2048, 16, 64,  4},
    };
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  if (options.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  } else {
    hw_info.sm_count = options.sm_count;
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: seq=%d, heads=%d, hdim=%d, batch=%d ===\n",
            cfg.label, cfg.seq_q, cfg.heads, cfg.head_dim, cfg.batch);

    options.q = cfg.seq_q;
    options.k = cfg.seq_k;
    options.h = cfg.heads;
    options.h_k = cfg.heads;
    options.d = cfg.head_dim;
    options.b = cfg.batch;
    options.verify = true;
    options.iterations = 20;
    options.warmup_iterations = 3;

    ExampleResult result;
    if (cfg.head_dim <= 64) {
      result = run_fwd_64_single(NoMask{}, options, hw_info);
    } else {
      result = run_fwd_128_single(NoMask{}, options, hw_info);
    }

    if (!result.passed) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }

    std::cout << "Passed" << std::endl;

    // Output profiling results
    if (options.iterations > 0) {
      fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
              result.runtime_ms, options.iterations, result.runtime_ms);

#ifdef KH_TEST_SOLUTION
      if (result.sol_iterations > 0) {
        fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
                result.sol_total_ms / result.sol_iterations, result.sol_iterations, result.sol_min_ms);
        fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", result.runtime_ms / result.sol_min_ms);
      }
#endif
    }
  }
#endif

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
