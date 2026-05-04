/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief CUTLASS Attention Example.

    This workload computes a fused multi head attention.
    Because it keeps the attention matrix in shared memory, it's both faster and
    uses less global memory.

    This is based on `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_,
    and very similar to `"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" <https://arxiv.org/abs/2205.14135>`_.

    Algorithm:
      In short, we can compute the output incrementally in blocks of size B,
      we just need to divide the final result by the sum of all coefficients in
      the softmax (which we compute incrementally) with the following pseudo-code:

      ```
      s_prime = torch.zeros([num_queries, B])
      O = torch.zeros([num_queries, head_size_v])
      for i in range(0, K.shape[0], B):
        si = exp((Q . K[i * B:(i+1) * B].t) * scale)
        sum_coefs += attn_unscaled.sum(-1)
        O  += si . V[i * B:(i+1) * B]
      O = O / s_prime
      ```

      In practice, and for numerical stability reasons,
      we also subtract the maximum so far (`mi`) before doing
      the exponential. When we encounter new keys, the maximum
      used to compute O so far (`m_prime`) can differ from the
      current maximum, so we update O before accumulating with

      ```
      O       = O * exp(m_prime - mi)
      m_prime = mi
      ```

    Implementation details:
      - `si` is stored in shared memory between the 2 back to back gemms
      - we keep and accumulate the output
      directly in registers if we can (`head_size_v <= 128`).
      Otherwise, we store it & accumulate in global memory (slower)
      - blocks are parallelized across the batch dimension, the number
      of heads, and the query sequence size


    Examples:

      # Run an attention example with default setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen

      # Run an attention example with custom setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen --head_number=2 --batch_size=3 --head_size=32 --head_size_v=64 --seq_length=512 --seq_length_kv=1024 --causal=true

      Acknowledgement: Fixed-sequence-length FMHA code was upstreamed by Meta xFormers (https://github.com/facebookresearch/xformers).
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"
#include "kernel_forward.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  bool use_mask;
  bool causal;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes0;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes0_real;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1_real;

  int alignment;
  int head_number;
  int batch_size;
  int head_size;
  int head_size_v;
  int seq_length;
  int seq_length_kv;
  int iterations;

  // alpha0, alpha1 and beta are fixed 
  // in this multi-head attention example
  float alpha0;
  float alpha1;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    error(false),
    alignment(1),
    reference_check(true),
    head_number(12),
    batch_size(16),
    head_size(64),
    head_size_v(64),
    seq_length(1024),
    seq_length_kv(1024),
    use_mask(false),
    iterations(20),
    causal(false)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alignment", alignment, 1);
    cmd.get_cmd_line_argument("head_number", head_number, 12);
    cmd.get_cmd_line_argument("batch_size", batch_size, 16);
    cmd.get_cmd_line_argument("head_size", head_size, 64);
    cmd.get_cmd_line_argument("head_size_v", head_size_v, head_size);
    cmd.get_cmd_line_argument("seq_length", seq_length, 1024);
    cmd.get_cmd_line_argument("seq_length_kv", seq_length_kv, seq_length);
    cmd.get_cmd_line_argument("use_mask", use_mask, false);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("causal", causal, true);

    randomize_problems();

  }

  void randomize_problems() {

    int problem_count = head_number * batch_size;

    problem_sizes0.reserve(problem_count);
    problem_sizes1.reserve(problem_count);

    // When using mask, the original inputs are not padded
    // and we need to save these info.
    if (use_mask) {
      problem_sizes0_real.reserve(problem_count);
      problem_sizes1_real.reserve(problem_count);
    }

    for (int i = 0; i < batch_size; ++i) {
      // problems belonging to the same batch share the same seq len
      int m_real = seq_length;
      int mkv_real = seq_length_kv;
      int m = (m_real + alignment - 1) / alignment * alignment;
      int mkv = (mkv_real + alignment - 1) / alignment * alignment;
      int k0 = head_size;
      int k1 = head_size_v;

      for (int j = 0; j < head_number; ++j) {
        cutlass::gemm::GemmCoord problem0(m, mkv, k0);
        cutlass::gemm::GemmCoord problem1(m, k1, mkv);
        problem_sizes0.push_back(problem0);
        problem_sizes1.push_back(problem1);

        if (use_mask) {
          cutlass::gemm::GemmCoord problem0_real(m_real, mkv_real, k0);
          cutlass::gemm::GemmCoord problem1_real(m_real, k1, mkv_real);
          problem_sizes0_real.push_back(problem0_real);
          problem_sizes1_real.push_back(problem1_real);
        }
      }
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "41_fused_multi_head_attention_fixed_seqlen\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --head_number=<int>         Head number in multi-head attention (default: --head_number=12)\n"
      << "  --batch_size=<int>          Batch size in multi-head attention (default: --batch_size=16)\n"
      << "  --head_size=<int>           Head size in multi-head attention (default: --head_size=64)\n"
      << "  --head_size_v=<int>         Head size in multi-head attention for V (default: --head_size_v=head_size)\n"
      << "  --seq_length=<int>          Sequence length in multi-head attention for Q (default: --seq_length=1024)\n"
      << "  --seq_length_kv=<int>       Sequence length in multi-head attention for K/V (default: --seq_length_kv=seq_length)\n"
      << "  --use_mask=<bool>           If true, performs padding-like masking in softmax.\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n"
      << "  --causal=<bool>             If true, uses causal masking.\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fops = int64_t();

    for (size_t i = 0; i < problem_sizes0.size(); ++i) {
      auto const& problem0 = problem_sizes0[i];
      auto const& problem1 = problem_sizes1[i];
      for (int row = 0; row < problem0.m(); ++row) {
        int num_cols0 = problem0.n();
        if (causal) {
          num_cols0 = std::min(row + 1, num_cols0);
        }
        // P <- Q . K_t
        fops += 2 * num_cols0 * problem0.k();
        // P <- exp(P - max(P))
        fops += 2 * num_cols0;
        // S <- sum(P)
        fops += num_cols0 - 1;
        // O <- P . V
        fops += 2 * num_cols0 * problem1.n();
        // O <- O / S
        fops += num_cols0 * problem1.n();
      }
    }

    return double(fops) / double(1.0e9) / runtime_s;
  }
};



///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Attention>
class TestbedAttention {
public:

  //
  // Type definitions
  //

  using ElementQ = typename Attention::scalar_t;
  using ElementK = typename Attention::scalar_t;
  using ElementP = typename Attention::accum_t;
  using ElementAccumulator = typename Attention::accum_t;
  using ElementV = typename Attention::scalar_t;
  using ElementO = typename Attention::output_t;

  using ElementCompute = typename Attention::accum_t;

  using ElementNorm = typename Attention::accum_t;
  using ElementSum = typename Attention::accum_t;
  using ElementSoftmaxCompute = typename Attention::accum_t;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutP = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using MatrixCoord = typename LayoutP::TensorCoord;

private:

  //
  // Data members
  //

  Options & options;

  /// Initialization
  cutlass::Distribution::Kind init_Q;
  cutlass::Distribution::Kind init_K;
  cutlass::Distribution::Kind init_P;
  cutlass::Distribution::Kind init_V;
  cutlass::Distribution::Kind init_O;
  uint32_t seed;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device0;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device1;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device0_real;

  std::vector<int64_t> offset_Q;
  std::vector<int64_t> offset_K;
  std::vector<int64_t> offset_P;
  std::vector<int64_t> offset_V;
  std::vector<int64_t> offset_O;

  std::vector<int64_t> ldq_host;
  std::vector<int64_t> ldk_host;
  std::vector<int64_t> ldp_host;
  std::vector<int64_t> ldv_host;
  std::vector<int64_t> ldo_host;
  std::vector<int64_t> seqlen_host;

  cutlass::DeviceAllocation<int64_t> ldq;
  cutlass::DeviceAllocation<int64_t> ldk;
  cutlass::DeviceAllocation<int64_t> ldp;
  cutlass::DeviceAllocation<int64_t> ldv;
  cutlass::DeviceAllocation<int64_t> ldo;
  cutlass::DeviceAllocation<int64_t> seqlen;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementP> block_P;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementO> block_O;
  cutlass::DeviceAllocation<ElementNorm> block_Norm;
  cutlass::DeviceAllocation<ElementSum> block_Sum;

  cutlass::DeviceAllocation<int64_t> offset_P_Device;

  cutlass::DeviceAllocation<ElementQ *> ptr_Q;
  cutlass::DeviceAllocation<ElementK *> ptr_K;
  cutlass::DeviceAllocation<ElementP *> ptr_P;
  cutlass::DeviceAllocation<ElementV *> ptr_V;
  cutlass::DeviceAllocation<ElementO *> ptr_O;

#ifdef KH_TEST_SOLUTION
  cutlass::DeviceAllocation<ElementO> sol_O;
#endif

public:

  //
  // Methods
  //

  TestbedAttention(
    Options &options_,
    cutlass::Distribution::Kind init_Q_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_K_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_P_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_V_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_O_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ):
    options(options_), init_Q(init_Q_), init_K(init_K_), init_P(init_P_), init_V(init_V_), init_O(init_O_), seed(seed_) { }

  int problem_count() const {
    return (options.head_number * options.batch_size);
  }

private:

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor_(
    Element *ptr,
    size_t capacity, 
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementP>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 8;
        scope_min = -8;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
    } 
    else {

      // Fill with all 1s
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
    }
  }

  /// Initializes data structures
  void initialize_() {

    //
    // Set scalors for the mha example
    //

    options.alpha0 = 1.0f / sqrt(float(options.head_size));
    options.alpha1 = 1.0f;
    options.beta = 0;

    //
    // Choose random problem sizes
    //

    // construct a few problems of random sizes
    srand(seed);

    int64_t total_elements_Q = 0;
    int64_t total_elements_K = 0;
    int64_t total_elements_P = 0;
    int64_t total_elements_V = 0;
    int64_t total_elements_O = 0;

    ldq_host.resize(problem_count());
    ldk_host.resize(problem_count());
    ldp_host.resize(problem_count());
    ldv_host.resize(problem_count());
    ldo_host.resize(problem_count());
    seqlen_host.resize(problem_count());

    // Create tensors in BMHK format, where
    // B = batch_size
    // M = sequence length
    // H = num_heads
    // K = embedding size per head
    int64_t batch_offset_Q, batch_offset_K, batch_offset_V, batch_offset_O;

    for (int32_t b = 0; b < options.batch_size; ++b) {
      batch_offset_Q = total_elements_Q;
      batch_offset_K = total_elements_K;
      batch_offset_V = total_elements_V;
      batch_offset_O = total_elements_O;
      for (int32_t h = 0; h < options.head_number; ++h) {
        int32_t i = h + b * options.head_number;

        auto problem0 = options.problem_sizes0.at(i);
        auto problem1 = options.problem_sizes1.at(i);

        ldq_host.at(i) = LayoutQ::packed({problem0.m(), options.head_number * problem0.k()}).stride(0);
        ldk_host.at(i) = LayoutK::packed({options.head_number * problem0.k(), problem0.n()}).stride(0);
        ldp_host.at(i) = LayoutP::packed({problem0.m(), problem0.n()}).stride(0);
        ldv_host.at(i) = LayoutV::packed({problem1.k(), options.head_number * problem1.n()}).stride(0);
        ldo_host.at(i) = LayoutO::packed({problem1.m(), options.head_number * problem1.n()}).stride(0);

        // m = n for attention problems.
        seqlen_host.at(i) = problem0.m();

        offset_Q.push_back(batch_offset_Q + h * problem0.k());
        offset_K.push_back(batch_offset_K + h * problem0.k());
        offset_P.push_back(total_elements_P);
        offset_V.push_back(batch_offset_V + h * problem0.k());
        offset_O.push_back(batch_offset_O + h * problem1.n());

        int64_t elements_Q = problem0.m() * problem0.k();
        int64_t elements_K = problem0.k() * problem0.n();
        int64_t elements_P = problem0.m() * problem0.n();
        int64_t elements_V = problem1.k() * problem1.n();
        int64_t elements_O = problem1.m() * problem1.n();

        total_elements_Q += elements_Q;
        total_elements_K += elements_K;
        total_elements_P += elements_P;
        total_elements_V += elements_V;
        total_elements_O += elements_O;
      }
    }

    problem_sizes_device0.reset(problem_count());
    problem_sizes_device1.reset(problem_count());
    problem_sizes_device0.copy_from_host(options.problem_sizes0.data());
    problem_sizes_device1.copy_from_host(options.problem_sizes1.data());

    if (options.use_mask) {
      problem_sizes_device0_real.reset(problem_count());
      problem_sizes_device0_real.copy_from_host(options.problem_sizes0_real.data());
    }

    ldq.reset(problem_count());
    ldk.reset(problem_count());
    ldp.reset(problem_count());
    ldv.reset(problem_count());
    ldo.reset(problem_count());
    seqlen.reset(problem_count());

    ldq.copy_from_host(ldq_host.data());
    ldk.copy_from_host(ldk_host.data());
    ldp.copy_from_host(ldp_host.data());
    ldv.copy_from_host(ldv_host.data());
    ldo.copy_from_host(ldo_host.data());
    seqlen.copy_from_host(seqlen_host.data());

    //
    // Assign pointers
    //

    block_Q.reset(total_elements_Q);
    block_K.reset(total_elements_K);
    block_P.reset(total_elements_P);
    block_V.reset(total_elements_V);
    block_O.reset(total_elements_O);

    offset_P_Device.reset(problem_count());

    // sync offset with device
    cutlass::device_memory::copy_to_device(offset_P_Device.get(), offset_P.data(), offset_P.size());

    std::vector<ElementQ *> ptr_Q_host(problem_count());
    std::vector<ElementK *> ptr_K_host(problem_count());
    std::vector<ElementP *> ptr_P_host(problem_count());
    std::vector<ElementV *> ptr_V_host(problem_count());
    std::vector<ElementO *> ptr_O_host(problem_count());
    std::vector<ElementNorm *> ptr_norm_host(problem_count());
    std::vector<ElementSum *> ptr_sum_host(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {
      ptr_Q_host.at(i) = block_Q.get() + offset_Q.at(i);
      ptr_K_host.at(i) = block_K.get() + offset_K.at(i);
      ptr_P_host.at(i) = block_P.get() + offset_P.at(i);
      ptr_V_host.at(i) = block_V.get() + offset_V.at(i);
      ptr_O_host.at(i) = block_O.get() + offset_O.at(i);
    }

    ptr_Q.reset(problem_count());
    ptr_Q.copy_from_host(ptr_Q_host.data());
    
    ptr_K.reset(problem_count());
    ptr_K.copy_from_host(ptr_K_host.data());
    
    ptr_P.reset(problem_count());
    ptr_P.copy_from_host(ptr_P_host.data());

    ptr_V.reset(problem_count());
    ptr_V.copy_from_host(ptr_V_host.data());

    ptr_O.reset(problem_count());
    ptr_O.copy_from_host(ptr_O_host.data());

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor_(block_Q.get(), total_elements_Q, init_Q, seed + 1);
    initialize_tensor_(block_K.get(), total_elements_K, init_K, seed + 2);
    initialize_tensor_(block_V.get(), total_elements_V, init_V, seed + 3);

#ifdef KH_TEST_SOLUTION
    sol_O.reset(total_elements_O);
#endif

  }

  template<typename Element>
  bool verify_tensor_(std::vector<Element> vector_Input, \
                       std::vector<Element> vector_Input_Ref,
                       int64_t verify_length = -1) {

    int64_t size = (vector_Input.size() < vector_Input_Ref.size()) ? vector_Input.size() : vector_Input_Ref.size();
    size = (verify_length == -1) ? size : verify_length;

    // 0.05 for absolute error
    float abs_tol = 5e-2f;
    // 10% for relative error
    float rel_tol = 1e-1f;
    for (int64_t i = 0; i < size; ++i) {
      float diff = (float)(vector_Input.at(i) - vector_Input_Ref.at(i));
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref.at(i) + 1e-5f);
      float relative_diff = abs_diff / abs_ref;
      if ( (isnan(vector_Input_Ref.at(i)) || isnan(abs_diff) || isinf(abs_diff)) ||  (abs_diff > abs_tol && relative_diff > rel_tol)) {
        printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n", int(i), int(size), abs_diff, relative_diff, (float)(vector_Input.at(i)), (float)(vector_Input_Ref.at(i)));
        return false;
      }

    }

    return true;
  }

  /// Verifies the result is a GEMM
  bool verify_() {

    bool passed = true;

    for (int32_t b = 0; b < options.batch_size; ++b) {
      int32_t i = b * options.head_number;
      // Problem size is the same for all heads
      cutlass::gemm::GemmCoord problem0 = options.problem_sizes0.at(b * options.head_number);
      cutlass::gemm::GemmCoord problem1 = options.problem_sizes1.at(b * options.head_number);

      MatrixCoord extent_Q{problem0.m(), problem0.k()};
      MatrixCoord extent_K{problem0.k(), problem0.n()};
      MatrixCoord extent_P{problem0.m(), problem0.n()};
      MatrixCoord extent_V{problem1.k(), problem1.n()};
      MatrixCoord extent_O{problem1.m(), problem1.n()};

      LayoutO layout_O(ldo_host.at(i));
      std::vector<ElementO> matrix_O(layout_O.capacity(extent_O));
      cutlass::device_memory::copy_to_host(matrix_O.data(),   block_O.get() + offset_O.at(i), matrix_O.size());
      cutlass::DeviceAllocation<ElementO>    block_Ref_O(layout_O.capacity(extent_O));

      for (int32_t h = 0; h < options.head_number; ++h) {
        i = h + b * options.head_number;

        LayoutQ layout_Q(ldq_host.at(i));
        LayoutK layout_K(ldk_host.at(i));
        LayoutP layout_P(ldp_host.at(i));
        LayoutV layout_V(ldv_host.at(i));

        cutlass::TensorView<ElementQ, LayoutQ> view_Q(block_Q.get() + offset_Q.at(i), layout_Q, extent_Q);
        cutlass::TensorView<ElementK, LayoutK> view_K(block_K.get() + offset_K.at(i), layout_K, extent_K);
        cutlass::TensorView<ElementV, LayoutV> view_V(block_V.get() + offset_V.at(i), layout_V, extent_V);
        cutlass::TensorView<ElementO, LayoutO> view_Ref_O_device(block_Ref_O.get() + offset_O.at(i) - offset_O.at(b * options.head_number), layout_O, extent_O);

        cutlass::DeviceAllocation<ElementP>    block_Ref_P(layout_P.capacity(extent_P));
        cutlass::TensorView<ElementP, LayoutP> view_Ref_P_device(block_Ref_P.get(), layout_P, extent_P);

        // Reference GEMM
        cutlass::reference::device::GemmComplex<
            ElementQ, LayoutQ,
            ElementK, LayoutK,
            ElementP, LayoutP, 
            ElementCompute, ElementAccumulator
        >(
          problem0,
          ElementAccumulator(options.alpha0), 
          view_Q,
          Attention::MM0::Mma::kTransformA,
          view_K,
          Attention::MM0::Mma::kTransformB,
          ElementAccumulator(options.beta), 
          view_Ref_P_device, 
          view_Ref_P_device, 
          ElementAccumulator(0)
        );

        // Compute softmax for P. We need to explicitly compute softmax
        // over P because softmax is fused to the second GEMM in the
        // profiled implementation.
        std::vector<ElementP> matrix_Ref(layout_P.capacity(extent_P));
        cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref_P.get(), matrix_Ref.size());
        cutlass::TensorView<ElementP, LayoutP> view_Ref_host(matrix_Ref.data(), layout_P, extent_P);
        std::vector<ElementNorm> vector_Norm_Ref(problem0.m());
        std::vector<ElementSum> vector_Sum_Ref(problem0.m());

        int n_dim = options.use_mask ? options.problem_sizes0_real.at(i).n() : problem0.n();

        // Compute softmax for reference matrix
        for (int m = 0; m < problem0.m(); m++) {
          int n_dim_row = n_dim;
          if (options.causal) {
            n_dim_row = std::min(m + 1, n_dim);
          }
          ElementSoftmaxCompute max = ElementSoftmaxCompute(view_Ref_host.ref().at({m, 0}));
          for (int n = 1; n < n_dim_row; n++) {
            max = std::max(max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
          }

          vector_Norm_Ref.at(m) = ElementNorm(max);

          ElementSoftmaxCompute sum = ElementSoftmaxCompute();
          for (int n = 0; n < n_dim_row; n++) {
            sum += std::exp( ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max );
          }
          ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

          vector_Sum_Ref.at(m) = ElementSum(inv_sum);

          for (int n = 0; n < n_dim_row; n++) {
            view_Ref_host.ref().at({m, n}) = ElementP(
              std::exp( ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max ) * inv_sum
            );
          }
          // Mask out the rest of the attention matrix
          for (int n = n_dim_row; n < n_dim; ++n) {
            view_Ref_host.ref().at({m, n}) = ElementP(0);
          }
        }

        // when not using mask, problem_real and problem share the same sizes
        if (options.use_mask) {
          for (int m = 0; m < problem0.m(); m++) {
            for (int n = n_dim; n < problem0.n(); n++) {
              view_Ref_host.ref().at({m, n}) = ElementP(0);
            }
          }
        }

        cutlass::device_memory::copy_to_device(block_Ref_P.get(), matrix_Ref.data(), matrix_Ref.size());

        // Reference GEMM
        cutlass::reference::device::GemmComplex<
            ElementP, LayoutP,
            ElementV, LayoutV,
            ElementO, LayoutO, 
            ElementCompute, ElementAccumulator
        >(
          problem1,
          ElementAccumulator(options.alpha1), 
          view_Ref_P_device,
          Attention::MM0::Mma::kTransformA,
          view_V,
          Attention::MM0::Mma::kTransformB,
          ElementAccumulator(options.beta), 
          view_Ref_O_device, 
          view_Ref_O_device, 
          ElementAccumulator(0)
        );
      }

      // Copy to host memory
      std::vector<ElementO> matrix_Ref_O(layout_O.capacity(extent_O));
      cutlass::device_memory::copy_to_host(matrix_Ref_O.data(), block_Ref_O.get(), matrix_Ref_O.size());

      // printf("Pb %d: \n    Q=(offset=%d, ldq=%d)\n    K=(offset=%d, ldk=%d)\n    O=(offset=%d, ldo=%d)\n",
      //   int(i), int(offset_Q[i]), int(ldq_host[i]), int(offset_K[i]), int(ldk_host[i]), int(offset_O[i]), int(ldo_host[i]));
  
      bool verified_O = false;

      if (!verified_O) {
        verified_O = verify_tensor_<ElementO>(matrix_O, matrix_Ref_O);
      }

      passed = passed && verified_O;

      if (!passed) {
        std::cerr << "\n***\nError - problem " << i << " (batch " << b << ") failed the QA check\n***\n" << std::endl;

        if (!verified_O) {
          std::cout << "Final matrix output is incorrect" << std::endl;
        }

        return passed;
      }
    }

    return passed;
  }

#ifdef KH_TEST_SOLUTION
  cudaError_t execute_solution_kernel() {
    // The CUTLASS example uses BMHK layout: [batch, seq_len, num_heads, head_dim]
    // Q_strideM = num_heads * head_size, Q_strideH = head_size, Q_strideB = seq_len * Q_strideM
    int q_strideH = options.head_size;
    int k_strideH = options.head_size;
    int v_strideH = options.head_size_v;
    int q_strideM = int(ldq_host[0]);
    int k_strideM = int(ldk_host[0]);
    int v_strideM = int(ldv_host[0]);
    int q_strideB = q_strideM * options.seq_length;
    int k_strideB = k_strideM * options.seq_length_kv;
    int v_strideB = v_strideM * options.seq_length_kv;
    int o_strideM = options.head_size_v * options.head_number;

    return fmha_solution(
      options.batch_size, options.seq_length, options.seq_length_kv,
      options.head_number, options.head_size, options.head_size_v,
      reinterpret_cast<const __half*>(block_Q.get()),
      reinterpret_cast<const __half*>(block_K.get()),
      reinterpret_cast<const __half*>(block_V.get()),
      reinterpret_cast<__half*>(sol_O.get()),
      q_strideB, q_strideM, q_strideH,
      k_strideB, k_strideM, k_strideH,
      v_strideB, v_strideM, v_strideH,
      o_strideM,
      options.causal);
  }

  bool verify_solution() {
    // Compare sol_O against block_O
    int64_t total_O = 0;
    for (int i = 0; i < problem_count(); i++) {
      auto const& problem1 = options.problem_sizes1[i];
      total_O += problem1.m() * problem1.n();
    }

    std::vector<ElementO> ref_O(total_O);
    std::vector<ElementO> sol_O_host(total_O);
    cutlass::device_memory::copy_to_host(ref_O.data(), block_O.get(), total_O);
    cutlass::device_memory::copy_to_host(sol_O_host.data(), sol_O.get(), total_O);

    float abs_tol = 5e-2f;
    float rel_tol = 1e-1f;

    for (int64_t i = 0; i < total_O; ++i) {
      float diff = fabs(float(sol_O_host[i]) - float(ref_O[i]));
      float abs_ref = fabs(float(ref_O[i]) + 1e-5f);
      float relative_diff = diff / abs_ref;
      if ((isnan(diff) || isinf(diff)) || (diff > abs_tol && relative_diff > rel_tol)) {
        printf("Solution verification failed at element %d: sol=%f, ref=%f, diff=%f\n",
               int(i), float(sol_O_host[i]), float(ref_O[i]), diff);
        return false;
      }
    }
    return true;
  }
#endif

public:


  /// Executes a CUTLASS Attention kernel and measures runtime.
  Result profile() {

    Result result;
    result.passed = false;

    // Initialize the problem
    initialize_();

    typename Attention::Params p;
    { // set parameters
      p.query_ptr = block_Q.get();
      p.key_ptr = block_K.get();
      p.value_ptr = block_V.get();
      p.logsumexp_ptr = nullptr; // Only needed for bw
      p.output_accum_ptr = nullptr;
      if (Attention::kNeedsOutputAccumulatorBuffer) {
        cudaMalloc(&p.output_accum_ptr, block_O.size() * sizeof(typename Attention::output_accum_t));
      }
      p.output_ptr = block_O.get();

      // TODO: support arbitrary seq lengths
      // if (cu_seqlens_q.has_value()) {
      //   p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      //   p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
      // }

      p.scale = options.alpha0;

      p.num_heads = options.head_number;
      p.num_batches = options.batch_size;
      p.head_dim = options.head_size;
      p.head_dim_value = options.head_size_v;
      p.num_queries = options.seq_length;
      p.num_keys = options.seq_length_kv;
      if (options.causal) {
        p.custom_mask_type = Attention::CausalFromTopLeft;
      }

      // All tensors are in BMHK shapes
      p.q_strideH = options.head_size;
      p.k_strideH = options.head_size;
      p.v_strideH = options.head_size_v;
      p.q_strideM = int32_t(ldq_host[0]);
      p.k_strideM = int32_t(ldk_host[0]);
      p.v_strideM = int32_t(ldv_host[0]);
      p.q_strideB = p.q_strideM * options.seq_length;
      p.k_strideB = p.k_strideM * options.seq_length_kv;
      p.v_strideB = p.v_strideM * options.seq_length_kv;
      p.o_strideM = p.head_dim_value * p.num_heads;
    }

    // launch kernel :)
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return result;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_();
    }

#ifdef KH_TEST_SOLUTION
    //
    // Run and verify LLM solution against CUTLASS reference
    //
    if (result.passed) {
      cudaError_t sol_err = execute_solution_kernel();
      if (sol_err != cudaSuccess) {
        std::cerr << "Solution execution failed: " << cudaGetErrorString(sol_err) << std::endl;
        result.passed = false;
      } else {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
          std::cerr << "Solution synchronize failed: " << cudaGetErrorString(sync_err) << std::endl;
          result.passed = false;
        } else if (!verify_solution()) {
          result.passed = false;
        }
      }
    }
#endif

    // Print correctness result before profiling
    if (!result.passed) {
      std::cout << "Incorrect" << std::endl;
      return result;
    }
    std::cout << "Passed" << std::endl;

    //
    // Warm-up run of reference
    //
    for (int i = 0; i < 3; i++) {
      kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
      cudaDeviceSynchronize();
    }

    //
    // Profile reference (per-iteration timing)
    //
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int const kIterations = options.iterations;
    float ref_total_ms = 0;
    float ref_min_ms = 1e30f;

    for (int iter = 0; iter < kIterations; ++iter) {
      cudaEventRecord(ev_start);
      kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
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
    //
    // Warmup solution
    //
    for (int i = 0; i < 3; i++) {
      execute_solution_kernel();
      cudaDeviceSynchronize();
    }

    //
    // Profile solution (per-iteration timing)
    //
    float sol_total_ms = 0;
    float sol_min_ms = 1e30f;

    for (int iter = 0; iter < kIterations; ++iter) {
      cudaEventRecord(ev_start);
      execute_solution_kernel();
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      sol_total_ms += ms;
      if (ms < sol_min_ms) sol_min_ms = ms;
    }

    fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            sol_total_ms / kIterations, kIterations, sol_min_ms);

    float speedup = ref_min_ms / sol_min_ms;
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", speedup);
#endif

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(ref_total_ms) / double(kIterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    std::cout << std::endl;
    std::cout << "CUTLASS Attention:\n"
      << "====================================================" << std::endl;
    std::cout << "    " << " {seq length Q, seq length KV, head size, head size V, head number, batch size} = {" << options.seq_length \
      << ", " << options.seq_length_kv << ", " << options.head_size << ", " << options.head_size_v << ", " << options.head_number\
      << ", " << options.batch_size << "}." << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int kQueriesPerBlock,
  int kKeysPerBlock,
  int kMaxK
>
int run_attention(Options& options) {
  using Attention = AttentionKernel<
    cutlass::half_t,      // scalar_t
    cutlass::arch::Sm80,  // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK,
    false,                // Supports dropout
    false                 // Supports bias
  >;

  //
  // Test and profile
  //

  TestbedAttention<Attention> testbed(options);

  Result result = testbed.profile();
  if (!result.passed) {
    return -1;
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
    std::cout
      << "CUTLASS's CUTLASS Attention example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";
    return 0;
  }

  // Parse options for explicit-size mode
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
    int batch_size;
    int seq_length;
    int head_number;
    int head_size;
  };

  std::vector<TestConfig> configs;

  // Check if user specified any size parameters
  bool has_explicit = false;
  for (int i = 1; i < argc; i++) {
    std::string arg(args[i]);
    if (arg.find("batch_size") != std::string::npos ||
        arg.find("seq_length") != std::string::npos ||
        arg.find("head_number") != std::string::npos ||
        arg.find("head_size") != std::string::npos) {
      has_explicit = true;
      break;
    }
  }

  if (has_explicit) {
    configs.push_back({"custom", options.batch_size, options.seq_length,
                       options.head_number, options.head_size});
  } else {
    // Batched: small={128,batch=4}, medium={512,batch=16}, large={2048,batch=4}
    // Using seq_length as the "size" parameter, with appropriate head count
    configs = {
      {"small",   4,  128, 4, 64},
      {"medium", 16,  512, 4, 64},
      {"large",   4, 2048, 4, 64},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: batch=%d, seq_len=%d, heads=%d, head_dim=%d ===\n",
            cfg.label, cfg.batch_size, cfg.seq_length, cfg.head_number, cfg.head_size);

    Options opts;
    opts.batch_size = cfg.batch_size;
    opts.seq_length = cfg.seq_length;
    opts.seq_length_kv = cfg.seq_length;
    opts.head_number = cfg.head_number;
    opts.head_size = cfg.head_size;
    opts.head_size_v = cfg.head_size;
    opts.iterations = options.iterations;
    opts.causal = false;
    opts.use_mask = false;
    opts.alignment = 1;
    opts.reference_check = true;
    opts.randomize_problems();

    int ret;
    if (opts.head_size_v > 64) {
      static int const kQueriesPerBlock = 32;
      static int const kKeysPerBlock = 128;
      if (opts.head_size_v <= 128) {
        ret = run_attention<kQueriesPerBlock, kKeysPerBlock, 128>(opts);
      } else {
        ret = run_attention<kQueriesPerBlock, kKeysPerBlock, 65536>(opts);
      }
    } else {
      static constexpr int kMaxK = 64;
      static int const kQueriesPerBlock = 64;
      static int const kKeysPerBlock = 64;
      ret = run_attention<kQueriesPerBlock, kKeysPerBlock, kMaxK>(opts);
    }

    if (ret != 0) return ret;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
