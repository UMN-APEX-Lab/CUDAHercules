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
    \brief Fused multi-head attention (FP16, BMHK layout) benchmark harness.

    O = softmax(Q K^T / sqrt(head_dim) [+ causal mask]) V for every (batch, head).

    The reference is FlashAttention-2 (fa2_ref.cu, built from reference_sources/flash_attn); the
    candidate (solution.h, -DKH_TEST_SOLUTION) implements fmha_solution(). Both are checked
    element-wise against an FP32 golden result computed on the device (S = Q K^T in FP32,
    row-wise softmax in FP32, O = P V in FP32).

    Usage:
      ./test                                   # all problem sizes
      ./test --batch_size=4 --seq_length=256 --head_number=8 --head_size=64 [--causal=true]
      ./test --iterations=50

    Originally derived from CUTLASS example 41 (fixed sequence length FMHA).
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <limits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reference and golden-reference headers (included after the solution on purpose)

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "fa2_ref.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Golden-reference helper: in-place row-wise softmax (FP32) of a row-major [rows x cols] matrix.
/// Columns >= n_valid (and, if causal, > row) are masked out and set to 0.
__global__ void golden_softmax_rows_kernel(float *P, int64_t ldp, int cols, int n_valid, bool causal) {
  __shared__ float red[32];
  __shared__ float bcast;
  int const row = blockIdx.x;
  float *p = P + int64_t(row) * ldp;
  int const n_row = causal ? min(row + 1, n_valid) : n_valid;
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;

  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n_row; j += blockDim.x) mx = fmaxf(mx, p[j]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = red[0];
    for (int w = 1; w < nwarps; ++w) v = fmaxf(v, red[w]);
    bcast = v;
  }
  __syncthreads();
  mx = bcast;
  __syncthreads();

  float sum = 0.f;
  for (int j = threadIdx.x; j < n_row; j += blockDim.x) sum += expf(p[j] - mx);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = 0.f;
    for (int w = 0; w < nwarps; ++w) v += red[w];
    bcast = 1.f / v;
  }
  __syncthreads();
  float const inv_sum = bcast;

  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    p[j] = (j < n_row) ? expf(p[j] - mx) * inv_sum : 0.f;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct ProblemConfig {
  char const *label;
  int batch_size;
  int seq_length;
  int head_number;
  int head_size;
  bool causal;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

class TestbedAttention {
public:

  using Element = cutlass::half_t;
  using LayoutQ = cutlass::layout::RowMajor;     // [seq, head_dim] slice of BMHK, ld = heads * head_dim
  using LayoutK = cutlass::layout::ColumnMajor;  // K^T: [head_dim, seq], ld = heads * head_dim
  using LayoutP = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  // Input distributions: Q, K uniform in [-kQKRange, kQKRange) and V uniform in [-1, 1), at full
  // FP16 precision. The attention logits Q.K / sqrt(head_dim) then have a standard deviation of
  // kQKRange^2 / 3 = 3 (row maxima of about 10-14), i.e. realistic, far from one-hot softmax rows.
  static constexpr float kQKRange = 3.0f;

  // Verification: every element |O - gold| <= kAtol + kRtol * |gold| (non-finite values fail), and
  // rms(O - gold) <= kRmsTol * rms(gold). The golden result is FP32 end to end. The reference
  // (FP16 tensor cores, FP32 accumulation) stays at rms(err)/rms(gold) ~ 2.2e-4 and max |err| ~ 4e-4,
  // i.e. FP16 output rounding; FP16 accumulation (1.4e-3 - 3e-3 at the long-context sizes), BF16 or
  // FP8 operands (>= 5e-3) exceed the limits.
  static constexpr double kAtol = 1e-3;
  static constexpr double kRtol = 5e-3;
  static constexpr double kRmsTol = 7e-4;

private:

  ProblemConfig cfg_;
  int iterations_;
  bool reference_check_;
  uint32_t seed_;

  int64_t stride_h_, stride_m_, stride_b_;
  int64_t count_;
  float scale_;

  cutlass::DeviceAllocation<Element> block_Q;
  cutlass::DeviceAllocation<Element> block_K;
  cutlass::DeviceAllocation<Element> block_V;
  cutlass::DeviceAllocation<Element> block_O;
  cutlass::DeviceAllocation<float> block_LSE;
  cutlass::DeviceAllocation<float> block_Gold_O;
  std::vector<float> host_Gold_O;

#ifdef KH_TEST_SOLUTION
  cutlass::DeviceAllocation<Element> sol_O;
#endif

public:

  TestbedAttention(ProblemConfig const &cfg, int iterations, bool reference_check, uint32_t seed = 3080):
    cfg_(cfg), iterations_(iterations), reference_check_(reference_check), seed_(seed) {
    stride_h_ = cfg.head_size;
    stride_m_ = int64_t(cfg.head_number) * cfg.head_size;
    stride_b_ = stride_m_ * cfg.seq_length;
    count_ = stride_b_ * cfg.batch_size;
    scale_ = 1.0f / std::sqrt(float(cfg.head_size));
  }

private:

  void initialize_() {
    block_Q.reset(count_);
    block_K.reset(count_);
    block_V.reset(count_);
    block_O.reset(count_);
    block_LSE.reset(int64_t(cfg_.batch_size) * cfg_.head_number * cfg_.seq_length);

    cutlass::reference::device::BlockFillRandomUniform(block_Q.get(), count_, seed_ + 1, Element(kQKRange), Element(-kQKRange), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_K.get(), count_, seed_ + 2, Element(kQKRange), Element(-kQKRange), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_V.get(), count_, seed_ + 3, Element(1.0f), Element(-1.0f), -1);

    // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
    cudaMemset(block_O.get(), 0xFF, count_ * sizeof(Element));
#ifdef KH_TEST_SOLUTION
    sol_O.reset(count_);
#endif
  }

  cudaError_t run_reference_() {
    return kh_ref_fmha_fwd(
      cfg_.batch_size, cfg_.seq_length, cfg_.seq_length, cfg_.head_number, cfg_.head_size,
      block_Q.get(), block_K.get(), block_V.get(), block_O.get(), block_LSE.get(),
      stride_b_, stride_m_, stride_h_,
      stride_b_, stride_m_, stride_h_,
      stride_b_, stride_m_, stride_h_,
      stride_b_, stride_m_, stride_h_,
      scale_, cfg_.causal, nullptr);
  }

#ifdef KH_TEST_SOLUTION
  cudaError_t run_solution_() {
    return fmha_solution(
      cfg_.batch_size, cfg_.seq_length, cfg_.seq_length,
      cfg_.head_number, cfg_.head_size, cfg_.head_size,
      reinterpret_cast<const __half*>(block_Q.get()),
      reinterpret_cast<const __half*>(block_K.get()),
      reinterpret_cast<const __half*>(block_V.get()),
      reinterpret_cast<__half*>(sol_O.get()),
      int(stride_b_), int(stride_m_), int(stride_h_),
      int(stride_b_), int(stride_m_), int(stride_h_),
      int(stride_b_), int(stride_m_), int(stride_h_),
      int(stride_m_),
      cfg_.causal);
  }

  /// Poisons the solution output, runs the solution once and verifies it.
  bool run_and_verify_solution_(char const *who) {
    cudaMemset(sol_O.get(), 0xFF, count_ * sizeof(Element));
    cudaError_t err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output_(who, sol_O);
  }
#endif

  /// Golden result on the device, one (batch, head) at a time:
  ///   S = scale * Q K^T (FP32), P = row-softmax(S) with causal mask (FP32), O = P V (FP32)
  bool compute_golden_() {
    int const S = cfg_.seq_length;
    int const D = cfg_.head_size;
    block_Gold_O.reset(count_);
    cutlass::DeviceAllocation<float> block_P(int64_t(S) * S);

    for (int b = 0; b < cfg_.batch_size; ++b) {
      for (int h = 0; h < cfg_.head_number; ++h) {
        int64_t const offset = b * stride_b_ + h * stride_h_;
        cutlass::TensorRef<Element, LayoutQ> ref_Q(block_Q.get() + offset, LayoutQ(stride_m_));
        cutlass::TensorRef<Element, LayoutK> ref_K(block_K.get() + offset, LayoutK(stride_m_));
        cutlass::TensorRef<Element, LayoutV> ref_V(block_V.get() + offset, LayoutV(stride_m_));
        cutlass::TensorRef<float, LayoutP> ref_P(block_P.get(), LayoutP(S));
        cutlass::TensorRef<float, LayoutO> ref_O(block_Gold_O.get() + offset, LayoutO(stride_m_));

        cutlass::reference::device::GemmComplex<
            Element, LayoutQ, Element, LayoutK, float, LayoutP, float, float>(
          {S, S, D}, scale_, ref_Q, cutlass::ComplexTransform::kNone,
          ref_K, cutlass::ComplexTransform::kNone, 0.0f, ref_P, ref_P, 0.0f);

        golden_softmax_rows_kernel<<<S, 256>>>(block_P.get(), S, S, S, cfg_.causal);

        cutlass::reference::device::GemmComplex<
            float, LayoutP, Element, LayoutV, float, LayoutO, float, float>(
          {S, D, S}, 1.0f, ref_P, cutlass::ComplexTransform::kNone,
          ref_V, cutlass::ComplexTransform::kNone, 0.0f, ref_O, ref_O, 0.0f);
      }
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Golden reference failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    host_Gold_O.resize(count_);
    cutlass::device_memory::copy_to_host(host_Gold_O.data(), block_Gold_O.get(), count_);
    return true;
  }

  /// Element-wise check of an output against the golden result (scans every element, reports the
  /// first mismatch and the error statistics).
  bool check_output_(char const *who, cutlass::DeviceAllocation<Element> const &out) {
    std::vector<Element> x(count_);
    cutlass::device_memory::copy_to_host(x.data(), out.get(), count_);
    double max_err = 0, max_ratio = 0, err_sq = 0, gold_sq = 0;
    int64_t bad = 0;
    for (int64_t i = 0; i < count_; ++i) {
      double v = double(float(x[i]));
      double g = double(host_Gold_O[i]);
      double err = std::fabs(v - g);
      double tol = kAtol + kRtol * std::fabs(g);
      if (!std::isfinite(v) || !(err <= tol)) {
        if (bad++ == 0) {
          int64_t const b = i / stride_b_, m = (i % stride_b_) / stride_m_;
          int64_t const h = (i % stride_m_) / stride_h_, d = i % stride_h_;
          fprintf(stderr, "%s: mismatch at batch %lld, row %lld, head %lld, dim %lld: got %g, expected %g (tol %g)\n",
                  who, (long long)b, (long long)m, (long long)h, (long long)d, v, g, tol);
        }
        if (!std::isfinite(v)) {
          err = INFINITY;
        }
      }
      max_err = std::max(max_err, err);
      max_ratio = std::max(max_ratio, err / tol);
      err_sq += err * err;
      gold_sq += g * g;
    }
    double const rel_rms = std::sqrt(err_sq / std::max(gold_sq, 1e-300));
    printf("%s: max |err| = %.3g, max |err|/tol = %.3f, rms(err)/rms(ref) = %.3g (limit %.3g), "
           "rms(ref) = %.3g, %lld of %lld elements out of tolerance\n",
           who, max_err, max_ratio, rel_rms, kRmsTol, std::sqrt(gold_sq / double(count_)),
           (long long)bad, (long long)count_);
    if (bad) {
      return false;
    }
    if (!(rel_rms <= kRmsTol)) {
      fprintf(stderr, "%s: RMS error too large: %g > %g\n", who, rel_rms, kRmsTol);
      return false;
    }
    return true;
  }

  template <typename Fn>
  float time_min_ms_(Fn &&fn, char const *label) {
    for (int i = 0; i < 3; ++i) {
      fn();
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0, min_ms = 1e30f;
    for (int iter = 0; iter < iterations_; ++iter) {
      cudaEventRecord(start);
      fn();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
      min_ms = std::min(min_ms, ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            label, total_ms / iterations_, iterations_, min_ms);
    return min_ms;
  }

public:

  /// Verifies and profiles. Returns true if everything that was checked passed.
  bool run() {

    if (!kh_ref_fmha_supported(cfg_.head_size)) {
      std::cerr << "Reference supports head_size 64 or 128 only" << std::endl;
      return false;
    }

    initialize_();

    cudaError_t err = run_reference_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << "Reference kernel failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }

#ifdef KH_TEST_SOLUTION
    bool const need_golden = true;
#else
    bool const need_golden = reference_check_;
#endif
    if (need_golden && !compute_golden_()) {
      return false;
    }

    if (reference_check_ && !check_output_("Reference", block_O)) {
      std::cerr << "Reference verification failed" << std::endl;
      return false;
    }

#ifdef KH_TEST_SOLUTION
    if (!run_and_verify_solution_("Solution")) {
      return false;
    }
#endif

    std::cout << "Passed" << std::endl;

    if (iterations_ <= 0) {
      return true;
    }

    float const ref_min_ms = time_min_ms_([&]() { run_reference_(); }, "Ref time");

#ifdef KH_TEST_SOLUTION
    float const sol_min_ms = time_min_ms_([&]() { run_solution_(); }, "Kernel time");
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);

    // The output must still be produced by a fresh call after the timed runs.
    if (!run_and_verify_solution_("Solution (re-run after timing)")) {
      return false;
    }
#endif

    double const flops = 4.0 * cfg_.batch_size * cfg_.head_number * double(cfg_.seq_length) *
                         cfg_.seq_length * cfg_.head_size * (cfg_.causal ? 0.5 : 1.0);
    printf("Reference: %.1f TFLOP/s\n", flops / ref_min_ms / 1e9);
    return true;
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
  if (props.major < 8) {
    std::cout << "This benchmark requires compute capability 80 or greater.\n";
    return 0;
  }

  cutlass::CommandLine cmd(argc, args);
  if (cmd.check_cmd_line_flag("help")) {
    std::cout << "41_fused_multi_head_attention\n\n"
              << "  --batch_size=<int> --seq_length=<int> --head_number=<int> --head_size=<64|128>\n"
              << "  --causal=<bool>           causal masking (custom size only, default false)\n"
              << "  --iterations=<int>        profiling iterations (default 20, 0 disables)\n"
              << "  --reference-check=<bool>  verify the reference too (default true)\n";
    return 0;
  }

  int iterations = 20;
  bool reference_check = true;
  cmd.get_cmd_line_argument("iterations", iterations, 20);
  cmd.get_cmd_line_argument("reference-check", reference_check, true);

  std::vector<ProblemConfig> configs;
  bool const has_explicit = cmd.check_cmd_line_flag("batch_size") || cmd.check_cmd_line_flag("seq_length") ||
                            cmd.check_cmd_line_flag("head_number") || cmd.check_cmd_line_flag("head_size");

  if (has_explicit) {
    ProblemConfig c{"custom", 16, 1024, 12, 64, false};
    cmd.get_cmd_line_argument("batch_size", c.batch_size, 16);
    cmd.get_cmd_line_argument("seq_length", c.seq_length, 1024);
    cmd.get_cmd_line_argument("head_number", c.head_number, 12);
    cmd.get_cmd_line_argument("head_size", c.head_size, 64);
    cmd.get_cmd_line_argument("causal", c.causal, false);
    configs.push_back(c);
  } else {
    // Self-attention as run on one RTX PRO 6000 (96 GB), {batch, seq, heads, head_dim, causal}:
    //   bert_large_train:        BERT-large encoder training, 64 sequences x 512 tokens, 16 heads x 64
    //   vit_l14:                 ViT-L/14 at 224 px (256 patches + CLS = 257 tokens), batch 64, 16 heads x 64
    //   qwen3_32b_prefill_16k:   Qwen3-32B prefill of a 16K-token prompt, 64 query heads x 128, causal
    //                            (GQA K/V heads expanded to one per query head)
    //   gemma3_27b_prefill_32k:  Gemma-3-27B global-attention prefill of a 32K-token prompt,
    //                            32 query heads x 128, causal (K/V expanded per head)
    configs = {
      {"bert_large_train",       64,   512, 16,  64, false},
      {"vit_l14",                64,   257, 16,  64, false},
      {"qwen3_32b_prefill_16k",   1, 16384, 64, 128, true},
      {"gemma3_27b_prefill_32k",  1, 32768, 32, 128, true},
    };
  }

  for (auto const &cfg : configs) {
    fprintf(stdout, "\n=== %s: batch=%d, seq_len=%d, heads=%d, head_dim=%d, causal=%d ===\n",
            cfg.label, cfg.batch_size, cfg.seq_length, cfg.head_number, cfg.head_size, int(cfg.causal));
    fflush(stdout);

    TestbedAttention testbed(cfg, iterations, reference_check);
    if (!testbed.run()) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
