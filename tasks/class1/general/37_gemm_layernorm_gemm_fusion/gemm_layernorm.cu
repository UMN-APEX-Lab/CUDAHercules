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
    \brief GEMM -> LayerNorm -> GEMM (FP16 data, FP32 accumulation) benchmark harness.

    Column-major formulation of CUTLASS example 37 (transformer block: hidden size H, T tokens):
      C0   = alpha * A0 B0            A0: H x H (row-major), B0: H x T (column-major)
      Norm = LayerNorm over H of every column of C0 (eps 1e-6), scaled by gamma and shifted by beta
      C1   = alpha * A1 Norm          A1: 4H x H (column-major), C1: 4H x T (column-major)

    The reference is CUTLASS GemmLayernorm (gemm_with_layernorm.h: LayerNorm statistics fused into
    the epilogue of the first GEMM and the normalization fused into the mainloop of the second) with
    FP32 accumulation and a per-shape tile configuration. The candidate (solution.h,
    -DKH_TEST_SOLUTION) implements gemm_layernorm_gemm_solution(). Both are verified against a golden
    result: FP32-accumulated GEMMs on the device, LayerNorm statistics in FP64 on the host.

    Usage:
      ./test                                          # all problem sizes
      ./test --hidden_dim=1024 --valid_word_num=4096  # custom size (hidden_dim multiple of 128, valid_word_num multiple of 8)
      ./test --iterations=50                          # profiling iterations (0 disables)
*/

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reference and golden-reference headers (included after the solution on purpose)

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/fast_math.h"
#include "gemm_with_layernorm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

using Element = cutlass::half_t;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS GemmLayernorm, FP32 accumulation, per-shape tile configuration
// (offline sweep on RTX PRO 6000 Blackwell)

template <int TbM, int TbN, int WarpM, int WarpN, int Stages0, int Stages1, int Swizzle>
struct kh_ref_config {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
  using GemmLayernorm = cutlass::GemmLayernorm<
    Element, RowMajor,       // A0
    Element, ColumnMajor,    // B0
    Element, ColumnMajor,    // C0 / C1
    float,                   // accumulation and epilogue compute type
    EpilogueOp,
    cutlass::gemm::GemmShape<TbM, TbN, 32>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Stages0, Stages1,
    false,                   // no shifted variance
    Swizzle>;
};

/// Arguments of one reference invocation (column-major output formulation).
struct kh_ref_problem {
  int hidden, tokens;
  float alpha, beta;
  Element const *A0, *B0, *A1, *gamma, *beta_ln;
  Element *C0, *C1;
  Element *variance, *mean;  // partial statistics, (hidden / 64) x tokens elements each
};

template <typename Config>
cutlass::Status kh_ref_run_config(kh_ref_problem const &p) {
  using G = typename Config::GemmLayernorm;
  int const H = p.hidden, T = p.tokens, N1 = 4 * p.hidden;
  cutlass::gemm::GemmCoord problem0(H, T, H), problem1(N1, T, H);
  typename G::TensorVariance ref_var(p.variance, RowMajor(T));
  typename G::TensorMean ref_mean(p.mean, RowMajor(T));
  typename G::TensorVariance ref_gamma(const_cast<Element *>(p.gamma), RowMajor(H));
  typename G::TensorMean ref_beta(const_cast<Element *>(p.beta_ln), RowMajor(H));
  typename G::Arguments args(
    problem0, problem1,
    const_cast<Element *>(p.A0), const_cast<Element *>(p.B0), p.C0, p.C0,
    const_cast<Element *>(p.A1), p.C1,
    H, H, H, H, N1, N1,
    {p.alpha, p.beta},
    ref_var, ref_mean, ref_gamma, ref_beta, nullptr);
  G op;
  cutlass::Status status = op.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  return op();
}

cutlass::Status kh_ref_run(kh_ref_problem const &p) {
  // 256x128x32 tiles, 3 stages and a threadblock swizzle of 4 for both GEMMs (offline sweep; the
  // CUTLASS example's 128x128 tiles with identity rasterization thrash the L2 cache at H >= 5120).
  return kh_ref_run_config<kh_ref_config<256, 128, 64, 64, 3, 3, 4>>(p);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  // Verification of C1 against the golden result G (non-finite values always fail):
  //   every element: |x - g| <= kAtol * rms(G) + kRtol * |g|;   whole tensor: rms(x - G) <= kRmsTol * rms(G)
  // The reference (FP32 accumulation; C0, LayerNorm statistics and normalized operand rounded to
  // FP16) stays at rms(err)/rms(G) ~ 4.6e-4; FP16 accumulation (1.7e-3 - 3.7e-3), BF16 operands
  // (3.1e-3) and FP8 operands (4.8e-2) exceed the RMS limit.
  static constexpr double kAtol = 0.01;
  static constexpr double kRtol = 4e-3;
  static constexpr double kRmsTol = 1.4e-3;

  int H, T, N1;
  float alpha, beta;
  bool verify_reference;
  int iterations;
  unsigned seed;

  int64_t count_A0, count_B0, count_C0, count_A1, count_C1, count_stats;

  cutlass::DeviceAllocation<Element> block_A0, block_B0, block_C0, block_A1, block_C1;
  cutlass::DeviceAllocation<Element> block_gamma, block_beta, block_var, block_mean;
  cutlass::DeviceAllocation<Element> block_shift;  // optional shift for the variance (zeros)
  std::vector<float> host_golden_C1;
#ifdef KH_TEST_SOLUTION
  cutlass::DeviceAllocation<Element> sol_C1;
#endif

  Testbed(int hidden, int tokens, float alpha_, float beta_, bool verify_ref, int iters, unsigned seed_):
    H(hidden), T(tokens), N1(4 * hidden), alpha(alpha_), beta(beta_), verify_reference(verify_ref),
    iterations(iters), seed(seed_) {
    count_A0 = int64_t(H) * H;
    count_B0 = int64_t(H) * T;
    count_C0 = int64_t(H) * T;
    count_A1 = int64_t(N1) * H;
    count_C1 = int64_t(N1) * T;
    count_stats = int64_t((H + 63) / 64) * T;
  }

  void initialize() {
    block_A0.reset(count_A0);
    block_B0.reset(count_B0);
    block_C0.reset(count_C0);
    block_A1.reset(count_A1);
    block_C1.reset(count_C1);
    block_gamma.reset(H);
    block_beta.reset(H);
    block_var.reset(count_stats);
    block_mean.reset(count_stats);
    block_shift.reset(T);
    cudaMemset(block_shift.get(), 0, T * sizeof(Element));

    // Random values at full FP16 precision: activations and weights uniform in [-1, 1),
    // LayerNorm scale uniform in [0.5, 1.5) and shift uniform in [-0.5, 0.5).
    cutlass::reference::device::BlockFillRandomUniform(block_A0.get(), count_A0, seed, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_B0.get(), count_B0, seed + 1, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_A1.get(), count_A1, seed + 2, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_gamma.get(), H, seed + 3, Element(1.5f), Element(0.5f), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_beta.get(), H, seed + 4, Element(0.5f), Element(-0.5f), -1);

    // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
    cudaMemset(block_C1.get(), 0xFF, count_C1 * sizeof(Element));
#ifdef KH_TEST_SOLUTION
    sol_C1.reset(count_C1);
#endif
  }

  cutlass::Status run_reference_() {
    kh_ref_problem p{H, T, alpha, beta, block_A0.get(), block_B0.get(), block_A1.get(), block_gamma.get(),
                 block_beta.get(), block_C0.get(), block_C1.get(), block_var.get(), block_mean.get()};
    return kh_ref_run(p);
  }

#ifdef KH_TEST_SOLUTION
  cudaError_t run_solution_() {
    // Row-major view of the column-major problem: M_sol = tokens, N0_sol = K0_sol = hidden,
    // N1_sol = 4 * hidden (see description.txt for the layouts).
    return gemm_layernorm_gemm_solution(
      T, H, H, N1,
      alpha, beta,
      reinterpret_cast<const __half*>(block_A0.get()), H,
      reinterpret_cast<const __half*>(block_B0.get()), H,
      reinterpret_cast<const __half*>(block_A1.get()), N1,
      reinterpret_cast<__half*>(sol_C1.get()), N1,
      reinterpret_cast<const __half*>(block_gamma.get()),
      reinterpret_cast<const __half*>(block_beta.get()),
      reinterpret_cast<const __half*>(block_shift.get()),
      true);
  }

  /// Poisons the solution output (NaN), runs the solution once and verifies it.
  bool run_and_verify_solution_(char const *who) {
    cudaMemset(sol_C1.get(), 0xFF, count_C1 * sizeof(Element));
    cudaError_t err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output(who, sol_C1);
  }
#endif

  /// Golden result: C0 = alpha A0 B0 (FP32 accumulation, not rounded), LayerNorm over the hidden
  /// dimension with FP64 statistics on the host, C1 = alpha A1 Norm (FP32 accumulation).
  bool compute_golden() {
    cutlass::DeviceAllocation<float> golden_C0(count_C0), golden_C1(count_C1);
    cudaMemset(golden_C0.get(), 0, count_C0 * sizeof(float));
    cudaMemset(golden_C1.get(), 0, count_C1 * sizeof(float));

    cutlass::reference::device::Gemm<Element, RowMajor, Element, ColumnMajor, float, ColumnMajor, float, float> gemm0;
    cutlass::reference::device::Gemm<Element, ColumnMajor, float, ColumnMajor, float, ColumnMajor, float, float> gemm1;

    cutlass::TensorRef<float, ColumnMajor> ref_C0(golden_C0.get(), ColumnMajor(H));
    cutlass::TensorRef<float, ColumnMajor> ref_C1(golden_C1.get(), ColumnMajor(N1));

    gemm0({H, T, H}, alpha,
          cutlass::TensorRef<Element, RowMajor>(block_A0.get(), RowMajor(H)),
          cutlass::TensorRef<Element, ColumnMajor>(block_B0.get(), ColumnMajor(H)),
          0.0f, ref_C0, ref_C0);

    std::vector<float> host_C0(count_C0);
    std::vector<Element> host_gamma(H), host_beta(H);
    cutlass::device_memory::copy_to_host(host_C0.data(), golden_C0.get(), count_C0);
    cutlass::device_memory::copy_to_host(host_gamma.data(), block_gamma.get(), H);
    cutlass::device_memory::copy_to_host(host_beta.data(), block_beta.get(), H);

    // LayerNorm over the hidden dimension (column n of C0 = token n), two-pass, FP64 statistics
    for (int n = 0; n < T; ++n) {
      float *col = host_C0.data() + int64_t(n) * H;
      double sum = 0;
      for (int m = 0; m < H; ++m) sum += col[m];
      double const mean = sum / H;
      double sq = 0;
      for (int m = 0; m < H; ++m) {
        double const d = col[m] - mean;
        sq += d * d;
      }
      double const inv_std = 1.0 / std::sqrt(sq / H + 1e-6);
      for (int m = 0; m < H; ++m) {
        col[m] = float((col[m] - mean) * inv_std * double(float(host_gamma[m])) + double(float(host_beta[m])));
      }
    }
    cutlass::device_memory::copy_to_device(golden_C0.get(), host_C0.data(), count_C0);

    gemm1({N1, T, H}, alpha,
          cutlass::TensorRef<Element, ColumnMajor>(block_A1.get(), ColumnMajor(N1)),
          cutlass::TensorRef<float, ColumnMajor>(golden_C0.get(), ColumnMajor(H)),
          0.0f, ref_C1, ref_C1);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Golden reference failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    host_golden_C1.resize(count_C1);
    cutlass::device_memory::copy_to_host(host_golden_C1.data(), golden_C1.get(), count_C1);
    return true;
  }

  /// Checks every element of C1 and the aggregate error against the golden result.
  bool check_output(char const *who, cutlass::DeviceAllocation<Element> const &out) {
    std::vector<Element> x(count_C1);
    cutlass::device_memory::copy_to_host(x.data(), out.get(), count_C1);

    double gold_sq = 0;
    for (int64_t i = 0; i < count_C1; ++i) gold_sq += double(host_golden_C1[i]) * double(host_golden_C1[i]);
    double const rms = std::sqrt(gold_sq / double(count_C1));
    double const atol = kAtol * rms;

    double max_ratio = 0, err_sq = 0;
    int64_t bad = 0;
    for (int64_t i = 0; i < count_C1; ++i) {
      double const v = double(float(x[i]));
      double const g = double(host_golden_C1[i]);
      double err = std::fabs(v - g);
      double const tol = atol + kRtol * std::fabs(g);
      if (!std::isfinite(v) || !(err <= tol)) {
        if (bad++ == 0) {
          fprintf(stderr, "%s: C1 mismatch at row %lld, col %lld: got %g, expected %g (tol %g)\n",
                  who, (long long)(i % N1), (long long)(i / N1), v, g, tol);
        }
        if (!std::isfinite(v)) err = INFINITY;
      }
      max_ratio = std::max(max_ratio, err / tol);
      err_sq += err * err;
    }
    double const rel_rms_err = std::sqrt(err_sq / double(count_C1)) / rms;
    printf("%s: max |err|/tol = %.3f, rms(err)/rms(C1) = %.3g (limit %.3g), rms(C1) = %.4g, "
           "%lld of %lld elements out of tolerance\n",
           who, max_ratio, rel_rms_err, kRmsTol, rms, (long long)bad, (long long)count_C1);
    if (bad) {
      return false;
    }
    if (!(rel_rms_err <= kRmsTol)) {
      fprintf(stderr, "%s: C1 RMS error too large: %g > %g\n", who, rel_rms_err, kRmsTol);
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
    for (int iter = 0; iter < iterations; ++iter) {
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
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n", label, total_ms / iterations, iterations, min_ms);
    return min_ms;
  }

  bool run() {
    initialize();

    cutlass::Status status = run_reference_();
    cudaError_t err = cudaDeviceSynchronize();
    if (status != cutlass::Status::kSuccess || err != cudaSuccess) {
      std::cerr << "Reference execution failed: " << cutlass::cutlassGetStatusString(status) << ", "
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

#ifdef KH_TEST_SOLUTION
    bool const need_golden = true;
#else
    bool const need_golden = verify_reference;
#endif
    if (need_golden && !compute_golden()) {
      return false;
    }
    if (verify_reference && !check_output("Reference", block_C1)) {
      std::cerr << "Reference verification failed" << std::endl;
      return false;
    }

#ifdef KH_TEST_SOLUTION
    if (!run_and_verify_solution_("Solution")) {
      return false;
    }
#endif

    std::cout << "Passed" << std::endl;
    if (iterations <= 0) {
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
    double const flops = 2.0 * T * double(H) * H + 2.0 * T * double(N1) * H;
    printf("Reference: %.1f TFLOP/s\n", flops / ref_min_ms / 1e9);
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  cutlass::CommandLine cmd(argc, argv);
  if (cmd.check_cmd_line_flag("help")) {
    std::cout << "37_gemm_layernorm_gemm_fusion\n\n"
              << "  --hidden_dim=<int>        hidden size (multiple of 128)\n"
              << "  --valid_word_num=<int>    number of tokens (multiple of 8)\n"
              << "  --iterations=<int>        profiling iterations (default 20, 0 disables)\n"
              << "  --verify=<bool>           verify the reference too (default true)\n"
              << "  --seed=<int>\n";
    return 0;
  }

  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess || props.major < 8) {
    std::cerr << "This benchmark requires a GPU with compute capability 80 or greater." << std::endl;
    return 0;
  }

  int iterations = 20;
  bool verify = true;
  unsigned seed = 2022;
  cmd.get_cmd_line_argument("iterations", iterations, 20);
  cmd.get_cmd_line_argument("verify", verify, true);
  cmd.get_cmd_line_argument("seed", seed, 2022u);

  struct TestConfig {
    const char* label;
    int hidden_dim;
    int valid_word_num;
  };

  std::vector<TestConfig> configs;
  if (cmd.check_cmd_line_flag("hidden_dim") || cmd.check_cmd_line_flag("valid_word_num")) {
    TestConfig c{"custom", 768, 4096};
    cmd.get_cmd_line_argument("hidden_dim", c.hidden_dim, 768);
    cmd.get_cmd_line_argument("valid_word_num", c.valid_word_num, 4096);
    configs.push_back(c);
  } else {
    // Transformer blocks as run on one RTX PRO 6000 (96 GB): hidden size x tokens, FFN = 4 x hidden
    //   bert_large_train: BERT-large training, hidden 1024, 64 sequences x 512 tokens
    //   qwen3_8b:         Qwen3-8B width (hidden 4096), 16K-token prefill chunk
    //   qwen3_32b:        Qwen3-32B width (hidden 5120), 16K-token prefill chunk
    //   gemma3_27b:       Gemma-3-27B width (hidden 5376), 16K-token prefill chunk
    configs = {
      {"bert_large_train", 1024, 32768},
      {"qwen3_8b",         4096, 16384},
      {"qwen3_32b",        5120, 16384},
      {"gemma3_27b",       5376, 16384},
    };
  }

  for (auto const &cfg : configs) {
    fprintf(stdout, "\n=== %s: hidden_dim=%d, valid_word_num=%d ===\n", cfg.label, cfg.hidden_dim, cfg.valid_word_num);
    fflush(stdout);
    if (cfg.hidden_dim % 128 || cfg.valid_word_num % 8 || cfg.hidden_dim <= 0 || cfg.valid_word_num <= 0) {
      std::cerr << "hidden_dim must be a multiple of 128 and valid_word_num a multiple of 8" << std::endl;
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    Testbed testbed(cfg.hidden_dim, cfg.valid_word_num, 1.0f, 0.0f, verify, iterations, seed);
    if (!testbed.run()) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
