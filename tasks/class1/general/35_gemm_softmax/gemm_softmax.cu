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
    \brief Batched GEMM + row-wise softmax (FP16 in/out, FP32 accumulation) benchmark harness.

    For each batch: D = alpha * A B + beta * C (A row-major, B column-major, C/D row-major),
    Softmax = row-wise softmax of D. The reference is CUTLASS GemmSoftmax (gemm_with_softmax.h,
    from CUTLASS example 35) with a per-shape tile / apply-kernel configuration; the candidate
    (solution.h, -DKH_TEST_SOLUTION) implements gemm_softmax_solution(). Both are verified on the
    device against an FP32 golden result (FP32-accumulated GEMM, softmax of the FP32 logits).

    Usage:
      ./test                                              # all problem sizes
      ./test --m=1024 --n=512 --k=256 --batch_count=8     # custom size [--alpha= --beta=]
      ./test --iterations=50                              # profiling iterations (0 disables)
*/

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "gemm_with_softmax.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementInput = cutlass::half_t;   // A, B, C
using ElementOutput = cutlass::half_t;  // D, Softmax

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS GemmSoftmax with a per-shape configuration (offline sweep on RTX PRO 6000)

template <int TbM, int TbN, int WarpM, int WarpN, int ApplyRows, int ApplyCols>
struct kh_ref_config {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, float, float>;
  using GemmSoftmax = cutlass::GemmSoftmax<
    ElementInput, cutlass::layout::RowMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM, TbN, 32>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    3,
    cutlass::MatrixShape<ApplyRows, ApplyCols>>;
};

/// Arguments of one reference invocation (all matrices packed, strided batches).
struct kh_ref_problem {
  int m, n, k, batch;
  float alpha, beta;
  ElementInput const *A, *B, *C;
  ElementOutput *D, *Softmax;
  float *norm, *sum;  // partial-reduction workspaces of max_partials(n) * m * batch floats
};

/// Largest number of per-row partial reductions any configuration needs (tile N >= 128).
inline int64_t max_partials(int n) { return (n + 127) / 128; }

template <typename Config>
cutlass::Status kh_ref_run_config(kh_ref_problem const &p) {
  using G = typename Config::GemmSoftmax;
  int64_t const block_num = (p.n + G::ThreadblockShape::kN - 1) / G::ThreadblockShape::kN;
  int64_t const mn = int64_t(p.m) * p.n;
  typename G::Arguments args(
    {p.m, p.n, p.k}, p.batch,
    {const_cast<ElementInput *>(p.A), p.k},
    {const_cast<ElementInput *>(p.B), p.k},
    {const_cast<ElementInput *>(p.C), p.n},
    {p.D, p.n},
    {p.alpha, p.beta},
    {p.norm, p.m},
    {p.sum, p.m},
    {p.Softmax, p.n},
    int64_t(p.m) * p.k, int64_t(p.k) * p.n, mn, mn,
    block_num * p.m, block_num * p.m, mn);
  G op;
  cutlass::Status status = op.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  return op();
}

/// Tile: 64x256 for skinny M (decode), 128x128 for N <= 128 (router), else 128x256.
/// Apply kernel: 4x32 threads for N <= 256, 2x64 for N <= 1024, else 1x256.
template <int TbM, int TbN, int WarpM, int WarpN>
cutlass::Status kh_ref_run_tile(kh_ref_problem const &p) {
  if (p.n <= 256) return kh_ref_run_config<kh_ref_config<TbM, TbN, WarpM, WarpN, 4, 32>>(p);
  if (p.n <= 1024) return kh_ref_run_config<kh_ref_config<TbM, TbN, WarpM, WarpN, 2, 64>>(p);
  return kh_ref_run_config<kh_ref_config<TbM, TbN, WarpM, WarpN, 1, 256>>(p);
}

cutlass::Status kh_ref_run(kh_ref_problem const &p) {
  if (p.m <= 64) return kh_ref_run_tile<64, 256, 32, 64>(p);
  if (p.n <= 128) return kh_ref_run_tile<128, 128, 64, 64>(p);
  return kh_ref_run_tile<128, 256, 64, 64>(p);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side verification against the FP32 golden result

// Tolerances. D: |x - g| <= kAtolD + kRtolD * |g|. Softmax: |x - s| <= kAtolS + kRtolS * |s|, with
// s = softmax of the FP32 golden logits; every Softmax row must sum to 1 within kRowSumTol.
// Non-finite values always fail. The reference (FP32 accumulation, FP16 outputs, softmax of the
// FP16-rounded D) reaches at most 0.23 of the D tolerance, 0.27 of the Softmax tolerance and a row-sum
// error of 2.7e-4; FP16 accumulation exceeds the D tolerance by 3-21x, BF16 operands by 13-16x.
static constexpr float kAtolD = 1e-3f;
static constexpr float kRtolD = 2e-3f;
static constexpr float kAtolS = 1e-6f;
static constexpr float kRtolS = 1.5e-2f;
static constexpr float kRowSumTol = 1e-3f;

struct RowCheck {
  float max_ratio_d;      // max |err| / tol over the row (D)
  float max_ratio_s;      // max |err| / tol over the row (Softmax)
  double row_sum;         // sum of the candidate's Softmax row
  int fails;              // elements out of tolerance (D or Softmax)
  int first_col;          // first failing column (or -1)
};

__device__ __forceinline__ float block_reduce_max(float v, float *red) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
  for (int w = 1; w < int(blockDim.x >> 5); ++w) r = fmaxf(r, red[w]);
  return r;
}

__device__ __forceinline__ double block_reduce_sum(double v, double *red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  double r = 0;
  for (int w = 0; w < int(blockDim.x >> 5); ++w) r += red[w];
  return r;
}

/// One block per row: golden softmax of the FP32 golden row G, then element-wise checks of D and
/// Softmax (FP16) and the row sum.
__global__ void check_rows_kernel(int N, ElementOutput const *D, ElementOutput const *S,
                                  float const *G, int64_t ld, RowCheck *out) {
  __shared__ float redf[32];
  __shared__ double redd[32];
  __shared__ int first_bad;
  int const row = blockIdx.x;
  D += int64_t(row) * ld;
  S += int64_t(row) * ld;
  G += int64_t(row) * ld;

  float mx = -INFINITY;
  for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, G[j]);
  mx = block_reduce_max(mx, redf);
  double se = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) se += exp(double(G[j]) - double(mx));
  double const sum_exp = block_reduce_sum(se, redd);

  if (threadIdx.x == 0) first_bad = N;
  __syncthreads();

  float rd = 0, rs = 0;
  double row_sum = 0;
  int fails = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float const g = G[j];
    float const gs = float(exp(double(g) - double(mx)) / sum_exp);
    float const d = float(D[j]);
    float const s = float(S[j]);
    float const td = kAtolD + kRtolD * fabsf(g);
    float const ts = kAtolS + kRtolS * fabsf(gs);
    float const ed = fabsf(d - g), es = fabsf(s - gs);
    bool const bad = !isfinite(d) || !isfinite(s) || !(ed <= td) || !(es <= ts);
    if (bad) {
      ++fails;
      atomicMin(&first_bad, j);
    }
    rd = fmaxf(rd, isfinite(d) ? ed / td : INFINITY);
    rs = fmaxf(rs, isfinite(s) ? es / ts : INFINITY);
    row_sum += double(s);
  }
  rd = block_reduce_max(rd, redf);
  rs = block_reduce_max(rs, redf);
  row_sum = block_reduce_sum(row_sum, redd);
  double const nf = block_reduce_sum(double(fails), redd);
  if (threadIdx.x == 0) {
    out[row] = RowCheck{rd, rs, row_sum, int(nf), first_bad < N ? first_bad : -1};
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help = false;
  int m = 16, n = 24, k = 64, batch_count = 16;
  int iterations = 20;
  unsigned seed = 2022;
  float alpha = 1, beta = 0;
  bool verification_enabled = true;
  bool alpha_set = false, beta_set = false, explicit_size = false;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    help = cmd.check_cmd_line_flag("help");
    explicit_size = cmd.check_cmd_line_flag("m") || cmd.check_cmd_line_flag("n") ||
                    cmd.check_cmd_line_flag("k") || cmd.check_cmd_line_flag("batch_count");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("batch_count", batch_count);
    alpha_set = cmd.check_cmd_line_flag("alpha");
    beta_set = cmd.check_cmd_line_flag("beta");
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
  }

  std::ostream &print_usage(std::ostream &out) const {
    out << "35_gemm_softmax\n\n"
        << "  --m=<int> --n=<int> --k=<int> --batch_count=<int>   custom problem size\n"
        << "  --alpha=<f32>       (default 1/sqrt(K))   --beta=<f32>   (default 0 for a custom size)\n"
        << "  --iterations=<int>  profiling iterations (default 20, 0 disables)\n"
        << "  --verify=<bool>     verify the reference too (default true)\n"
        << "  --seed=<int>\n";
    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  int M, N, K, batch;
  float alpha, beta;
  bool verify_reference;
  int iterations;
  unsigned seed;

  int64_t mn, count_A, count_B, count_C;

  cutlass::DeviceAllocation<ElementInput> block_A, block_B, block_C;
  cutlass::DeviceAllocation<ElementOutput> ref_D, ref_Softmax;
  cutlass::DeviceAllocation<float> block_Norm, block_Sum;
  cutlass::DeviceAllocation<float> gold_D;       // one batch, FP32
  cutlass::DeviceAllocation<RowCheck> row_checks;
#ifdef KH_TEST_SOLUTION
  cutlass::DeviceAllocation<ElementOutput> sol_D, sol_Softmax;
#endif

  Testbed(int m, int n, int k, int b, float alpha_, float beta_, bool verify_ref, int iters, unsigned seed_):
    M(m), N(n), K(k), batch(b), alpha(alpha_), beta(beta_), verify_reference(verify_ref),
    iterations(iters), seed(seed_) {
    mn = int64_t(M) * N;
    count_A = int64_t(M) * K * batch;
    count_B = int64_t(K) * N * batch;
    count_C = mn * batch;
  }

  void initialize() {
    block_A.reset(count_A);
    block_B.reset(count_B);
    block_C.reset(count_C);
    ref_D.reset(count_C);
    ref_Softmax.reset(count_C);
    block_Norm.reset(max_partials(N) * M * batch);
    block_Sum.reset(max_partials(N) * M * batch);

    // Uniform random values in [-2, 2) at full FP16 precision.
    cutlass::reference::device::BlockFillRandomUniform(block_A.get(), count_A, seed, ElementInput(2), ElementInput(-2), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_B.get(), count_B, seed + 1, ElementInput(2), ElementInput(-2), -1);
    cutlass::reference::device::BlockFillRandomUniform(block_C.get(), count_C, seed + 2, ElementInput(2), ElementInput(-2), -1);

    // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
    cudaMemset(ref_D.get(), 0xFF, count_C * sizeof(ElementOutput));
    cudaMemset(ref_Softmax.get(), 0xFF, count_C * sizeof(ElementOutput));
#ifdef KH_TEST_SOLUTION
    sol_D.reset(count_C);
    sol_Softmax.reset(count_C);
#endif
  }

  cutlass::Status run_reference_() {
    kh_ref_problem p{M, N, K, batch, alpha, beta, block_A.get(), block_B.get(), block_C.get(),
                 ref_D.get(), ref_Softmax.get(), block_Norm.get(), block_Sum.get()};
    return kh_ref_run(p);
  }

#ifdef KH_TEST_SOLUTION
  cudaError_t run_solution_() {
    return gemm_softmax_solution(
      M, N, K, alpha, beta,
      reinterpret_cast<const __half*>(block_A.get()), K,
      reinterpret_cast<const __half*>(block_B.get()), K,
      reinterpret_cast<const __half*>(block_C.get()), N,
      reinterpret_cast<__half*>(sol_D.get()), N,
      reinterpret_cast<__half*>(sol_Softmax.get()), N,
      batch,
      int64_t(M) * K, int64_t(K) * N, mn, mn, mn);
  }
#endif

  /// Verifies the given outputs (all batches) against the golden result.
  struct Outputs { char const *who; ElementOutput const *D; ElementOutput const *S; };

  bool verify(std::vector<Outputs> const &outs) {
    gold_D.reset(mn);
    row_checks.reset(M);
    std::vector<RowCheck> host(M);
    std::vector<bool> ok(outs.size(), true);
    std::vector<double> max_rd(outs.size(), 0), max_rs(outs.size(), 0), max_sum_err(outs.size(), 0);
    std::vector<long long> nfail(outs.size(), 0);

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    for (int b = 0; b < batch; ++b) {
      cutlass::reference::device::GemmComplex<
          ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC, float, float, float>(
        {M, N, K}, alpha,
        cutlass::TensorRef<ElementInput, LayoutA>(block_A.get() + int64_t(M) * K * b, LayoutA(K)),
        cutlass::ComplexTransform::kNone,
        cutlass::TensorRef<ElementInput, LayoutB>(block_B.get() + int64_t(K) * N * b, LayoutB(K)),
        cutlass::ComplexTransform::kNone,
        beta,
        cutlass::TensorRef<ElementInput, LayoutC>(block_C.get() + mn * b, LayoutC(N)),
        cutlass::TensorRef<float, LayoutC>(gold_D.get(), LayoutC(N)),
        0.0f);

      for (size_t o = 0; o < outs.size(); ++o) {
        check_rows_kernel<<<M, 256>>>(N, outs[o].D + mn * b, outs[o].S + mn * b, gold_D.get(), N, row_checks.get());
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          std::cerr << "Verification kernel failed: " << cudaGetErrorString(err) << std::endl;
          return false;
        }
        cutlass::device_memory::copy_to_host(host.data(), row_checks.get(), M);
        for (int r = 0; r < M; ++r) {
          RowCheck const &c = host[r];
          max_rd[o] = std::max(max_rd[o], double(c.max_ratio_d));
          max_rs[o] = std::max(max_rs[o], double(c.max_ratio_s));
          double const se = std::isfinite(c.row_sum) ? std::fabs(c.row_sum - 1.0) : INFINITY;
          max_sum_err[o] = std::max(max_sum_err[o], se);
          if ((c.fails || !(se <= kRowSumTol)) && ok[o]) {
            ok[o] = false;
            if (c.fails) {
              int const col = c.first_col;
              float gd = 0, d = 0, s = 0;
              cutlass::device_memory::copy_to_host(&gd, gold_D.get() + int64_t(r) * N + col, 1);
              ElementOutput hd, hs;
              cutlass::device_memory::copy_to_host(&hd, outs[o].D + mn * b + int64_t(r) * N + col, 1);
              cutlass::device_memory::copy_to_host(&hs, outs[o].S + mn * b + int64_t(r) * N + col, 1);
              d = float(hd);
              s = float(hs);
              fprintf(stderr, "%s: mismatch at batch %d, row %d, col %d: D = %g (golden %g), Softmax = %g\n",
                      outs[o].who, b, r, col, d, gd, s);
            } else {
              fprintf(stderr, "%s: Softmax row %d of batch %d sums to %.6f\n", outs[o].who, r, b, c.row_sum);
            }
          }
          nfail[o] += c.fails;
        }
      }
    }

    bool all_ok = true;
    for (size_t o = 0; o < outs.size(); ++o) {
      printf("%s: max |err|/tol D = %.3f, Softmax = %.3f; max |row sum - 1| = %.3g (tol %.3g); "
             "%lld of %lld elements out of tolerance\n",
             outs[o].who, max_rd[o], max_rs[o], max_sum_err[o], double(kRowSumTol), nfail[o], (long long)count_C);
      all_ok = all_ok && ok[o];
    }
    return all_ok;
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

#ifdef KH_TEST_SOLUTION
  /// Poisons the solution outputs (NaN), runs the solution once and verifies it.
  bool run_and_verify_solution_(char const *who) {
    cudaMemset(sol_D.get(), 0xFF, count_C * sizeof(ElementOutput));
    cudaMemset(sol_Softmax.get(), 0xFF, count_C * sizeof(ElementOutput));
    cudaError_t err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return verify({{who, sol_D.get(), sol_Softmax.get()}});
  }
#endif

  bool run() {
    initialize();

    cutlass::Status status = run_reference_();
    cudaError_t err = cudaDeviceSynchronize();
    if (status != cutlass::Status::kSuccess || err != cudaSuccess) {
      std::cerr << "Reference execution failed: " << cutlass::cutlassGetStatusString(status) << ", "
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    if (verify_reference && !verify({{"Reference", ref_D.get(), ref_Softmax.get()}})) {
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
    // The outputs must still be produced by a fresh call after the timed runs.
    if (!run_and_verify_solution_("Solution (re-run after timing)")) {
      return false;
    }
#endif
    double const flops = 2.0 * M * N * double(K) * batch;
    printf("Reference: %.1f TFLOP/s\n", flops / ref_min_ms / 1e9);
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess || props.major < 8) {
    std::cerr << "This benchmark requires a GPU with compute capability 80 or greater." << std::endl;
    return 0;
  }

  struct TestConfig {
    const char* label;
    int m, n, k, batch;
    float beta;  // alpha is 1/sqrt(K)
  };

  std::vector<TestConfig> configs;

  if (options.explicit_size) {
    configs.push_back({"custom", options.m, options.n, options.k, options.batch_count, options.beta});
  } else {
    // GEMM + softmax as run on one RTX PRO 6000 (96 GB):
    configs = {
      // Attention scores, prefill: batch = sequences x heads, M = N = tokens, K = head dim, C = additive mask/bias.
      {"attn_prefill_d64",    512,    512,   64, 1024, 1.0f},  // BERT-large training: 64 seqs x 16 heads, 512 tokens
      {"attn_prefill_d128",  4096,   4096,  128,   64, 1.0f},  // Qwen3-32B: 64 query heads, 4K-token prompt
      {"attn_prefill_d256",  8192,   8192,  256,   16, 1.0f},  // Gemma-2-9B: 16 heads x 256, 8K-token prompt
      // Attention scores, decode with GQA: M = query heads per KV head, N = context length.
      {"attn_decode_gqa",       8,   6144,  128,  128, 1.0f},  // Qwen3-32B: 16 seqs x 8 KV heads, 6K context
      // LM head logits + softmax over the vocabulary: M = tokens, K = hidden size.
      {"lm_head_softmax",     128, 262208, 5376,    1, 0.0f},  // Gemma-3-27B: 128 tokens, 262K vocabulary
      // MoE router logits + softmax over experts.
      {"moe_router",        16384,    128, 2048,    1, 0.0f},  // Qwen3-30B-A3B: 16K-token prefill chunk, 128 experts
    };
  }

  for (const auto& cfg : configs) {
    float alpha = (options.explicit_size && options.alpha_set) ? options.alpha : 1.0f / std::sqrt(float(cfg.k));

    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch);
    fflush(stdout);

    if (cfg.n % 8 || cfg.k % 8) {
      std::cerr << "N and K must be multiples of 8" << std::endl;
      return -1;
    }

    Testbed testbed(cfg.m, cfg.n, cfg.k, cfg.batch, alpha, cfg.beta, options.verification_enabled,
                    options.iterations, options.seed);
    if (!testbed.run()) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
