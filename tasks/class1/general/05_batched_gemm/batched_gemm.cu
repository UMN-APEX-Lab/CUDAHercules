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
    \brief Strided batched SGEMM (FP32, column-major) benchmark harness.

    C_b = alpha * A_b * B_b + beta * C_b for b in [0, batch_count), all matrices FP32 column-major,
    B batches interleaved along its leading dimension (ldb = K * batch_count, batch stride K).

    The reference is CUTLASS GemmBatched (SIMT FP32) with a per-shape threadblock configuration;
    the candidate (solution.h, -DKH_TEST_SOLUTION) implements StridedBatchedSgemm(). Both are verified
    on the device against a golden result with FP64 accumulation.

    Usage:
      ./test                                  # all problem sizes
      ./test M N K batch [alpha beta iterations]
      ./test ... --skip-verification          # do not verify the reference
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
#include <sstream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"

#pragma warning( disable : 4503)

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Reference and golden-reference headers (included after the solution on purpose)
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/util/reference/device/gemm_complex.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS GemmBatched, FP32 SIMT, per-shape threadblock configuration
// (offline sweep on RTX PRO 6000 Blackwell)

template <int TbM, int TbN, int WarpM, int WarpN, int Stages>
cudaError_t kh_ref_sgemm_config(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float *C, int ldc, long long int batch_stride_C,
  float beta, int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM, TbN, 8>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    Stages,
    1, 1>;

  Gemm gemm_op;
  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, batch_stride_A,
    {B, ldb}, batch_stride_B,
    {C, ldc}, batch_stride_C,
    {C, ldc}, batch_stride_C,
    {alpha, beta},
    batch_count
  });
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t kh_ref_strided_batched_sgemm(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float *C, int ldc, long long int batch_stride_C,
  float beta, int batch_count) {
  // Skinny M (GQA decode): 128x64 threadblocks (more CTAs along N); otherwise CUTLASS's default
  // FP32 SIMT configuration (128x128x8, 32x64 warps, 2 stages), fastest at the other shapes.
  if (m <= 64) {
    return kh_ref_sgemm_config<128, 64, 64, 32, 4>(m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B,
                                                  C, ldc, batch_stride_C, beta, batch_count);
  }
  return kh_ref_sgemm_config<128, 128, 32, 64, 2>(m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B,
                                                 C, ldc, batch_stride_C, beta, batch_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Golden reference for strided batched GEMM NN (column-major), computed on the device with
/// FP64 accumulation (CUTLASS device reference GEMM). Reads C before it is overwritten.
cudaError_t kh_golden_strided_batched_gemm(
  int m, int n, int k, float alpha,
  float const *A, int lda, long long int batch_stride_A,
  float const *B, int ldb, long long int batch_stride_B,
  float const *C, int ldc, long long int batch_stride_C,
  float beta, float *D, int batch_count) {

  using Layout = cutlass::layout::ColumnMajor;
  cutlass::reference::device::GemmComplex<
    float, Layout, float, Layout, float, Layout, double, double
  >(
    {m, n, k},
    double(alpha),
    cutlass::TensorRef<float, Layout>(const_cast<float *>(A), Layout(lda)),
    cutlass::ComplexTransform::kNone,
    cutlass::TensorRef<float, Layout>(const_cast<float *>(B), Layout(ldb)),
    cutlass::ComplexTransform::kNone,
    double(beta),
    cutlass::TensorRef<float, Layout>(const_cast<float *>(C), Layout(ldc)),
    cutlass::TensorRef<float, Layout>(D, Layout(ldc)),
    double(0),
    batch_count, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_C);
  return cudaGetLastError();
}

/// Deterministic fill with uniform values in [-1, 1) carrying 24 random bits each, so that the
/// data needs the full FP32 mantissa (lower-precision arithmetic cannot reproduce the result).
__global__ void FillRandomUniform_kernel(float *ptr, size_t count, uint32_t seed) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9E3779B9u) ^ (seed * 0x85EBCA6Bu);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    ptr[i] = float(int32_t(h >> 8) - (1 << 23)) * (1.0f / float(1 << 23));
  }
}

cudaError_t FillRandomUniform(float *ptr, size_t count, uint32_t seed) {
  FillRandomUniform_kernel<<<1024, 256>>>(ptr, count, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side element-wise check against the golden result.
//
// With inputs in [-1, 1), FP32 accumulation errors stay below kTolScale * (|alpha| * sqrt(K) + |beta|)
// (sequential FP32 accumulation over K = 8192 reaches 0.31 of it); TF32/FP16 operands exceed it by
// 16-19x, BF16 operands by > 100x. Non-finite values always fail.

static constexpr double kTolScale = 3e-5;

struct CheckStats {
  unsigned long long fails;
  unsigned long long first_fail;   // linear index of the first failing element
  unsigned int max_err_bits;       // float bits of max |err| (non-negative floats order like ints)
  double err_sq;
  double gold_sq;
};

__global__ void check_kernel(float const *x, float const *gold, size_t count, float tol, CheckStats *stats) {
  __shared__ double red_e[32], red_g[32];
  __shared__ float red_m[32];
  double es = 0, gs = 0;
  float mx = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    float const v = x[i], g = gold[i];
    float err = fabsf(v - g);
    if (!isfinite(v) || !(err <= tol)) {
      atomicAdd(&stats->fails, 1ull);
      atomicMin(&stats->first_fail, (unsigned long long)i);
      if (!isfinite(v)) err = INFINITY;
    }
    mx = fmaxf(mx, err);
    es += double(err) * double(err);
    gs += double(g) * double(g);
  }
  for (int o = 16; o > 0; o >>= 1) {
    es += __shfl_xor_sync(0xffffffffu, es, o);
    gs += __shfl_xor_sync(0xffffffffu, gs, o);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { red_e[warp] = es; red_g[warp] = gs; red_m[warp] = mx; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < int(blockDim.x >> 5); ++w) { es += red_e[w]; gs += red_g[w]; mx = fmaxf(mx, red_m[w]); }
    atomicAdd(&stats->err_sq, es);
    atomicAdd(&stats->gold_sq, gs);
    atomicMax(&stats->max_err_bits, __float_as_uint(mx));
  }
}

bool check_output(char const *who, float const *d_x, float const *d_gold,
                  int m, int n, int k, long long int batch_stride_C, int batch_count,
                  float alpha, float beta) {
  double const tol = kTolScale * (std::fabs(double(alpha)) * std::sqrt(double(k)) + std::fabs(double(beta)));
  size_t const count = size_t(batch_stride_C) * batch_count;  // packed: ldc = m

  CheckStats init{0ull, ~0ull, 0u, 0.0, 0.0}, stats;
  CheckStats *d_stats = nullptr;
  cudaMalloc(&d_stats, sizeof(CheckStats));
  cudaMemcpy(d_stats, &init, sizeof(CheckStats), cudaMemcpyHostToDevice);
  check_kernel<<<2048, 256>>>(d_x, d_gold, count, float(tol), d_stats);
  cudaError_t err = cudaMemcpy(&stats, d_stats, sizeof(CheckStats), cudaMemcpyDeviceToHost);
  cudaFree(d_stats);
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: verification kernel failed: %s\n", who, cudaGetErrorString(err));
    return false;
  }

  float max_err;
  unsigned int bits = stats.max_err_bits;
  std::memcpy(&max_err, &bits, sizeof(float));
  double const rel_rms = std::sqrt(stats.err_sq / std::max(stats.gold_sq, 1e-300));
  if (stats.fails) {
    size_t const i = stats.first_fail;
    float got, expected;
    cudaMemcpy(&got, d_x + i, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&expected, d_gold + i, sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%s: mismatch at batch %lld, row %lld, col %lld: got %.8g, expected %.8g (tol %.3g)\n",
            who, (long long)(i / batch_stride_C), (long long)(i % m), (long long)((i % batch_stride_C) / m),
            got, expected, tol);
  }
  fprintf(stdout, "%s: max |err| = %.3g (tol %.3g, max |err|/tol = %.3f), rms(err)/rms(ref) = %.3g, "
          "%llu of %zu elements out of tolerance\n",
          who, max_err, tol, max_err / tol, rel_rms, stats.fails, count);
  return stats.fails == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fn>
float time_min_ms(Fn &&fn, int iterations, char const *label) {
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

/// Verifies the reference (unless skipped) and the solution (if KH_TEST_SOLUTION) against the
/// golden result, then profiles both. Returns false on any failure.
bool RunSize(int m, int n, int k, int batch_count, float alpha, float beta, int iterations,
             bool skip_reference_verification) {

  // Column-major layout matching the original example
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  size_t const count_A = size_t(batch_count) * lda * k;
  size_t const count_B = size_t(ldb) * n;
  size_t const count_C = size_t(batch_count) * ldc * n;

  long long int const batch_stride_A = (long long int)(lda) * (long long int)(k);
  long long int const batch_stride_B = (long long int)(k);
  long long int const batch_stride_C = (long long int)(ldc) * (long long int)(n);

  size_t const sizeof_C = count_C * sizeof(float);
  uint32_t const seed_C = 3;

#ifdef KH_TEST_SOLUTION
  bool const need_golden = true;
#else
  bool const need_golden = !skip_reference_verification;
#endif

  float *d_A = nullptr, *d_B = nullptr, *d_C_ref = nullptr, *d_golden = nullptr, *d_C_sol = nullptr;
  auto free_all = [&]() {
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_golden); cudaFree(d_C_sol);
  };

  cudaError_t result;
  if ((result = cudaMalloc(&d_A, count_A * sizeof(float))) != cudaSuccess ||
      (result = cudaMalloc(&d_B, count_B * sizeof(float))) != cudaSuccess ||
      (result = cudaMalloc(&d_C_ref, sizeof_C)) != cudaSuccess ||
      (need_golden && (result = cudaMalloc(&d_golden, sizeof_C)) != cudaSuccess)
#ifdef KH_TEST_SOLUTION
      || (result = cudaMalloc(&d_C_sol, sizeof_C)) != cudaSuccess
#endif
      ) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(result) << std::endl;
    free_all();
    return false;
  }

  // Random inputs; C holds random values as well (it is an input when beta != 0), so unwritten
  // outputs are detected.
  FillRandomUniform(d_A, count_A, 1);
  FillRandomUniform(d_B, count_B, 2);
  FillRandomUniform(d_C_ref, count_C, seed_C);

  // Golden result (reads the initial C)
  if (need_golden) {
    result = kh_golden_strided_batched_gemm(
      m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
      d_C_ref, ldc, batch_stride_C, beta, d_golden, batch_count);
    if (result == cudaSuccess) result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Golden reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
      free_all();
      return false;
    }
  }

  // Reference
  result = kh_ref_strided_batched_sgemm(
    m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
    d_C_ref, ldc, batch_stride_C, beta, batch_count);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Reference batched GEMM failed: " << cudaGetErrorString(result) << std::endl;
    free_all();
    return false;
  }
  if (!skip_reference_verification) {
    if (!check_output("Reference", d_C_ref, d_golden, m, n, k, batch_stride_C, batch_count, alpha, beta)) {
      free_all();
      return false;
    }
  } else {
    std::cout << "Baseline reference verification skipped." << std::endl;
  }

#ifdef KH_TEST_SOLUTION
  // Runs the solution once on a freshly initialized C and verifies it.
  auto run_and_verify_solution = [&](char const *who) -> bool {
    FillRandomUniform(d_C_sol, count_C, seed_C);
    cudaError_t err = StridedBatchedSgemm(
      m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
      d_C_sol, ldc, batch_stride_C, beta, batch_count);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output(who, d_C_sol, d_golden, m, n, k, batch_stride_C, batch_count, alpha, beta);
  };
  if (!run_and_verify_solution("Solution")) {
    std::cerr << "Solution incorrect." << std::endl;
    free_all();
    return false;
  }
#endif

  std::cout << "Passed" << std::endl;

  if (iterations > 0) {
    float const ref_min_ms = time_min_ms([&]() {
      kh_ref_strided_batched_sgemm(m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
                                   d_C_ref, ldc, batch_stride_C, beta, batch_count);
    }, iterations, "Ref time");
#ifdef KH_TEST_SOLUTION
    float const sol_min_ms = time_min_ms([&]() {
      StridedBatchedSgemm(m, n, k, alpha, d_A, lda, batch_stride_A, d_B, ldb, batch_stride_B,
                          d_C_sol, ldc, batch_stride_C, beta, batch_count);
    }, iterations, "Kernel time");
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
    // The output must still be produced by a fresh call after the timed runs.
    if (!run_and_verify_solution("Solution (re-run after timing)")) {
      free_all();
      return false;
    }
#endif
    printf("Reference: %.1f TFLOP/s\n", 2.0 * m * n * double(k) * batch_count / ref_min_ms / 1e9);
  }

  free_all();
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k, batch_count;
  };

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;
  bool skip_reference_verification = false;
  std::vector<const char*> positional;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(arg[i], "--skip-verification") || !strcmp(arg[i], "--no_verif")) {
      skip_reference_verification = true;
    } else {
      positional.push_back(arg[i]);
    }
  }

  bool explicit_size = (positional.size() >= 4);

  if (explicit_size) {
    int M = 128, N = 128, K = 64, B = 4;
    std::stringstream(positional[0]) >> M;
    std::stringstream(positional[1]) >> N;
    std::stringstream(positional[2]) >> K;
    std::stringstream(positional[3]) >> B;
    if (positional.size() > 4) std::stringstream(positional[4]) >> alpha;
    if (positional.size() > 5) std::stringstream(positional[5]) >> beta;
    if (positional.size() > 6) std::stringstream(positional[6]) >> iterations;
    configs.push_back({"custom", M, N, K, B});
  } else {
    // FP32 batched GEMMs as run on one RTX PRO 6000 (96 GB); column-major, batch = heads or experts:
    //   attn_scores_prefill:    S = Q K^T per head, Qwen3-32B (64 query heads, head_dim 128),
    //                           4K-token query chunk against 8K keys
    //   attn_pv_prefill:        O = P V per head, same shapes
    //   attn_scores_decode_gqa: Qwen3-32B decode, 16 seqs x 8 KV heads, 8 query heads per KV head,
    //                           6K-token context
    //   moe_experts:            Qwen3-30B-A3B, 128 experts x 1024 tokens (equal capacity),
    //                           2048 -> 1536 (gate + up projection, 2 x 768)
    configs = {
      {"attn_scores_prefill",    4096, 8192,  128,  64},
      {"attn_pv_prefill",        4096,  128, 8192,  64},
      {"attn_scores_decode_gqa",    8, 6144,  128, 128},
      {"moe_experts",            1024, 1536, 2048, 128},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d, batch=%d ===\n",
            cfg.label, cfg.m, cfg.n, cfg.k, cfg.batch_count);
    fflush(stdout);

    if (!RunSize(cfg.m, cfg.n, cfg.k, cfg.batch_count, alpha, beta, iterations, skip_reference_verification)) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
