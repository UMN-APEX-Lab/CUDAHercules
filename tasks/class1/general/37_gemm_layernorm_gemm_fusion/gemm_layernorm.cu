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
    result: FP32-accumulated GEMMs and LayerNorm with FP64 statistics on the device. Two input sets
    are used (plain random, and realistic LayerNorm inputs with outlier / near-constant columns).

    Usage:
      ./test                                          # all problem sizes
      ./test --hidden_dim=1024 --valid_word_num=4096  # custom size (hidden_dim multiple of 128, valid_word_num multiple of 8)
      ./test --iterations=50                          # profiling iterations (0 disables)
*/

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include <cstdio>   // std::fprintf (KH_TRIAL_TIMES)
#include <cstdlib>  // std::getenv / std::atoi (KH_WARMUP / KH_ITERS overrides)

/// Optional environment overrides for the timing loops (reference and solution):
///   KH_WARMUP = warmup launches before timing, KH_ITERS = number of timed trials.
/// Unset (or empty) -> the built-in defaults are used unchanged.
static int kh_env_int(const char *name, int dflt) {
  const char *v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : dflt;
}

/// KH_TRIAL_TIMES=1: additionally log every timed trial to stderr as
/// "KH_TRIAL <label> <trial> <ms>" (stdout output is unchanged).
static void kh_trial_log(const char *label, int trial, float ms) {
  if (kh_env_int("KH_TRIAL_TIMES", 0)) std::fprintf(stderr, "KH_TRIAL %s %d %.6f\n", label, trial, ms);
}

// ===================================== KH_TIMING_BEGIN ======================================
// Timing protocol of the Class 1 harnesses (this block is identical in every harness).
//
// kh_bench(label, warmup, iters, prep, call, check) measures one implementation; the reference
// and the solution are measured the same way:
//  * prep(s) runs before every warmup and timed call, outside the timed region. It copies input
//    set s into the fixed input buffers (the call always sees the same pointers) and re-poisons
//    the outputs. s alternates 0, 1, 0, 1, ... over all calls and the two sets hold different
//    data, so a call can never reuse the result of an earlier call.
//  * Before every timed call the L2 is flushed by READING a max(256 MB, 2 x L2) buffer (L2-cached
//    loads, no stores), then cudaDeviceSynchronize(). This also writes back the dirty lines left by
//    prep, and the L2 then holds only clean, unrelated lines, so the timed call pays no write-back
//    of flush data (a write-based flush would leave up to a full L2 of dirty lines behind).
//  * Every trial is timed with CUDA events on the legacy default stream and with the host clock,
//    from just before the start event to after the cudaDeviceSynchronize() that follows the stop
//    event. Trial time = max(event time, wall time - fixed overhead), so work the call leaves
//    running on other streams (private non-blocking streams, graphs, host callbacks) when it
//    returns is counted in full. The fixed overhead (clock reads, event records, synchronization
//    wake-up) is calibrated once per process as the 90th percentile of (wall - event) over 64
//    empty-kernel calls measured the same way, clamped to [2, 50] us.
//  * check(s) runs after every timed call, outside the timed region: a device-side comparison of
//    the outputs with the golden result of input set s. Any failure (or CUDA error) prints
//    "Incorrect" and exits with status 1.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void kh_empty_kernel() {}

__global__ void kh_l2_flush_kernel(int4 const *p, size_t n, int *sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    int4 const v = __ldcg(p + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x13579bdf) *sink = 1;  // keeps the loads alive
}

/// Aborts the run as incorrect.
[[noreturn]] static void kh_fail(char const *what) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s\n", what);
  std::cout << "Incorrect" << std::endl;
  std::exit(1);
}

[[maybe_unused]] static void kh_cuda_ok(cudaError_t e, char const *what) {
  if (e != cudaSuccess) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "%s: CUDA error: %s", what, cudaGetErrorString(e));
    kh_fail(msg);
  }
}

/// Asynchronous device-to-device copy on the legacy default stream (input-set restore).
[[maybe_unused]] static void kh_copy(void *dst, void const *src, size_t bytes) {
  kh_cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice), "input restore");
}

static double kh_host_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct KhTimer {
  cudaEvent_t start = nullptr, stop = nullptr;
  int4 *flush = nullptr;
  size_t flush_n = 0;
  int *sink = nullptr;
  double overhead_ms = 0.0;
};

static void kh_l2_flush(KhTimer &t) {
  kh_l2_flush_kernel<<<1024, 256>>>(t.flush, t.flush_n, t.sink);
  kh_cuda_ok(cudaDeviceSynchronize(), "L2 flush");
}

/// One measured call: L2 flush, then event and host-clock timing up to device-wide completion.
template <typename Call>
static void kh_timed_call(KhTimer &t, Call &&call, char const *label, float *event_ms,
                          double *wall_ms) {
  kh_l2_flush(t);
  double const t0 = kh_host_ms();
  cudaEventRecord(t.start);
  call();
  cudaEventRecord(t.stop);
  cudaError_t const e1 = cudaEventSynchronize(t.stop);
  cudaError_t const e2 = cudaDeviceSynchronize();
  double const t1 = kh_host_ms();
  kh_cuda_ok(e1 != cudaSuccess ? e1 : e2, label);
  kh_cuda_ok(cudaEventElapsedTime(event_ms, t.start, t.stop), label);
  *wall_ms = t1 - t0;
}

static KhTimer &kh_timer() {
  static KhTimer t;
  static bool ready = false;
  if (ready) return t;
  int dev = 0, l2 = 0;
  kh_cuda_ok(cudaGetDevice(&dev), "timer init");
  kh_cuda_ok(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev), "timer init");
  size_t const bytes = std::max<size_t>(size_t(256) << 20, 2 * size_t(l2));
  kh_cuda_ok(cudaMalloc(&t.flush, bytes), "L2 flush buffer");
  kh_cuda_ok(cudaMemset(t.flush, 0x5a, bytes), "L2 flush buffer");
  kh_cuda_ok(cudaMalloc(&t.sink, sizeof(int)), "L2 flush buffer");
  t.flush_n = bytes / sizeof(int4);
  kh_cuda_ok(cudaEventCreate(&t.start), "timer init");
  kh_cuda_ok(cudaEventCreate(&t.stop), "timer init");
  std::vector<double> d;
  for (int i = 0; i < 74; ++i) {
    float ev = 0.f;
    double wall = 0.0;
    kh_timed_call(t, [] { kh_empty_kernel<<<1, 32>>>(); }, "timer calibration", &ev, &wall);
    if (i >= 10) d.push_back(wall - double(ev));
  }
  std::sort(d.begin(), d.end());
  t.overhead_ms = std::min(0.05, std::max(0.002, d[d.size() * 9 / 10]));
  std::fprintf(stderr, "KH timing: L2 flush %zu MB, fixed overhead %.4f ms (p90 of %zu empty-kernel "
               "trials, median %.4f ms)\n", bytes >> 20, t.overhead_ms, d.size(), d[d.size() / 2]);
  ready = true;
  return t;
}

struct KhTiming {
  float avg_ms, min_ms;
};

/// check() for implementations whose output is not verified during timing (the reference).
[[maybe_unused]] static bool kh_no_check(int) { return true; }

template <typename Prep, typename Call, typename Check>
static KhTiming kh_bench(char const *label, int warmup, int iters, Prep &&prep, Call &&call,
                         Check &&check) {
  KhTimer &t = kh_timer();
  int set = 0;
  for (int i = 0; i < warmup; ++i, set ^= 1) {
    prep(set);
    kh_cuda_ok(cudaDeviceSynchronize(), "input restore");
    call();
    kh_cuda_ok(cudaDeviceSynchronize(), label);
  }
  double total = 0.0;
  float mn = 1e30f;
  for (int it = 0; it < iters; ++it, set ^= 1) {
    prep(set);
    float ev = 0.f;
    double wall = 0.0;
    kh_timed_call(t, call, label, &ev, &wall);
    float const ms = float(std::max<double>(ev, wall - t.overhead_ms));
    kh_trial_log(label, it, ms);
    if (!check(set)) {
      char msg[256];
      std::snprintf(msg, sizeof(msg), "%s: output of timed call %d (input set %d) does not match "
                    "the golden result.", label, it, set);
      kh_fail(msg);
    }
    total += ms;
    mn = std::min(mn, ms);
  }
  return {iters > 0 ? float(total / iters) : 0.f, mn};
}
// ====================================== KH_TIMING_END =======================================

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
// Device-side golden helpers and the realistic input set.

/// LayerNorm over the hidden dimension of every column of C0 (column n = token n), in place,
/// two-pass with FP64 statistics (eps 1e-6). One block per column.
__global__ void kh_golden_layernorm_kernel(float *C0, int H, int T, Element const *gamma,
                                           Element const *beta_ln) {
  __shared__ double red[32];
  __shared__ double bcast;
  int const n = blockIdx.x;
  if (n >= T) return;
  float *col = C0 + int64_t(n) * H;
  auto block_sum = [&](double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
      double t = 0;
      for (int w = 0; w < int(blockDim.x >> 5); ++w) t += red[w];
      bcast = t;
    }
    __syncthreads();
    double const r = bcast;
    __syncthreads();
    return r;
  };
  double s = 0;
  for (int m = threadIdx.x; m < H; m += blockDim.x) s += double(col[m]);
  double const mean = block_sum(s) / H;
  double sq = 0;
  for (int m = threadIdx.x; m < H; m += blockDim.x) {
    double const d = double(col[m]) - mean;
    sq += d * d;
  }
  double const inv_std = 1.0 / sqrt(block_sum(sq) / H + 1e-6);
  for (int m = threadIdx.x; m < H; m += blockDim.x)
    col[m] = float((double(col[m]) - mean) * inv_std * double(float(gamma[m])) + double(float(beta_ln[m])));
}

/// Sum of squares of a float buffer (FP64 accumulation).
__global__ void kh_sumsq_f32_kernel(float const *g, size_t n, double *out) {
  double s = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    s += double(g[i]) * double(g[i]);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

struct KhC1Stats {
  double err_sq;
  unsigned long long bad, first_bad;
  unsigned int max_ratio_bits;   // float bits of max |err|/tol (non-negative floats order like ints)
};

/// Element-wise C1 check: |x - g| <= atol + rtol * |g| (NaN/Inf fail); accumulates sum err^2.
__global__ void kh_c1_compare_kernel(Element const *x, float const *g, size_t n, double atol,
                                     double rtol, KhC1Stats *st) {
  double es = 0;
  float mr = 0.f;
  unsigned long long bad = 0, first = ~0ull;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    double const v = double(float(x[i]));
    double const gv = double(g[i]);
    double err = fabs(v - gv);
    double const tol = atol + rtol * fabs(gv);
    if (!isfinite(v) || !(err <= tol)) {
      ++bad;
      if (i < first) first = i;
      if (!isfinite(v)) err = INFINITY;
    }
    float const r = float(err / tol);
    mr = fmaxf(mr, (r >= 0.f) ? r : INFINITY);
    es += err * err;
  }
  for (int o = 16; o > 0; o >>= 1) {
    es += __shfl_xor_sync(0xffffffffu, es, o);
    mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, o));
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
    unsigned long long const f = __shfl_xor_sync(0xffffffffu, first, o);
    first = f < first ? f : first;
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&st->err_sq, es);
    atomicMax(&st->max_ratio_bits, __float_as_uint(mr));
    if (bad) {
      atomicAdd(&st->bad, bad);
      atomicMin(&st->first_bad, first);
    }
  }
}

__host__ __device__ inline unsigned kh_ln_hash(unsigned x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

/// Uniform in [-1, 1) with 24 random bits (a full FP16 significand for |v| >= 2^-13).
__device__ inline float kh_ln_uniform(size_t i, unsigned seed) {
  unsigned const h = kh_ln_hash(unsigned(i) ^ kh_ln_hash(unsigned(i >> 32) + seed * 0x9E3779B9u));
  return float(int(h >> 7) - (1 << 24)) * (1.0f / 16777216.0f);
}

/// Token classes of the realistic input set (set 1):
///   4: massive activation (the first token of every 512-token block, i.e. a BOS-like token, plus
///      ~0.4% of the others): two fixed hidden dimensions reach ~H, 100-200x the typical |C0|, and
///      dominate the variance of the column;
///   1: mean-offset column (~0.8%): |mean| / std of the LayerNorm column ~1.5-1.7;
///   2: near-constant column (~0.4%): variance ~1e-8, far below eps = 1e-6;
///   3: all-zero column (~0.4%, padding-like): variance exactly 0;
///   0: ordinary token.
__host__ __device__ inline int kh_ln_token_class(int n) {
  if (n % 512 == 0) return 4;
  unsigned const h = kh_ln_hash(unsigned(n) * 0x2545F491u + 0x9E3779B9u) % 256u;
  return h < 2 ? 1 : (h == 2 ? 2 : (h == 3 ? 3 : (h == 4 ? 4 : 0)));
}

/// The two hidden dimensions carrying the massive activations (in different 64-row tiles).
__host__ __device__ inline bool kh_ln_massive_dim(int m, int H) {
  return m == (H * 5) / 16 + 7 || m == (H * 11) / 16 + 3;
}

/// Shared direction s[k] in {-1, +1} of the realistic set (mean offsets), and an independent
/// direction s2[k] (massive activations).
__device__ inline float kh_ln_sign(int k) { return (kh_ln_hash(unsigned(k) + 0x51ED270Bu) & 1u) ? 1.f : -1.f; }
__device__ inline float kh_ln_sign2(int k) { return (kh_ln_hash(unsigned(k) * 0x2C1B3C6Du + 0x297A2D39u) & 1u) ? 1.f : -1.f; }

/// Realistic A0 (H x H, row-major): every row carries the shared direction s,
/// A0[m, k] = (1 - eps) * U + eps * s[k], so that a token aligned with s gets a common (mean)
/// component in all H outputs; the rows of the two massive dimensions are s2.
__global__ void kh_fill_realistic_A0(Element *A0, int H, unsigned seed, float eps) {
  size_t const n = size_t(H) * H;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int const k = int(i % H), m = int(i / H);
    A0[i] = Element(kh_ln_massive_dim(m, H) ? kh_ln_sign2(k)
                                           : (1.f - eps) * kh_ln_uniform(i, seed) + eps * kh_ln_sign(k));
  }
}

/// Realistic B0 (H x T, column-major, column n = token n), by token class:
///   0: U;  1: a * sign_n * s + (1 - a) * U with a in [0.5, 0.75);  2: tiny * U;  3: 0;
///   4: sign_n * s2 (C0 = sign_n * H in the two massive dimensions).
__global__ void kh_fill_realistic_B0(Element *B0, int H, int T, unsigned seed, float tiny) {
  size_t const n_el = size_t(H) * T;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n_el; i += size_t(gridDim.x) * blockDim.x) {
    int const k = int(i % H), n = int(i / H);
    float const u = kh_ln_uniform(i, seed);
    float v = u;
    switch (kh_ln_token_class(n)) {
      case 1: {
        unsigned const h = kh_ln_hash(unsigned(n) ^ 0x68E31DA4u);
        float const a = 0.5f + 0.25f * float(h >> 8) * (1.0f / 16777216.0f);
        float const sgn = (h & 1u) ? 1.f : -1.f;
        v = a * sgn * kh_ln_sign(k) + (1.f - a) * u;
        break;
      }
      case 2: v = tiny * u; break;
      case 3: v = 0.f; break;
      case 4: v = ((kh_ln_hash(unsigned(n) ^ 0x68E31DA4u) & 1u) ? 1.f : -1.f) * kh_ln_sign2(k); break;
      default: break;
    }
    B0[i] = Element(v);
  }
}

/// |mean| / std of the LayerNorm columns of class-1 tokens at a = 0.5 (a in [0.5, 0.75) gives
/// 1.0x - 1.13x of it). Bounded by the accuracy of the reference (CUTLASS GemmLayernorm computes
/// the variance as E[x^2] - E[x]^2 from FP16 partial statistics): at the benchmark sizes its max
/// |err|/tol grows from ~0.2 (plain data) to ~0.75 at ratios 2.2-2.6 and exceeds 1 beyond ~4.
static constexpr float kOutlierMeanToStd = 1.5f;

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

  // Buffers passed to the reference / solution.
  cutlass::DeviceAllocation<Element> block_A0, block_B0, block_C0, block_A1, block_C1;
  cutlass::DeviceAllocation<Element> block_gamma, block_beta, block_var, block_mean;
  cutlass::DeviceAllocation<Element> block_shift;  // optional shift for the variance (zeros)
  // Two input sets (0: plain random, 1: realistic LayerNorm inputs), restored into the buffers
  // above before every call, and their golden results.
  cutlass::DeviceAllocation<Element> set_A0[2], set_B0[2], set_A1[2], set_gamma[2], set_beta[2];
  cutlass::DeviceAllocation<float> golden_C1[2];
  double golden_rms[2] = {0.0, 0.0};
  cutlass::DeviceAllocation<KhC1Stats> check_stats;
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
    check_stats.reset(1);
    for (int s = 0; s < 2; ++s) {
      set_A0[s].reset(count_A0);
      set_B0[s].reset(count_B0);
      set_A1[s].reset(count_A1);
      set_gamma[s].reset(H);
      set_beta[s].reset(H);
    }

    // Set 0: random values at full FP16 precision: activations and weights uniform in [-1, 1),
    // LayerNorm scale uniform in [0.5, 1.5) and shift uniform in [-0.5, 0.5).
    cutlass::reference::device::BlockFillRandomUniform(set_A0[0].get(), count_A0, seed, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_B0[0].get(), count_B0, seed + 1, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_A1[0].get(), count_A1, seed + 2, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_gamma[0].get(), H, seed + 3, Element(1.5f), Element(0.5f), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_beta[0].get(), H, seed + 4, Element(0.5f), Element(-0.5f), -1);

    // Set 1: realistic LayerNorm inputs (see kh_ln_token_class): massive activations, columns with
    // |mean| / std ~ kOutlierMeanToStd, near-constant and all-zero columns, ordinary tokens.
    // C0 = A0 B0 has std ~ sqrt(H) / 3 per column for ordinary tokens; eps sets the outlier ratio
    // (~1.5 eps sqrt(H)) and `tiny` gives near-constant columns a std of ~1e-4.
    float const eps = kOutlierMeanToStd / (1.5f * std::sqrt(float(H)));
    float const tiny = 3e-4f / std::sqrt(float(H));
    kh_fill_realistic_A0<<<1024, 256>>>(set_A0[1].get(), H, seed + 10, eps);
    kh_fill_realistic_B0<<<1024, 256>>>(set_B0[1].get(), H, T, seed + 11, tiny);
    cutlass::reference::device::BlockFillRandomUniform(set_A1[1].get(), count_A1, seed + 12, Element(1), Element(-1), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_gamma[1].get(), H, seed + 13, Element(1.5f), Element(0.5f), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_beta[1].get(), H, seed + 14, Element(0.5f), Element(-0.5f), -1);

#ifdef KH_TEST_SOLUTION
    sol_C1.reset(count_C1);
#endif
  }

  /// Copies input set s into the buffers passed to the reference / solution.
  void restore_inputs(int s) {
    kh_copy(block_A0.get(), set_A0[s].get(), count_A0 * sizeof(Element));
    kh_copy(block_B0.get(), set_B0[s].get(), count_B0 * sizeof(Element));
    kh_copy(block_A1.get(), set_A1[s].get(), count_A1 * sizeof(Element));
    kh_copy(block_gamma.get(), set_gamma[s].get(), H * sizeof(Element));
    kh_copy(block_beta.get(), set_beta[s].get(), H * sizeof(Element));
    kh_cuda_ok(cudaMemsetAsync(block_shift.get(), 0, T * sizeof(Element)), "input restore");
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

  /// Poisons the solution output (NaN), runs the solution once on input set s (already restored)
  /// and verifies it.
  bool run_and_verify_solution_(char const *who, int s) {
    cudaMemset(sol_C1.get(), 0xFF, count_C1 * sizeof(Element));
    cudaDeviceSynchronize();  // inputs and poison complete before the call (it may use other streams)
    cudaError_t err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output(who, sol_C1, s);
  }
#endif

  /// Golden result of input set s: C0 = alpha A0 B0 (FP32 accumulation, not rounded), LayerNorm
  /// over the hidden dimension with FP64 statistics, C1 = alpha A1 Norm (FP32 accumulation).
  bool compute_golden(int s) {
    cutlass::DeviceAllocation<float> golden_C0(count_C0);
    golden_C1[s].reset(count_C1);
    cudaMemset(golden_C0.get(), 0, count_C0 * sizeof(float));
    cudaMemset(golden_C1[s].get(), 0, count_C1 * sizeof(float));

    cutlass::reference::device::Gemm<Element, RowMajor, Element, ColumnMajor, float, ColumnMajor, float, float> gemm0;
    cutlass::reference::device::Gemm<Element, ColumnMajor, float, ColumnMajor, float, ColumnMajor, float, float> gemm1;

    cutlass::TensorRef<float, ColumnMajor> ref_C0(golden_C0.get(), ColumnMajor(H));
    cutlass::TensorRef<float, ColumnMajor> ref_C1(golden_C1[s].get(), ColumnMajor(N1));

    gemm0({H, T, H}, alpha,
          cutlass::TensorRef<Element, RowMajor>(set_A0[s].get(), RowMajor(H)),
          cutlass::TensorRef<Element, ColumnMajor>(set_B0[s].get(), ColumnMajor(H)),
          0.0f, ref_C0, ref_C0);

    // LayerNorm over the hidden dimension (column n of C0 = token n), two-pass, FP64 statistics
    kh_golden_layernorm_kernel<<<T, 256>>>(golden_C0.get(), H, T, set_gamma[s].get(), set_beta[s].get());

    gemm1({N1, T, H}, alpha,
          cutlass::TensorRef<Element, ColumnMajor>(set_A1[s].get(), ColumnMajor(N1)),
          cutlass::TensorRef<float, ColumnMajor>(golden_C0.get(), ColumnMajor(H)),
          0.0f, ref_C1, ref_C1);

    cutlass::DeviceAllocation<double> sumsq(1);
    cudaMemset(sumsq.get(), 0, sizeof(double));
    kh_sumsq_f32_kernel<<<1024, 256>>>(golden_C1[s].get(), size_t(count_C1), sumsq.get());
    double h_sumsq = 0;
    cudaError_t err = cudaMemcpy(&h_sumsq, sumsq.get(), sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Golden reference failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    golden_rms[s] = std::sqrt(h_sumsq / double(count_C1));
    return true;
  }

  /// Checks every element of C1 and the aggregate error against the golden result of input set s
  /// (device-side). verbose = false prints only on failure.
  bool check_output(char const *who, cutlass::DeviceAllocation<Element> const &out, int s,
                    bool verbose = true) {
    double const rms = golden_rms[s];
    double const atol = kAtol * rms;
    KhC1Stats init{0.0, 0ull, ~0ull, 0u}, st;
    cudaMemcpy(check_stats.get(), &init, sizeof(init), cudaMemcpyHostToDevice);
    kh_c1_compare_kernel<<<1024, 256>>>(out.get(), golden_C1[s].get(), size_t(count_C1), atol, kRtol,
                                        check_stats.get());
    cudaError_t err = cudaMemcpy(&st, check_stats.get(), sizeof(st), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "%s: verification kernel failed: %s\n", who, cudaGetErrorString(err));
      return false;
    }
    float max_ratio;
    std::memcpy(&max_ratio, &st.max_ratio_bits, sizeof(float));
    double const rel_rms_err = std::sqrt(st.err_sq / double(count_C1)) / rms;
    if (st.bad) {
      int64_t const i = int64_t(st.first_bad);
      Element hx;
      float hg = 0;
      cutlass::device_memory::copy_to_host(&hx, out.get() + i, 1);
      cutlass::device_memory::copy_to_host(&hg, golden_C1[s].get() + i, 1);
      double const g = double(hg);
      fprintf(stderr, "%s: C1 mismatch at row %lld, col %lld: got %g, expected %g (tol %g)\n",
              who, (long long)(i % N1), (long long)(i / N1), double(float(hx)), g, atol + kRtol * std::fabs(g));
    }
    bool const ok = !st.bad && (rel_rms_err <= kRmsTol);
    if (verbose || !ok) {
      printf("%s: max |err|/tol = %.3f, rms(err)/rms(C1) = %.3g (limit %.3g), rms(C1) = %.4g, "
             "%lld of %lld elements out of tolerance\n",
             who, double(max_ratio), rel_rms_err, kRmsTol, rms, (long long)st.bad, (long long)count_C1);
    }
    if (st.bad) {
      return false;
    }
    if (!(rel_rms_err <= kRmsTol)) {
      fprintf(stderr, "%s: C1 RMS error too large: %g > %g\n", who, rel_rms_err, kRmsTol);
      return false;
    }
    return true;
  }

  bool run() {
    initialize();

#ifdef KH_TEST_SOLUTION
    bool const need_golden = true;
#else
    bool const need_golden = verify_reference;
#endif
    // Correctness on both input sets (0: plain random, 1: realistic LayerNorm inputs).
    for (int s = 0; s < 2; ++s) {
      restore_inputs(s);
      // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
      cudaMemset(block_C1.get(), 0xFF, count_C1 * sizeof(Element));
      cutlass::Status status = run_reference_();
      cudaError_t err = cudaDeviceSynchronize();
      if (status != cutlass::Status::kSuccess || err != cudaSuccess) {
        std::cerr << "Reference execution failed: " << cutlass::cutlassGetStatusString(status) << ", "
                  << cudaGetErrorString(err) << std::endl;
        return false;
      }
      if (need_golden && !compute_golden(s)) {
        return false;
      }
      if (verify_reference && !check_output(s ? "Reference (realistic set)" : "Reference", block_C1, s)) {
        std::cerr << "Reference verification failed" << std::endl;
        return false;
      }
#ifdef KH_TEST_SOLUTION
      if (!run_and_verify_solution_(s ? "Solution (realistic set)" : "Solution", s)) {
        return false;
      }
#endif
    }

    std::cout << "Passed" << std::endl;
    if (iterations <= 0) {
      return true;
    }

    // KH timing protocol: the two input sets alternate (restored outside the timed region); the
    // solution output is re-poisoned before and verified after every timed call.
    int const iters = kh_env_int("KH_ITERS", this->iterations);  // KH_ITERS override (default unchanged)
    int const warmup = kh_env_int("KH_WARMUP", 3);
    KhTiming ref = kh_bench("Ref time", warmup, iters, [&](int s) { restore_inputs(s); },
                            [&] { run_reference_(); }, kh_no_check);
    fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n", ref.avg_ms, iters, ref.min_ms);
#ifdef KH_TEST_SOLUTION
    KhTiming sol = kh_bench("Kernel time", warmup, iters,
      [&](int s) {
        restore_inputs(s);
        kh_cuda_ok(cudaMemsetAsync(sol_C1.get(), 0xFF, count_C1 * sizeof(Element)), "output poison");
      },
      [&] { run_solution_(); },
      [&](int s) { return check_output("Solution (timed call)", sol_C1, s, false); });
    fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n", sol.avg_ms, iters, sol.min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
    // The outputs must still be produced by a fresh call after the timed runs.
    restore_inputs(0);
    if (!run_and_verify_solution_("Solution (re-run after timing)", 0)) {
      return false;
    }
#endif
    double const flops = 2.0 * T * double(H) * H + 2.0 * T * double(N1) * H;
    printf("Reference: %.1f TFLOP/s\n", flops / ref.min_ms / 1e9);
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
