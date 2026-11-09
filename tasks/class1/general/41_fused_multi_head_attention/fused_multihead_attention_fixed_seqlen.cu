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
// Input set 1 (attention-like), BMHK layout, head_dim D. Q and K: channels 0 .. D-3 uniform in
// [-3.5, 3.5) (logit standard deviation ~4). Channel D-1 is a "sink" channel: Q = 8 for every
// query, K = 18 sqrt(D) / 8 for key 0 (like a BOS token) and uniform in [-1, 1) for the other keys,
// so key 0 gets a logit of ~ +18 from every query. Channel D-2 is an offset channel: K = 4 for every
// key and Q = beta(query), which shifts all logits of a query row by beta * 4 / sqrt(D): -120 to
// -150 for 1/8 of the rows (every logit of those rows is very negative), -20 to 20 otherwise.
// V uniform in [-1, 1).

__host__ __device__ inline uint64_t kh_mix64(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 33;
  return x;
}

/// Uniform in [-1, 1) with 24 random bits.
__device__ inline float kh_unif(uint64_t i, uint32_t seed) {
  uint64_t const h = kh_mix64(i * 0x9E3779B97F4A7C15ull + (uint64_t(seed) + 1) * 0xD1B54A32D192ED03ull);
  return float(int32_t(uint32_t(h >> 32)) >> 8) * (1.0f / 8388608.0f);
}

/// which: 0 = Q, 1 = K, 2 = V. `row` = (b * S + m) * H + h indexes one head-row of one token.
__global__ void fill_set1_kernel(cutlass::half_t *p, int which, int D, int S, int H, int64_t count,
                                 uint32_t seed) {
  float const sink_k = 18.0f * sqrtf(float(D)) / 8.0f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
    int const d = int(i % D);
    int64_t const row = i / D;
    int const m = int((row / H) % S);
    float v;
    if (which == 2) {
      v = kh_unif(i, seed);
    } else if (d < D - 2) {
      v = 3.5f * kh_unif(i, seed);
    } else if (d == D - 1) {  // sink channel
      v = (which == 0) ? 8.0f : (m == 0 ? sink_k : kh_unif(i, seed));
    } else {                  // offset channel
      if (which == 1) {
        v = 4.0f;
      } else {
        uint64_t const h = kh_mix64(uint64_t(row) * 0x9E3779B97F4A7C15ull + seed + 17);
        float const u = float(h >> 40) * (1.0f / 16777216.0f);  // [0, 1)
        float const offset = ((h & 7) == 0) ? -120.0f - 30.0f * u : -20.0f + 40.0f * u;
        v = offset * sqrtf(float(D)) / 4.0f;
      }
    }
    p[i] = cutlass::half_t(v);
  }
}

/// Element-wise check statistics of an output against the golden result.
struct OutCheck {
  double err_sq, gold_sq;
  unsigned max_err_bits, max_ratio_bits;
  unsigned long long bad, first_bad;
};

/// |O - gold| <= atol + rtol * |gold| for every element (non-finite values fail), plus the sums
/// for the relative RMS error.
__global__ void check_output_kernel(cutlass::half_t const *X, float const *G, int64_t n, double atol,
                                    double rtol, OutCheck *st) {
  double es = 0, gs = 0;
  float mx = 0, mr = 0;
  unsigned long long bad = 0, first = ~0ull;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    double const v = double(float(X[i]));
    double const g = double(G[i]);
    double err = fabs(v - g);
    double const tol = atol + rtol * fabs(g);
    if (!isfinite(v) || !(err <= tol)) {
      ++bad;
      if ((unsigned long long)i < first) first = (unsigned long long)i;
      if (!isfinite(v)) err = INFINITY;
    }
    mx = fmaxf(mx, float(err));
    mr = fmaxf(mr, float(err / tol));
    es += err * err;
    gs += g * g;
  }
  for (int o = 16; o > 0; o >>= 1) {
    es += __shfl_xor_sync(0xffffffffu, es, o);
    gs += __shfl_xor_sync(0xffffffffu, gs, o);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, o));
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
    unsigned long long const f = __shfl_xor_sync(0xffffffffu, first, o);
    first = f < first ? f : first;
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&st->err_sq, es);
    atomicAdd(&st->gold_sq, gs);
    atomicMax(&st->max_err_bits, __float_as_uint(mx));
    atomicMax(&st->max_ratio_bits, __float_as_uint(mr));
    if (bad) atomicAdd(&st->bad, bad);
    if (first != ~0ull) atomicMin(&st->first_bad, first);
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

  // Input set 0: Q, K uniform in [-kQKRange, kQKRange) and V uniform in [-1, 1), at full FP16
  // precision. The attention logits Q.K / sqrt(head_dim) then have a standard deviation of
  // kQKRange^2 / 3 = 3 (row maxima of about 10-14). Input set 1: attention-like (see above).
  static constexpr float kQKRange = 3.0f;

  // Verification: every element |O - gold| <= kAtol + kRtol * |gold| (non-finite values fail), and
  // rms(O - gold) <= kRmsTol * rms(gold). The golden result is FP32 end to end. The reference
  // (FP16 tensor cores, FP32 accumulation) stays at rms(err)/rms(gold) <= 2.3e-4 (input set 0) and
  // <= 1.9e-4 (input set 1), max |err|/tol <= 0.19, i.e. FP16 output rounding; FP16 accumulation
  // (1.4e-3 - 3e-3 at the long-context sizes), BF16 or FP8 operands (>= 5e-3) exceed the limits.
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

  // Buffers passed to the reference / solution, and the two input sets restored into them.
  cutlass::DeviceAllocation<Element> block_Q;
  cutlass::DeviceAllocation<Element> block_K;
  cutlass::DeviceAllocation<Element> block_V;
  cutlass::DeviceAllocation<Element> set_Q[2], set_K[2], set_V[2];
  cutlass::DeviceAllocation<Element> block_O;
  cutlass::DeviceAllocation<float> block_LSE;
  cutlass::DeviceAllocation<float> block_Gold_O[2];
  cutlass::DeviceAllocation<OutCheck> check_stats_;

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
    check_stats_.reset(1);
    for (int s = 0; s < 2; ++s) {
      set_Q[s].reset(count_);
      set_K[s].reset(count_);
      set_V[s].reset(count_);
    }

    cutlass::reference::device::BlockFillRandomUniform(set_Q[0].get(), count_, seed_ + 1, Element(kQKRange), Element(-kQKRange), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_K[0].get(), count_, seed_ + 2, Element(kQKRange), Element(-kQKRange), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_V[0].get(), count_, seed_ + 3, Element(1.0f), Element(-1.0f), -1);
    int const D = cfg_.head_size, S = cfg_.seq_length, H = cfg_.head_number;
    fill_set1_kernel<<<4096, 256>>>(set_Q[1].get(), 0, D, S, H, count_, seed_ + 101);
    fill_set1_kernel<<<4096, 256>>>(set_K[1].get(), 1, D, S, H, count_, seed_ + 102);
    fill_set1_kernel<<<4096, 256>>>(set_V[1].get(), 2, D, S, H, count_, seed_ + 103);
    restore_inputs_(0);

    // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
    cudaMemset(block_O.get(), 0xFF, count_ * sizeof(Element));
#ifdef KH_TEST_SOLUTION
    sol_O.reset(count_);
#endif
  }

  /// Copies input set s into the buffers passed to the reference / solution.
  void restore_inputs_(int s) {
    kh_copy(block_Q.get(), set_Q[s].get(), count_ * sizeof(Element));
    kh_copy(block_K.get(), set_K[s].get(), count_ * sizeof(Element));
    kh_copy(block_V.get(), set_V[s].get(), count_ * sizeof(Element));
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

  /// Restores input set `set`, poisons the solution output, runs the solution once and verifies it.
  bool run_and_verify_solution_(char const *who, int set) {
    restore_inputs_(set);
    cudaMemset(sol_O.get(), 0xFF, count_ * sizeof(Element));
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return check_output_(who, sol_O, set);
  }
#endif

  /// Golden result of input set s on the device, one (batch, head) at a time:
  ///   S = scale * Q K^T (FP32), P = row-softmax(S) with causal mask (FP32), O = P V (FP32)
  bool compute_golden_(int set) {
    int const S = cfg_.seq_length;
    int const D = cfg_.head_size;
    block_Gold_O[set].reset(count_);
    cutlass::DeviceAllocation<float> block_P(int64_t(S) * S);

    for (int b = 0; b < cfg_.batch_size; ++b) {
      for (int h = 0; h < cfg_.head_number; ++h) {
        int64_t const offset = b * stride_b_ + h * stride_h_;
        cutlass::TensorRef<Element, LayoutQ> ref_Q(set_Q[set].get() + offset, LayoutQ(stride_m_));
        cutlass::TensorRef<Element, LayoutK> ref_K(set_K[set].get() + offset, LayoutK(stride_m_));
        cutlass::TensorRef<Element, LayoutV> ref_V(set_V[set].get() + offset, LayoutV(stride_m_));
        cutlass::TensorRef<float, LayoutP> ref_P(block_P.get(), LayoutP(S));
        cutlass::TensorRef<float, LayoutO> ref_O(block_Gold_O[set].get() + offset, LayoutO(stride_m_));

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
    return true;
  }

  /// Element-wise check of an output against the golden result of input set `set`, on the device
  /// (every element; reports the first mismatch and the error statistics). Prints the statistics
  /// unless `quiet` (failures are always reported).
  bool check_output_(char const *who, cutlass::DeviceAllocation<Element> const &out, int set,
                     bool quiet = false) {
    OutCheck init{0.0, 0.0, 0u, 0u, 0ull, ~0ull}, h;
    cudaMemcpy(check_stats_.get(), &init, sizeof(init), cudaMemcpyHostToDevice);
    check_output_kernel<<<2048, 256>>>(out.get(), block_Gold_O[set].get(), count_, kAtol, kRtol, check_stats_.get());
    cudaError_t err = cudaMemcpy(&h, check_stats_.get(), sizeof(h), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cerr << who << ": verification kernel failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    float max_err, max_ratio;
    std::memcpy(&max_err, &h.max_err_bits, sizeof(float));
    std::memcpy(&max_ratio, &h.max_ratio_bits, sizeof(float));
    if (h.bad) {
      int64_t const i = int64_t(h.first_bad);
      int64_t const b = i / stride_b_, m = (i % stride_b_) / stride_m_;
      int64_t const hh = (i % stride_m_) / stride_h_, d = i % stride_h_;
      Element v;
      float g;
      cutlass::device_memory::copy_to_host(&v, out.get() + i, 1);
      cutlass::device_memory::copy_to_host(&g, block_Gold_O[set].get() + i, 1);
      fprintf(stderr, "%s: mismatch (input set %d) at batch %lld, row %lld, head %lld, dim %lld: got %g, expected %g (tol %g)\n",
              who, set, (long long)b, (long long)m, (long long)hh, (long long)d, double(float(v)), double(g),
              kAtol + kRtol * std::fabs(double(g)));
    }
    double const rel_rms = std::sqrt(h.err_sq / std::max(h.gold_sq, 1e-300));
    bool const ok = !h.bad && (rel_rms <= kRmsTol);
    if (!quiet || !ok) {
      printf("%s: max |err| = %.3g, max |err|/tol = %.3f, rms(err)/rms(ref) = %.3g (limit %.3g), "
             "rms(ref) = %.3g, %lld of %lld elements out of tolerance\n",
             who, double(max_err), double(max_ratio), rel_rms, kRmsTol, std::sqrt(h.gold_sq / double(count_)),
             (long long)h.bad, (long long)count_);
    }
    if (h.bad) {
      return false;
    }
    if (!(rel_rms <= kRmsTol)) {
      fprintf(stderr, "%s: RMS error too large: %g > %g\n", who, rel_rms, kRmsTol);
      return false;
    }
    return true;
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
    if (need_golden && (!compute_golden_(0) || !compute_golden_(1))) {
      return false;
    }

    if (reference_check_) {
      if (!check_output_("Reference", block_O, 0)) {
        std::cerr << "Reference verification failed" << std::endl;
        return false;
      }
      // Reference on input set 1 (attention-like inputs), once, untimed.
      restore_inputs_(1);
      cudaMemset(block_O.get(), 0xFF, count_ * sizeof(Element));
      err = run_reference_();
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      if (err != cudaSuccess || !check_output_("Reference (input set 1)", block_O, 1)) {
        std::cerr << "Reference verification failed (input set 1)" << std::endl;
        return false;
      }
      restore_inputs_(0);
    }

#ifdef KH_TEST_SOLUTION
    if (!run_and_verify_solution_("Solution", 0)) {
      return false;
    }
#endif

    std::cout << "Passed" << std::endl;

    int const iters = kh_env_int("KH_ITERS", this->iterations_);  // KH_ITERS override (default unchanged)
    if (iters <= 0) {
      return true;
    }
    int const warmup = kh_env_int("KH_WARMUP", 3);

    // KH timing protocol: input sets 0 and 1 alternate (restored outside the timed region).
    KhTiming const ref = kh_bench("Ref time", warmup, iters, [&](int s) { restore_inputs_(s); },
                                  [&] { run_reference_(); }, kh_no_check);
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n", "Ref time", ref.avg_ms, iters, ref.min_ms);

#ifdef KH_TEST_SOLUTION
    KhTiming const sol = kh_bench("Kernel time", warmup, iters,
      [&](int s) { restore_inputs_(s); cudaMemsetAsync(sol_O.get(), 0xFF, count_ * sizeof(Element)); },
      [&] { run_solution_(); },
      [&](int s) { return check_output_("Solution (timed call)", sol_O, s, true); });
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n", "Kernel time", sol.avg_ms, iters, sol.min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);

    // The output must still be produced by a fresh call after the timed runs.
    if (!run_and_verify_solution_("Solution (re-run after timing)", 0)) {
      return false;
    }
#endif

    double const flops = 4.0 * cfg_.batch_size * cfg_.head_number * double(cfg_.seq_length) *
                         cfg_.seq_length * cfg_.head_size * (cfg_.causal ? 0.5 : 1.0);
    printf("Reference: %.1f TFLOP/s\n", flops / ref.min_ms / 1e9);
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
