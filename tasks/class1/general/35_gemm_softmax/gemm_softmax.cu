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

// Tolerances. D: |x - g| <= kAtolD + kRtolD * |g|. Softmax: |x - s| <= kAtolS + |s| * (kRtolS +
// kUlpS * (h(g) + H)) and |row sum - 1| <= kRowSumTol + H, with s = softmax of the FP32 golden
// logits g, h(v) = half an FP16 ulp at |v| and H = sum_j s_j h(g_j) over the row: a softmax computed
// from the FP16-rounded D deviates by up to h(g) + H per element, which grows with the logit
// magnitude, and is allowed kUlpS times over; normalizing such a softmax with the FP32 logits shifts
// the row sum by up to H. Non-finite values always fail.
// Reference (FP32 accumulation, FP16 outputs, softmax of the FP16-rounded D), max error / tolerance
// over all sizes: D 0.22 (input set 0) / 0.24 (input set 1), Softmax 0.29 / 0.37, row sum <= 0.2;
// FP16 accumulation exceeds the D tolerance by 3-21x, BF16 operands by 13-16x.
static constexpr float kAtolD = 1e-3f;
static constexpr float kRtolD = 2e-3f;
static constexpr float kAtolS = 1e-6f;
static constexpr float kRtolS = 3e-3f;
static constexpr float kUlpS = 2.5f;
static constexpr float kRowSumTol = 5e-4f;

struct RowCheck {
  float max_ratio_d;      // max |err| / tol over the row (D)
  float max_ratio_s;      // max |err| / tol over the row (Softmax)
  double row_sum;         // sum of the candidate's Softmax row
  float sum_ratio;        // |row_sum - 1| / its tolerance (inf if non-finite)
  int fails;              // elements out of tolerance (D or Softmax)
  int first_col;          // first failing column (or -1)
};

/// Golden softmax statistics of one row (computed once per input set).
struct RowGold {
  float mx;               // row maximum of the golden logits
  double sum_exp;         // sum_j exp(g_j - mx)
  double h_mean;          // sum_j s_j * h(g_j)
};

/// Half an FP16 ulp at |x| (half the subnormal spacing below the smallest normal).
__device__ __forceinline__ float kh_half_ulp16(float x) {
  x = fabsf(x);
  if (!(x >= 6.103515625e-05f)) return 2.98023224e-08f;
  int e;
  frexpf(x, &e);
  return ldexpf(1.0f, e - 12);
}

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

/// One block per row of the golden logits G (all batches, rows of length N).
__global__ void golden_row_stats_kernel(int N, float const *G, RowGold *out) {
  __shared__ float redf[32];
  __shared__ double redd[32];
  int64_t const row = blockIdx.x;
  G += row * N;
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, G[j]);
  mx = block_reduce_max(mx, redf);
  double se = 0, sh = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    double const e = exp(double(G[j]) - double(mx));
    se += e;
    sh += e * double(kh_half_ulp16(G[j]));
  }
  double const sum_exp = block_reduce_sum(se, redd);
  double const sum_h = block_reduce_sum(sh, redd);
  if (threadIdx.x == 0) out[row] = RowGold{mx, sum_exp, sum_h / sum_exp};
}

/// One block per row: element-wise checks of D and Softmax (FP16) against the golden row and its
/// softmax, and the row sum.
__global__ void check_rows_kernel(int N, ElementOutput const *D, ElementOutput const *S,
                                  float const *G, RowGold const *gold_rows, RowCheck *out) {
  __shared__ float redf[32];
  __shared__ double redd[32];
  __shared__ int first_bad;
  int64_t const row = blockIdx.x;
  D += row * N;
  S += row * N;
  G += row * N;
  RowGold const rg = gold_rows[row];

  if (threadIdx.x == 0) first_bad = N;
  __syncthreads();

  float rd = 0, rs = 0;
  double row_sum = 0;
  int fails = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float const g = G[j];
    float const gs = float(exp(double(g) - double(rg.mx)) / rg.sum_exp);
    float const d = float(D[j]);
    float const s = float(S[j]);
    float const td = kAtolD + kRtolD * fabsf(g);
    float const ts = kAtolS + fabsf(gs) * (kRtolS + kUlpS * (kh_half_ulp16(g) + float(rg.h_mean)));
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
    float const sum_tol = kRowSumTol + float(rg.h_mean);
    float const sum_ratio = isfinite(row_sum) ? float(fabs(row_sum - 1.0)) / sum_tol : INFINITY;
    out[row] = RowCheck{rd, rs, row_sum, sum_ratio, int(nf), first_bad < N ? first_bad : -1};
  }
}

/// Reduction of all row checks: max ratios, max |row sum - 1|, total failures and the first row that
/// fails (element failures or row sum).
struct CheckSummary {
  unsigned max_rd_bits, max_rs_bits, max_se_bits, max_sr_bits;
  unsigned long long fails, first_bad_row;
};

__global__ void summarize_rows_kernel(RowCheck const *rc, int64_t rows, CheckSummary *out) {
  float rd = 0, rs = 0, se_max = 0, sr = 0;
  unsigned long long fails = 0, first = ~0ull;
  for (int64_t r = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; r < rows;
       r += int64_t(gridDim.x) * blockDim.x) {
    RowCheck const c = rc[r];
    float const se = isfinite(c.row_sum) ? float(fabs(c.row_sum - 1.0)) : INFINITY;
    rd = fmaxf(rd, c.max_ratio_d);
    rs = fmaxf(rs, c.max_ratio_s);
    se_max = fmaxf(se_max, se);
    sr = fmaxf(sr, c.sum_ratio);
    fails += (unsigned long long)c.fails;
    if ((c.fails || !(c.sum_ratio <= 1.0f)) && (unsigned long long)r < first) first = (unsigned long long)r;
  }
  for (int o = 16; o > 0; o >>= 1) {
    rd = fmaxf(rd, __shfl_xor_sync(0xffffffffu, rd, o));
    rs = fmaxf(rs, __shfl_xor_sync(0xffffffffu, rs, o));
    se_max = fmaxf(se_max, __shfl_xor_sync(0xffffffffu, se_max, o));
    sr = fmaxf(sr, __shfl_xor_sync(0xffffffffu, sr, o));
    fails += __shfl_xor_sync(0xffffffffu, fails, o);
    unsigned long long const f = __shfl_xor_sync(0xffffffffu, first, o);
    first = f < first ? f : first;
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(&out->max_rd_bits, __float_as_uint(rd));
    atomicMax(&out->max_rs_bits, __float_as_uint(rs));
    atomicMax(&out->max_se_bits, __float_as_uint(se_max));
    atomicMax(&out->max_sr_bits, __float_as_uint(sr));
    if (fails) atomicAdd(&out->fails, fails);
    if (first != ~0ull) atomicMin(&out->first_bad_row, first);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Input sets. Set 0: A, B, C uniform in [-2, 2) (logits with a standard deviation of ~1.8).
// Set 1 (attention-like): A = 3 r + u(b,k), with u = +-1 shared by all rows of a batch, B = 3.5 r
// except one "sink" column n_s(b) (like a BOS token) equal to (16 / sqrt(K)) u(b,k), C uniform in
// [-1, 1) (r uniform in [-1, 1)). The logits alpha * A.B (alpha = 1/sqrt(K)) then have a standard
// deviation of ~4, and the sink column a logit of ~ +16 for every row.

__host__ __device__ inline uint64_t kh_mix64(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 33;
  return x;
}

/// Uniform in [-1, 1) with 24 random bits.
__device__ inline float kh_unif(uint64_t i, uint32_t seed) {
  uint64_t const h = kh_mix64(i * 0x9E3779B97F4A7C15ull + (uint64_t(seed) + 1) * 0xD1B54A32D192ED03ull);
  return float(int32_t(uint32_t(h >> 32)) >> 8) * (1.0f / 8388608.0f);
}

__device__ inline float kh_sign(uint64_t i, uint32_t seed) { return (kh_mix64(i + (uint64_t(seed) << 40)) & 1) ? 1.f : -1.f; }

__host__ __device__ inline int kh_sink_col(int b, int N, uint32_t seed) {
  return int(kh_mix64(uint64_t(b) * 0x9E3779B97F4A7C15ull + seed) % uint64_t(N));
}

/// A: row-major [M, K] per batch.
__global__ void fill_set1_A_kernel(ElementInput *A, int M, int K, int64_t count, uint32_t seed) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t const b = i / (int64_t(M) * K);
    int const k = int(i % K);
    A[i] = ElementInput(3.0f * kh_unif(i, seed) + kh_sign(b * K + k, seed + 7));
  }
}

/// B: column-major [K, N] per batch; the sink column shares u(b,k) with the rows of A.
__global__ void fill_set1_B_kernel(ElementInput *B, int K, int N, int64_t count, uint32_t seed_u,
                                   uint32_t seed, float sink) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t const b = i / (int64_t(K) * N);
    int const n = int((i / K) % N), k = int(i % K);
    float const v = (n == kh_sink_col(int(b), N, seed)) ? sink * kh_sign(b * K + k, seed_u + 7)
                                                        : 3.5f * kh_unif(i, seed);
    B[i] = ElementInput(v);
  }
}

__global__ void fill_uniform_kernel(ElementInput *p, int64_t count, uint32_t seed, float scale) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x)
    p[i] = ElementInput(scale * kh_unif(i, seed));
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

  // Buffers passed to the reference / solution, and the two input sets restored into them.
  cutlass::DeviceAllocation<ElementInput> block_A, block_B, block_C;
  cutlass::DeviceAllocation<ElementInput> set_A[2], set_B[2], set_C[2];
  cutlass::DeviceAllocation<ElementOutput> ref_D, ref_Softmax;
  cutlass::DeviceAllocation<float> block_Norm, block_Sum;
  cutlass::DeviceAllocation<float> gold_D[2];        // FP32 golden logits of each input set (all batches)
  cutlass::DeviceAllocation<RowGold> gold_rows[2];   // golden softmax statistics per row
  cutlass::DeviceAllocation<RowCheck> row_checks;
  cutlass::DeviceAllocation<CheckSummary> summary;
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
    for (int s = 0; s < 2; ++s) {
      set_A[s].reset(count_A);
      set_B[s].reset(count_B);
      set_C[s].reset(count_C);
    }

    // Set 0: uniform random values in [-2, 2) at full FP16 precision.
    cutlass::reference::device::BlockFillRandomUniform(set_A[0].get(), count_A, seed, ElementInput(2), ElementInput(-2), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_B[0].get(), count_B, seed + 1, ElementInput(2), ElementInput(-2), -1);
    cutlass::reference::device::BlockFillRandomUniform(set_C[0].get(), count_C, seed + 2, ElementInput(2), ElementInput(-2), -1);
    // Set 1: attention-like logits (std ~4 plus a sink column, see above).
    fill_set1_A_kernel<<<4096, 256>>>(set_A[1].get(), M, K, count_A, seed + 100);
    fill_set1_B_kernel<<<4096, 256>>>(set_B[1].get(), K, N, count_B, seed + 100, seed + 101,
                                      16.0f / std::sqrt(float(K)));
    fill_uniform_kernel<<<4096, 256>>>(set_C[1].get(), count_C, seed + 102, 1.0f);

    // Golden logits (FP32 accumulation) and row statistics of both sets, computed once.
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    row_checks.reset(int64_t(M) * batch);
    summary.reset(1);
    for (int s = 0; s < 2; ++s) {
      gold_D[s].reset(count_C);
      gold_rows[s].reset(int64_t(M) * batch);
      cutlass::reference::device::GemmComplex<
          ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC, float, float, float>(
        {M, N, K}, alpha,
        cutlass::TensorRef<ElementInput, LayoutA>(set_A[s].get(), LayoutA(K)),
        cutlass::ComplexTransform::kNone,
        cutlass::TensorRef<ElementInput, LayoutB>(set_B[s].get(), LayoutB(K)),
        cutlass::ComplexTransform::kNone,
        beta,
        cutlass::TensorRef<ElementInput, LayoutC>(set_C[s].get(), LayoutC(N)),
        cutlass::TensorRef<float, LayoutC>(gold_D[s].get(), LayoutC(N)),
        0.0f, batch, int64_t(M) * K, int64_t(K) * N, mn, mn);
      golden_row_stats_kernel<<<unsigned(int64_t(M) * batch), 256>>>(N, gold_D[s].get(), gold_rows[s].get());
    }

    restore_inputs(0);
    // 0xFFFF is a NaN in FP16: any element left unwritten fails verification.
    poison(ref_D.get(), ref_Softmax.get());
#ifdef KH_TEST_SOLUTION
    sol_D.reset(count_C);
    sol_Softmax.reset(count_C);
#endif
  }

  /// Copies input set s into the buffers passed to the reference / solution.
  void restore_inputs(int s) {
    kh_copy(block_A.get(), set_A[s].get(), count_A * sizeof(ElementInput));
    kh_copy(block_B.get(), set_B[s].get(), count_B * sizeof(ElementInput));
    kh_copy(block_C.get(), set_C[s].get(), count_C * sizeof(ElementInput));
  }

  void poison(ElementOutput *D, ElementOutput *S) {
    cudaMemsetAsync(D, 0xFF, count_C * sizeof(ElementOutput));
    cudaMemsetAsync(S, 0xFF, count_C * sizeof(ElementOutput));
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

  /// Verifies outputs D and S (all batches) against the golden result of input set `set` on the
  /// device. Prints a summary line unless `quiet` (failures are always reported).
  bool verify(char const *who, ElementOutput const *D, ElementOutput const *S, int set, bool quiet = false) {
    int64_t const rows = int64_t(M) * batch;
    CheckSummary init{0u, 0u, 0u, 0u, 0ull, ~0ull}, h;
    cudaMemcpy(summary.get(), &init, sizeof(init), cudaMemcpyHostToDevice);
    check_rows_kernel<<<unsigned(rows), 256>>>(N, D, S, gold_D[set].get(), gold_rows[set].get(), row_checks.get());
    summarize_rows_kernel<<<1024, 256>>>(row_checks.get(), rows, summary.get());
    cudaError_t err = cudaMemcpy(&h, summary.get(), sizeof(h), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Verification kernel failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    float max_rd, max_rs, max_se, max_sr;
    std::memcpy(&max_rd, &h.max_rd_bits, sizeof(float));
    std::memcpy(&max_rs, &h.max_rs_bits, sizeof(float));
    std::memcpy(&max_se, &h.max_se_bits, sizeof(float));
    std::memcpy(&max_sr, &h.max_sr_bits, sizeof(float));
    bool const ok = (h.first_bad_row == ~0ull);
    if (!ok) {
      int64_t const r = int64_t(h.first_bad_row);
      int const b = int(r / M), row = int(r % M);
      RowCheck c;
      cutlass::device_memory::copy_to_host(&c, row_checks.get() + r, 1);
      if (c.fails) {
        int const col = c.first_col;
        float gd = 0;
        ElementOutput hd, hs;
        cutlass::device_memory::copy_to_host(&gd, gold_D[set].get() + r * N + col, 1);
        cutlass::device_memory::copy_to_host(&hd, D + r * N + col, 1);
        cutlass::device_memory::copy_to_host(&hs, S + r * N + col, 1);
        fprintf(stderr, "%s: mismatch (input set %d) at batch %d, row %d, col %d: D = %g (golden %g), Softmax = %g\n",
                who, set, b, row, col, float(hd), gd, float(hs));
      } else {
        fprintf(stderr, "%s: Softmax row %d of batch %d sums to %.6f (input set %d, |sum - 1| / tolerance = %.3f)\n",
                who, row, b, c.row_sum, set, c.sum_ratio);
      }
    }
    if (!quiet || !ok) {
      printf("%s: max |err|/tol D = %.3f, Softmax = %.3f; max |row sum - 1| = %.3g (max ratio to its "
             "tolerance %.3f); %lld of %lld elements out of tolerance\n",
             who, max_rd, max_rs, max_se, max_sr, (long long)h.fails, (long long)count_C);
    }
    return ok;
  }

#ifdef KH_TEST_SOLUTION
  /// Restores input set `set`, poisons the solution outputs (NaN), runs the solution once and
  /// verifies it.
  bool run_and_verify_solution_(char const *who, int set) {
    restore_inputs(set);
    poison(sol_D.get(), sol_Softmax.get());
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = run_solution_();
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      std::cerr << who << ": execution failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return verify(who, sol_D.get(), sol_Softmax.get(), set);
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

    if (verify_reference) {
      if (!verify("Reference", ref_D.get(), ref_Softmax.get(), 0)) {
        std::cerr << "Reference verification failed" << std::endl;
        return false;
      }
      // Reference on input set 1 (attention-like logits with a sink column), once, untimed.
      restore_inputs(1);
      poison(ref_D.get(), ref_Softmax.get());
      status = run_reference_();
      err = cudaDeviceSynchronize();
      if (status != cutlass::Status::kSuccess || err != cudaSuccess ||
          !verify("Reference (input set 1)", ref_D.get(), ref_Softmax.get(), 1)) {
        std::cerr << "Reference verification failed (input set 1)" << std::endl;
        return false;
      }
      restore_inputs(0);
    }

#ifdef KH_TEST_SOLUTION
    if (!run_and_verify_solution_("Solution", 0)) {
      return false;
    }
#endif

    std::cout << "Passed" << std::endl;
    int const iters = kh_env_int("KH_ITERS", this->iterations);  // KH_ITERS override (default unchanged)
    if (iters <= 0) {
      return true;
    }
    int const warmup = kh_env_int("KH_WARMUP", 3);

    // KH timing protocol: input sets 0 and 1 alternate (restored outside the timed region).
    KhTiming const ref = kh_bench("Ref time", warmup, iters, [&](int s) { restore_inputs(s); },
                                  [&] { run_reference_(); }, kh_no_check);
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n", "Ref time", ref.avg_ms, iters, ref.min_ms);
#ifdef KH_TEST_SOLUTION
    KhTiming const sol = kh_bench("Kernel time", warmup, iters,
      [&](int s) { restore_inputs(s); poison(sol_D.get(), sol_Softmax.get()); },
      [&] { run_solution_(); },
      [&](int s) { return verify("Solution (timed call)", sol_D.get(), sol_Softmax.get(), s, true); });
    fprintf(stdout, "%s: %.4f ms (avg over %d trials, min: %.4f ms)\n", "Kernel time", sol.avg_ms, iters, sol.min_ms);
    fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
    // The outputs must still be produced by a fresh call after the timed runs.
    if (!run_and_verify_solution_("Solution (re-run after timing)", 0)) {
      return false;
    }
#endif
    double const flops = 2.0 * M * N * double(K) * batch;
    printf("Reference: %.1f TFLOP/s\n", flops / ref.min_ms / 1e9);
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
