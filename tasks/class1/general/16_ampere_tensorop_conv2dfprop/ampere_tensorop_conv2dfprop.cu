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

/**
This example shows how to run CUTLASS's convolution kernels
based on the Implicit GEMM algorithm, that use the Tensor Cores
on an NVIDIA Ampere GPU.

Input tensors use cutlass::half_t (F16) in NHWC layout.
Output tensor uses float (F32) in NHWC layout.
Accumulation is done in float.

C = alpha * Conv2dFprop(A, B) + beta * C

Benchmark harness: the reference is the CUTLASS implicit-GEMM kernel with a per-problem-size
tile configuration (selected by an offline sweep on the target GPU). Correctness is checked
against an FP64 golden model computed on the device, element by element.
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include <functional>
#include <string>

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
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/device_memory.h"

#include "helper.h"

// KH_VERIFY_BEGIN
/////////////////////////////////////////////////////////////////////////////////////////////////
// Verification utilities: full-mantissa random inputs, FP64 golden model on the device,
// per-element tolerance check (every element, NaN/Inf fail).
/////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline unsigned kh_hash32(unsigned x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

/// Uniform double in [lo, hi) with 64 random bits per element (no periodic pattern), so every
/// mantissa bit of the stored FP16/FP32 value is random.
__device__ inline double kh_uniform(size_t i, unsigned seed, double lo, double hi) {
  unsigned h1 = kh_hash32(unsigned(i) * 0x9E3779B1u + seed * 0x85EBCA77u + unsigned(i >> 32) * 0xC2B2AE3Du + 0x165667B1u);
  unsigned h2 = kh_hash32(h1 ^ (unsigned(i) * 0x27D4EB2Fu + seed));
  double u = (double(h1) * 4294967296.0 + double(h2)) * (1.0 / 18446744073709551616.0);
  return lo + (hi - lo) * u;
}

template <typename T>
__global__ void kh_fill_uniform_kernel(T *p, size_t n, unsigned seed, double lo, double hi) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = T(float(kh_uniform(i, seed, lo, hi)));
}

template <typename T>
void kh_fill_uniform(T *p, size_t n, unsigned seed, double lo, double hi) {
  kh_fill_uniform_kernel<<<1024, 256>>>(p, n, seed, lo, hi);
}

template <typename T>
__global__ void kh_fill_value_kernel(T *p, size_t n, T v) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = v;
}

/// FP64 tiled "implicit GEMM" golden model: out[o(z,m,n)] = sum_k a(z,m,k) * b(z,k,n), accumulated
/// in double; also accumulates sum |a*b| and sum (a*b)^2 (FP32, used to scale the tolerance).
/// blockIdx.z = cls * splits + split; split-K partial sums are combined with atomics, so the
/// output buffers must be zeroed first.
template <typename Prob>
__global__ void __launch_bounds__(256) kh_golden_kernel(Prob pb, double *G, float *Aabs, float *Asq) {
  __shared__ double As[16][65];
  __shared__ double Bs[16][65];
  __shared__ float Asf[16][65];
  __shared__ float Bsf[16][65];
  int cls = blockIdx.z / pb.splits, split = blockIdx.z % pb.splits;
  int M = pb.rows(cls), Ncol = pb.ncols, Kd = pb.depth(cls);
  int m0 = blockIdx.x * 64, n0 = blockIdx.y * 64;
  if (m0 >= M || n0 >= Ncol) return;
  int chunk = ((Kd + pb.splits - 1) / pb.splits + 15) / 16 * 16;
  int kbeg = split * chunk, kend = min(Kd, kbeg + chunk);
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  double acc[4][4];
  float aab[4][4], asq[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0; aab[i][j] = 0.f; asq[i][j] = 0.f; }
  for (int k0 = kbeg; k0 < kend; k0 += 16) {
    for (int l = 0; l < 4; ++l) {
      int idx = threadIdx.x + l * 256, kk = idx % 16, rr = idx / 16;
      int m = m0 + rr, n = n0 + rr, k = k0 + kk;
      double av = (m < M && k < kend) ? pb.a(cls, m, k) : 0.0;
      double bv = (n < Ncol && k < kend) ? pb.b(cls, k, n) : 0.0;
      As[kk][rr] = av; Asf[kk][rr] = float(av);
      Bs[kk][rr] = bv; Bsf[kk][rr] = float(bv);
    }
    __syncthreads();
    for (int kk = 0; kk < 16; ++kk) {
      double a[4], b[4];
      float af[4], bf[4];
      for (int i = 0; i < 4; ++i) { a[i] = As[kk][ty * 4 + i]; af[i] = Asf[kk][ty * 4 + i]; }
      for (int j = 0; j < 4; ++j) { b[j] = Bs[kk][tx * 4 + j]; bf[j] = Bsf[kk][tx * 4 + j]; }
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fma(a[i], b[j], acc[i][j]);
          float p = af[i] * bf[j];
          aab[i][j] += fabsf(p);
          asq[i][j] = fmaf(p, p, asq[i][j]);
        }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
      if (m < M && n < Ncol) {
        size_t o = pb.out(cls, m, n);
        if (pb.splits == 1) { G[o] = acc[i][j]; Aabs[o] = aab[i][j]; Asq[o] = asq[i][j]; }
        else { atomicAdd(&G[o], acc[i][j]); atomicAdd(&Aabs[o], aab[i][j]); atomicAdd(&Asq[o], asq[i][j]); }
      }
    }
}

template <typename Prob>
void kh_golden(Prob pb, int max_rows, int classes, double *G, float *Aabs, float *Asq, size_t count) {
  cudaMemset(G, 0, count * sizeof(double));
  cudaMemset(Aabs, 0, count * sizeof(float));
  cudaMemset(Asq, 0, count * sizeof(float));
  dim3 grid((max_rows + 63) / 64, (pb.ncols + 63) / 64, classes * pb.splits);
  // (pb.ncols: number of GEMM columns)
  kh_golden_kernel<<<grid, 256>>>(pb, G, Aabs, Asq);
}

/// Result of a tolerance check over every output element.
struct KhCheckResult {
  float max_ratio;                  // max over elements of |got - expected| / tol (inf if NaN/Inf)
  unsigned long long n_bad;         // elements outside tolerance (including NaN/Inf)
  unsigned long long n_nonfinite;   // NaN/Inf elements
  bool ok() const { return n_bad == 0 && n_nonfinite == 0; }
};

/// Cmp functor: void operator()(size_t i, double &got, double &expected, double &tol) const
template <typename Cmp>
__global__ void kh_compare_kernel(Cmp cmp, size_t n, unsigned *max_bits, unsigned long long *n_bad,
                                  unsigned long long *n_nonfinite) {
  float local = 0.f;
  unsigned long long bad = 0, nf = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    double got, expected, tol;
    cmp(i, got, expected, tol);
    float r;
    if (!isfinite(got)) { ++nf; ++bad; r = INFINITY; }
    else {
      double e = fabs(got - expected);
      tol = fmax(tol, 1e-30);
      if (!(e <= tol)) ++bad;
      r = float(e / tol);
      if (!(r >= 0.f)) r = INFINITY;
    }
    local = fmaxf(local, r);
  }
  for (int o = 16; o > 0; o >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, o));
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
    nf += __shfl_xor_sync(0xffffffffu, nf, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(max_bits, __float_as_uint(local));
    if (bad) atomicAdd(n_bad, bad);
    if (nf) atomicAdd(n_nonfinite, nf);
  }
}

template <typename Cmp>
KhCheckResult kh_compare(Cmp cmp, size_t n) {
  static unsigned long long *d_buf = nullptr;
  if (!d_buf) cudaMalloc(&d_buf, 3 * sizeof(unsigned long long));
  cudaMemset(d_buf, 0, 3 * sizeof(unsigned long long));
  kh_compare_kernel<<<1024, 256>>>(cmp, n, reinterpret_cast<unsigned *>(d_buf), d_buf + 1, d_buf + 2);
  unsigned long long h[3];
  cudaMemcpy(h, d_buf, sizeof(h), cudaMemcpyDeviceToHost);
  KhCheckResult r;
  unsigned bits = unsigned(h[0] & 0xffffffffu);
  memcpy(&r.max_ratio, &bits, sizeof(float));
  r.n_bad = h[1];
  r.n_nonfinite = h[2];
  return r;
}
// KH_VERIFY_END

/// Golden problem: implicit GEMM view of conv2d fprop (M = N*P*Q, N = K, K-dim = R*S*C).
struct KhFpropProblem {
  cutlass::half_t const *x, *w;
  int N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w;
  int splits, ncols;
  __device__ int rows(int) const { return N * P * Q; }
  __device__ int depth(int) const { return R * S * C; }
  __device__ double a(int, int m, int k) const {
    int c = k % C, rs = k / C, s = rs % S, r = rs / S;
    int q = m % Q, t = m / Q, p = t % P, n = t / P;
    int h = p * stride_h + r - pad_h, wi = q * stride_w + s - pad_w;
    if (h < 0 || h >= H || wi < 0 || wi >= W) return 0.0;
    return double(float(x[((size_t(n) * H + h) * W + wi) * C + c]));
  }
  __device__ double b(int, int k, int n) const { return double(float(w[size_t(n) * R * S * C + k])); }
  __device__ size_t out(int, int m, int n) const { return size_t(m) * K + n; }
};

/// expected = alpha * conv + beta * C;  tol = kTolScale * (|alpha| * sum|x*w| + |beta * C|)
struct KhFpropCompare {
  float const *got; double const *G; float const *Aabs; float const *Cin;
  float alpha, beta; double tol_scale;
  __device__ void operator()(size_t i, double &g, double &e, double &tol) const {
    double c = Cin ? double(Cin[i]) : 0.0;
    g = got[i];
    e = double(alpha) * G[i] + double(beta) * c;
    tol = tol_scale * (fabs(double(alpha)) * Aabs[i] + fabs(double(beta) * c));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Data types for input and output tensors and computation between elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in filter tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized memory access.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

/// CUTLASS Ampere tensor-core fprop (F16 x F16 -> F32 accumulate -> F32) for a given tile configuration.
template <int TbM, int TbN, int TbK, int WM, int WN, int WK, int Stages>
struct FpropConfig {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM, TbN, TbK>,
    cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided
  >::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

// Tile configurations; the per-size choice comes from an offline sweep on the target GPU.
using Fprop_128x128x64_s3 = FpropConfig<128, 128, 64, 64, 64, 64, 3>;   // default (custom sizes)
using Fprop_128x64x32_s4  = FpropConfig<128,  64, 32, 64, 32, 32, 4>;
using Fprop_128x64x64_s3  = FpropConfig<128,  64, 64, 64, 32, 64, 3>;
using Fprop_128x256x32_s3 = FpropConfig<128, 256, 32, 64, 64, 32, 3>;

/// A prepared (initialized) reference convolution; run() only launches the kernel.
struct RefConv {
  std::function<cutlass::Status()> run;
  std::string name;
};

template <typename Op>
cutlass::Status MakeRefConv(RefConv &rc, char const *name,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    cutlass::half_t const *input, cutlass::half_t const *filter, float const *C, float *D,
    cutlass::Tensor4DCoord input_size, cutlass::Tensor4DCoord filter_size, cutlass::Tensor4DCoord output_size,
    float alpha, float beta) {
  typename Op::Arguments arguments{
    problem_size,
    {const_cast<cutlass::half_t *>(input), LayoutInputA::packed(input_size)},
    {const_cast<cutlass::half_t *>(filter), LayoutInputB::packed(filter_size)},
    {const_cast<float *>(C), LayoutOutput::packed(output_size)},
    {D, LayoutOutput::packed(output_size)},
    {alpha, beta},
  };
  auto op = std::make_shared<Op>();
  auto workspace = std::make_shared<cutlass::device_memory::allocation<uint8_t>>(op->get_workspace_size(arguments));
  cutlass::Status status = op->can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;
  status = op->initialize(arguments, workspace->get());
  if (status != cutlass::Status::kSuccess) return status;
  rc.name = name;
  rc.run = [op, workspace]() { return op->run(); };
  return cutlass::Status::kSuccess;
}

struct TestConfig {
  const char* label;
  int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
};

/// Per-size reference kernel selection (benchmark sizes); anything else uses the default tile.
cutlass::Status PrepareReference(RefConv &rc, TestConfig const &cfg,
    cutlass::conv::Conv2dProblemSize const &ps, cutlass::half_t const *x, cutlass::half_t const *w,
    float const *C, float *D, cutlass::Tensor4DCoord is, cutlass::Tensor4DCoord fs, cutlass::Tensor4DCoord os,
    float alpha, float beta) {
  std::string l = cfg.label;
  if (l == "resnet50_conv2_b256")
    return MakeRefConv<Fprop_128x64x32_s4::Op>(rc, "128x64x32_s4", ps, x, w, C, D, is, fs, os, alpha, beta);
  if (l == "resnet50_conv4_b256" || l == "sdxl_unet_128" || l == "sdxl_unet_64")
    return MakeRefConv<Fprop_128x64x64_s3::Op>(rc, "128x64x64_s3", ps, x, w, C, D, is, fs, os, alpha, beta);
  if (l == "sdxl_unet_32")
    return MakeRefConv<Fprop_128x256x32_s3::Op>(rc, "128x256x32_s3", ps, x, w, C, D, is, fs, os, alpha, beta);
  return MakeRefConv<Fprop_128x128x64_s3::Op>(rc, "128x128x64_s3", ps, x, w, C, D, is, fs, os, alpha, beta);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// The solution is included after all reference / verification code so that its macros cannot
// change the harness.
#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Tolerance: |out - expected| <= kTolScale * (|alpha| * sum_{c,r,s} |x*w| + |beta * C|), where
// `expected` is the FP64 golden result for the given FP16 inputs. FP32 accumulation of the exact
// FP16 products stays far below this; FP16 products/accumulation or BF16/FP8/INT8 operands do not.
constexpr double kTolScale = 5.0e-6;

#ifdef KH_TEST_SOLUTION
// Second correctness check: non-trivial alpha/beta with the prior output values as source.
constexpr float kCheckAlpha = 0.8125f;
constexpr float kCheckBeta = -1.375f;

// Finite poison for the output buffer when beta == 0 (beta * C must then contribute 0; any
// element the solution does not write stays at this value and fails the check).
constexpr float kPoison = -1.0e30f;
#endif

static bool ReportCheck(char const *what, KhCheckResult const &r) {
  fprintf(stdout, "%s: max err/tol = %.3g, elements outside tolerance = %llu, non-finite = %llu\n",
          what, double(r.max_ratio), r.n_bad, r.n_nonfinite);
  return r.ok();
}

/// Runs correctness checks and (if iterations > 0) timing for one problem size.
/// Returns false on any failure.
bool RunSize(TestConfig const &cfg, float alpha, float beta, int iterations) {
  int N = cfg.N, H = cfg.H, W = cfg.W, C = cfg.C, K = cfg.K, R = cfg.R, S = cfg.S;
  int pad_h = cfg.pad_h, pad_w = cfg.pad_w, stride_h = cfg.stride_h, stride_w = cfg.stride_w;
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);
  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, cutlass::Tensor4DCoord(pad_h, pad_h, pad_w, pad_w),
      cutlass::MatrixCoord(stride_h, stride_w), cutlass::MatrixCoord(1, 1),
      output_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  size_t in_count = size_t(input_size.product()), f_count = size_t(filter_size.product());
  size_t out_count = size_t(output_size.product());

  cutlass::device_memory::allocation<cutlass::half_t> d_x(in_count), d_w(f_count);
  cutlass::device_memory::allocation<float> d_c(out_count), d_ref(out_count), d_sol(out_count);
  cutlass::device_memory::allocation<double> d_golden(out_count);
  cutlass::device_memory::allocation<float> d_abs(out_count), d_sq(out_count);

  // Random inputs with full FP16 mantissa, uniform in [-1, 1); C (source) uniform in [-1, 1).
  kh_fill_uniform(d_x.get(), in_count, 11u, -1.0, 1.0);
  kh_fill_uniform(d_w.get(), f_count, 23u, -1.0, 1.0);
  kh_fill_uniform(d_c.get(), out_count, 37u, -1.0, 1.0);

  // FP64 golden model on the device.
  KhFpropProblem pb{d_x.get(), d_w.get(), N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, 1, K};
  kh_golden(pb, N * P * Q, 1, d_golden.get(), d_abs.get(), d_sq.get(), out_count);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Golden model failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Reference (CUTLASS) with the timed alpha/beta.
  RefConv ref;
  cutlass::Status st = PrepareReference(ref, cfg, problem_size, d_x.get(), d_w.get(), d_c.get(), d_ref.get(),
                                        input_size, filter_size, output_size, alpha, beta);
  if (st != cutlass::Status::kSuccess || ref.run() != cutlass::Status::kSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "CUTLASS reference failed: " << cutlassGetStatusString(st) << std::endl;
    return false;
  }
  KhFpropCompare cmp{d_ref.get(), d_golden.get(), d_abs.get(), d_c.get(), alpha, beta, kTolScale};
  if (!ReportCheck("Reference check (CUTLASS vs FP64 golden)", kh_compare(cmp, out_count))) {
    std::cerr << "CUTLASS reference incorrect vs FP64 golden model." << std::endl;
    return false;
  }

#ifdef KH_TEST_SOLUTION
  // Prepares the solution's output buffer: poison (beta == 0) or the source values C (beta != 0).
  auto prepare_output = [&](float b) {
    if (b == 0.0f) kh_fill_value_kernel<<<1024, 256>>>(d_sol.get(), out_count, kPoison);
    else cudaMemcpy(d_sol.get(), d_c.get(), out_count * sizeof(float), cudaMemcpyDeviceToDevice);
  };
  auto check_solution = [&](char const *what, float a, float b) {
    KhFpropCompare c{d_sol.get(), d_golden.get(), d_abs.get(), d_c.get(), a, b, kTolScale};
    return ReportCheck(what, kh_compare(c, out_count));
  };
  auto call_solution = [&](float a, float b) {
    return Conv2dFpropF16(
      reinterpret_cast<__half const *>(d_x.get()), reinterpret_cast<__half const *>(d_w.get()),
      d_sol.get(), N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, a, b);
  };

  float const check_alpha[2] = {alpha, kCheckAlpha};
  float const check_beta[2] = {beta, kCheckBeta};
  for (int t = 0; t < 2; ++t) {
    prepare_output(check_beta[t]);
    cudaDeviceSynchronize();
    cudaError_t result = call_solution(check_alpha[t], check_beta[t]);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result != cudaSuccess || sync != cudaSuccess) {
      std::cerr << "Solution kernel failed: "
                << cudaGetErrorString(result != cudaSuccess ? result : sync) << std::endl;
      return false;
    }
    char what[128];
    snprintf(what, sizeof(what), "Solution check (alpha=%g, beta=%g)", check_alpha[t], check_beta[t]);
    if (!check_solution(what, check_alpha[t], check_beta[t])) {
      std::cerr << "Solution incorrect (" << what << ")." << std::endl;
      return false;
    }
  }
#endif

  std::cout << "Passed" << std::endl;
  iterations = kh_env_int("KH_ITERS", iterations);  // KH_ITERS override (default unchanged)
  if (iterations <= 0) return true;
  int const warmup = kh_env_int("KH_WARMUP", 3);

  // Timing input sets, restored into d_x / d_w / d_c before every call (KH timing protocol):
  // set 0 = the correctness data above, set 1 = new random data with its own FP64 golden.
  cutlass::device_memory::allocation<cutlass::half_t> x_set0(in_count), w_set0(f_count);
  cutlass::device_memory::allocation<cutlass::half_t> x_set1(in_count), w_set1(f_count);
  cutlass::device_memory::allocation<float> c_set0(out_count), c_set1(out_count);
  kh_copy(x_set0.get(), d_x.get(), in_count * sizeof(cutlass::half_t));
  kh_copy(w_set0.get(), d_w.get(), f_count * sizeof(cutlass::half_t));
  kh_copy(c_set0.get(), d_c.get(), out_count * sizeof(float));
  kh_fill_uniform(x_set1.get(), in_count, 1011u, -1.0, 1.0);
  kh_fill_uniform(w_set1.get(), f_count, 1023u, -1.0, 1.0);
  kh_fill_uniform(c_set1.get(), out_count, 1037u, -1.0, 1.0);
  cutlass::half_t const *x_src[2] = {x_set0.get(), x_set1.get()};
  cutlass::half_t const *w_src[2] = {w_set0.get(), w_set1.get()};
  float const *c_src[2] = {c_set0.get(), c_set1.get()};
  auto restore_inputs = [&](int s) {
    kh_copy(d_x.get(), x_src[s], in_count * sizeof(cutlass::half_t));
    kh_copy(d_w.get(), w_src[s], f_count * sizeof(cutlass::half_t));
    kh_copy(d_c.get(), c_src[s], out_count * sizeof(float));
  };

  // Reference (kernel launch only; the CUTLASS operator is initialized once above).
  KhTiming rt = kh_bench("Ref time", warmup, iterations, restore_inputs, [&] { ref.run(); },
                         kh_no_check);
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms) [config %s]\n",
          rt.avg_ms, iterations, rt.min_ms, ref.name.c_str());

#ifdef KH_TEST_SOLUTION
  {
  cutlass::device_memory::allocation<double> golden1(out_count);
  cutlass::device_memory::allocation<float> abs1(out_count), sq1(out_count);
  KhFpropProblem pb1 = pb;
  pb1.x = x_set1.get();
  pb1.w = w_set1.get();
  kh_golden(pb1, N * P * Q, 1, golden1.get(), abs1.get(), sq1.get(), out_count);
  kh_cuda_ok(cudaDeviceSynchronize(), "Golden model (timing set 1)");
  double const *g_src[2] = {d_golden.get(), golden1.get()};
  float const *abs_src[2] = {d_abs.get(), abs1.get()};

  // Every timed call gets the alternate input set and a re-poisoned output, and is verified.
  KhTiming st = kh_bench("Kernel time", warmup, iterations,
    [&](int s) { restore_inputs(s); prepare_output(beta); },
    [&] { call_solution(alpha, beta); },
    [&](int s) {
      KhFpropCompare c{d_sol.get(), g_src[s], abs_src[s], c_src[s], alpha, beta, kTolScale};
      KhCheckResult r = kh_compare(c, out_count);
      if (!r.ok()) ReportCheck("Solution check (timed call)", r);
      return r.ok();
    });
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          st.avg_ms, iterations, st.min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", rt.min_ms / st.min_ms);
  }
#endif

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Ampere Tensor Core operations require SM80+
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.major < 8) {
    std::cerr << "Ampere Tensor Ops require compute capability >= 80." << std::endl;
    return 0;
  }

  bool explicit_size = (argc >= 8);
  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int N = 1, H = 56, W = 56, C = 64, K = 64, R = 3, S = 3;
    std::stringstream(arg[1]) >> N;
    std::stringstream(arg[2]) >> H;
    std::stringstream(arg[3]) >> W;
    std::stringstream(arg[4]) >> C;
    std::stringstream(arg[5]) >> K;
    std::stringstream(arg[6]) >> R;
    std::stringstream(arg[7]) >> S;
    int pad_h = R / 2, pad_w = S / 2;
    if (argc > 8) std::stringstream(arg[8]) >> alpha;
    if (argc > 9) std::stringstream(arg[9]) >> beta;
    if (argc > 10) std::stringstream(arg[10]) >> iterations;
    configs.push_back({"custom", N, H, W, C, K, R, S, pad_h, pad_w, 1, 1});
  } else {
    // Real 3x3 conv layers: ResNet-50 at training batch 256, and the SDXL UNet at 1024x1024
    // (128x128 latents; batch 4 = 2 images x classifier-free guidance).
    configs = {
      {"resnet50_conv2_b256",      256,  56,  56,   64,   64, 3, 3, 1, 1, 1, 1},
      {"resnet50_conv3_down_b256", 256,  56,  56,  128,  128, 3, 3, 1, 1, 2, 2},
      {"resnet50_conv4_b256",      256,  14,  14,  256,  256, 3, 3, 1, 1, 1, 1},
      {"sdxl_unet_128",              4, 128, 128,  320,  320, 3, 3, 1, 1, 1, 1},
      {"sdxl_unet_64",               4,  64,  64,  640,  640, 3, 3, 1, 1, 1, 1},
      {"sdxl_unet_32",               4,  32,  32, 1280, 1280, 3, 3, 1, 1, 1, 1},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: N=%d, H=%d, W=%d, C=%d, K=%d, R=%d, S=%d, pad=%d, stride=%d ===\n",
            cfg.label, cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S, cfg.pad_h, cfg.stride_h);
    if (!RunSize(cfg, alpha, beta, iterations)) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
