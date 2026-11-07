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

/*
This example shows how to compute 2d transposed convolution, also known as deconvolution, using CUTLASS
conv2d Dgrad kernels. Although two operations are computationaly equivalent, some care is needed to correctly
set up a problem size for CUTLASS.
In deep learning, transposed convolution is sometimes used for upscaling feature maps. This example
demonstrates the 2x upscaling case using the strided Dgrad kernel.

FP16 input/filter/output, FP32 accumulation (the original example accumulates in FP16; this task
requires FP32 accumulation, so the reference uses an FP32 accumulator and an FP16 output epilogue).

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

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
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

/// Golden problem: transposed conv2d, one implicit GEMM per output-pixel parity class
/// (ho % stride_h, wo % stride_w). Within a class the contributing filter taps are fixed:
/// r = r0 + stride_h * i with r0 = (ho + pad_h) % stride_h (same for s), so only the
/// C * ceil((R - r0)/stride_h) * ceil((S - s0)/stride_w) valid taps are visited.
///   output[n][ho][wo][k] = sum_{c,r,s} input[n][h][w][c] * filter[c][r][s][k],
///   ho = h * stride_h + r - pad_h,  wo = w * stride_w + s - pad_w.
struct KhTransposedConvProblem {
  cutlass::half_t const *x, *w;
  int N, C, H, W, K, R, S, Ho, Wo, pad_h, pad_w, stride_h, stride_w;
  int splits, ncols;
  __device__ int ph(int cls) const { return cls / stride_w; }
  __device__ int pw(int cls) const { return cls % stride_w; }
  __device__ int hc(int cls) const { return (Ho - ph(cls) + stride_h - 1) / stride_h; }
  __device__ int wc(int cls) const { return (Wo - pw(cls) + stride_w - 1) / stride_w; }
  __device__ int r0(int cls) const { return (ph(cls) + pad_h) % stride_h; }
  __device__ int s0(int cls) const { return (pw(cls) + pad_w) % stride_w; }
  __device__ int tr(int cls) const { return r0(cls) < R ? (R - r0(cls) + stride_h - 1) / stride_h : 0; }
  __device__ int ts(int cls) const { return s0(cls) < S ? (S - s0(cls) + stride_w - 1) / stride_w : 0; }
  __device__ int rows(int cls) const { return N * hc(cls) * wc(cls); }
  __device__ int depth(int cls) const { return C * tr(cls) * ts(cls); }
  __device__ double a(int cls, int m, int k) const {
    int c = k % C, t = k / C, js = t % ts(cls), ir = t / ts(cls);
    int r = r0(cls) + stride_h * ir, s = s0(cls) + stride_w * js;
    int j = m % wc(cls), u = m / wc(cls), i = u % hc(cls), n = u / hc(cls);
    int ho = ph(cls) + stride_h * i, wo = pw(cls) + stride_w * j;
    int h = (ho + pad_h - r) / stride_h, wi = (wo + pad_w - s) / stride_w;   // exact divisions
    if (ho + pad_h - r < 0 || wo + pad_w - s < 0 || h >= H || wi >= W) return 0.0;
    return double(float(x[((size_t(n) * H + h) * W + wi) * C + c]));
  }
  __device__ double b(int cls, int k, int col) const {
    int c = k % C, t = k / C, js = t % ts(cls), ir = t / ts(cls);
    int r = r0(cls) + stride_h * ir, s = s0(cls) + stride_w * js;
    return double(float(w[((size_t(c) * R + r) * S + s) * K + col]));
  }
  __device__ size_t out(int cls, int m, int col) const {
    int j = m % wc(cls), u = m / wc(cls), i = u % hc(cls), n = u / hc(cls);
    int ho = ph(cls) + stride_h * i, wo = pw(cls) + stride_w * j;
    return ((size_t(n) * Ho + ho) * Wo + wo) * K + col;
  }
};

/// FP16 output: expected = FP64 golden; tol = kTolScale * sum|x*w| + kOutputRoundTol * |expected|
struct KhHalfOutCompare {
  cutlass::half_t const *got; double const *G; float const *Aabs;
  double tol_scale, round_tol;
  __device__ void operator()(size_t i, double &g, double &e, double &tol) const {
    g = double(float(got[i]));
    e = G[i];
    tol = tol_scale * Aabs[i] + round_tol * fabs(e);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Data types for input and output tensors and computation between elements
using ElementAccumulator = float;                  // Data type of accumulator (FP32 accumulation)
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in filter tensor
using ElementOutput = cutlass::half_t;             // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized memory access.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

/// CUTLASS Ampere tensor-core strided dgrad (F16 x F16 -> F32 accumulate -> F16) for a given tile
/// configuration; transposed conv == dgrad with input/output roles swapped.
template <int TbM, int TbN, int TbK, int WM, int WN, int WK, int Stages>
struct DgradConfig {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
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
    cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided  // strided Dgrad specialization
  >::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

// Tile configurations; the per-size choice comes from an offline sweep on the target GPU.
using Dgrad_128x128x32_s3 = DgradConfig<128, 128, 32, 64, 64, 32, 3>;   // also the default (custom sizes)
using Dgrad_128x64x32_s4  = DgradConfig<128,  64, 32, 64, 32, 32, 4>;
using Dgrad_128x32x32_s4  = DgradConfig<128,  32, 32, 32, 32, 32, 4>;

/// A prepared (initialized) reference convolution; run() only launches the kernel.
struct RefConv {
  std::function<cutlass::Status()> run;
  std::string name;
};

/// problem_size is the dgrad problem: its "input" is the transposed-conv output (N,Ho,Wo,K), its
/// "output" is the transposed-conv input (N,H,W,C), and the filter is (C,R,S,K).
template <typename Op>
cutlass::Status MakeRefConv(RefConv &rc, char const *name,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    cutlass::half_t const *input, cutlass::half_t const *filter, cutlass::half_t *D,
    cutlass::Tensor4DCoord input_size, cutlass::Tensor4DCoord filter_size, cutlass::Tensor4DCoord output_size) {
  typename Op::Arguments arguments{
    problem_size,
    {const_cast<cutlass::half_t *>(input), LayoutInputA::packed(input_size)},
    {const_cast<cutlass::half_t *>(filter), LayoutInputB::packed(filter_size)},
    {D, LayoutOutput::packed(output_size)},
    {D, LayoutOutput::packed(output_size)},
    {1.0f, 0.0f},
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

/// H, W, C describe the transposed-conv input; K is the number of output channels.
struct TestConfig {
  const char* label;
  int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
};

/// Per-size reference kernel selection (benchmark sizes); anything else uses the default tile.
cutlass::Status PrepareReference(RefConv &rc, TestConfig const &cfg,
    cutlass::conv::Conv2dProblemSize const &ps, cutlass::half_t const *x, cutlass::half_t const *w,
    cutlass::half_t *D, cutlass::Tensor4DCoord is, cutlass::Tensor4DCoord fs, cutlass::Tensor4DCoord os) {
  std::string l = cfg.label;
  if (l == "decoder_up_128to256")
    return MakeRefConv<Dgrad_128x64x32_s4::Op>(rc, "128x64x32_s4", ps, x, w, D, is, fs, os);
  if (l == "decoder_up_256to512")
    return MakeRefConv<Dgrad_128x32x32_s4::Op>(rc, "128x32x32_s4", ps, x, w, D, is, fs, os);
  return MakeRefConv<Dgrad_128x128x32_s3::Op>(rc, "128x128x32_s3", ps, x, w, D, is, fs, os);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// The solution is included after all reference / verification code so that its macros cannot
// change the harness.
#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Tolerance: |out - expected| <= kTolScale * sum_{c,r,s} |x*w| + kOutputRoundTol * |expected|, where
// `expected` is the FP64 golden result for the given FP16 inputs. Rounding the FP32-accumulated
// result to the FP16 output costs at most 2^-11 * |expected| (half an ulp); the bound allows 4x that.
// FP16 products/accumulation or BF16/FP8/INT8 operands exceed it.
constexpr double kTolScale = 5.0e-6;
constexpr double kOutputRoundTol = 4.0 * 4.8828125e-4;

static bool ReportCheck(char const *what, KhCheckResult const &r) {
  fprintf(stdout, "%s: max err/tol = %.3g, elements outside tolerance = %llu, non-finite = %llu\n",
          what, double(r.max_ratio), r.n_bad, r.n_nonfinite);
  return r.ok();
}

/// Runs correctness checks and (if iterations > 0) timing for one problem size.
/// Returns false on any failure.
bool RunSize(TestConfig const &cfg, int iterations) {
  int N = cfg.N, H = cfg.H, W = cfg.W, C = cfg.C, K = cfg.K, R = cfg.R, S = cfg.S;
  int pad_h = cfg.pad_h, pad_w = cfg.pad_w, stride_h = cfg.stride_h, stride_w = cfg.stride_w;
  // Output size as defined by the task interface (no output_padding):
  // H_out = (H-1)*stride_h + R - 2*pad_h (R=S=4, pad=1, stride=2 gives an exact 2x upsample).
  int Ho = (H - 1) * stride_h - 2 * pad_h + R;
  int Wo = (W - 1) * stride_w - 2 * pad_w + S;

  cutlass::Tensor4DCoord input_size(N, H, W, C);       // transposed-conv input
  cutlass::Tensor4DCoord filter_size(C, R, S, K);      // CRSK
  cutlass::Tensor4DCoord output_size(N, Ho, Wo, K);    // transposed-conv output
  // Dgrad problem: "input" = transposed-conv output, "output" = transposed-conv input.
  cutlass::conv::Conv2dProblemSize problem_size(
      output_size, filter_size, cutlass::Tensor4DCoord(pad_h, pad_h, pad_w, pad_w),
      cutlass::MatrixCoord(stride_h, stride_w), cutlass::MatrixCoord(1, 1),
      input_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  size_t in_count = size_t(input_size.product()), f_count = size_t(filter_size.product());
  size_t out_count = size_t(output_size.product());

  cutlass::device_memory::allocation<cutlass::half_t> d_x(in_count), d_w(f_count);
  cutlass::device_memory::allocation<cutlass::half_t> d_ref(out_count), d_sol(out_count);
  cutlass::device_memory::allocation<double> d_golden(out_count);
  cutlass::device_memory::allocation<float> d_abs(out_count), d_sq(out_count);

  // Random inputs with full FP16 mantissa, uniform in [-1, 1).
  kh_fill_uniform(d_x.get(), in_count, 11u, -1.0, 1.0);
  kh_fill_uniform(d_w.get(), f_count, 23u, -1.0, 1.0);

  // FP64 golden model on the device (one GEMM per output parity class).
  KhTransposedConvProblem pb{d_x.get(), d_w.get(), N, C, H, W, K, R, S, Ho, Wo,
                             pad_h, pad_w, stride_h, stride_w, 1, K};
  int max_class_rows = N * ((Ho + stride_h - 1) / stride_h) * ((Wo + stride_w - 1) / stride_w);
  kh_golden(pb, max_class_rows, stride_h * stride_w, d_golden.get(), d_abs.get(), d_sq.get(), out_count);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Golden model failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Reference (CUTLASS strided dgrad).
  RefConv ref;
  cutlass::Status st = PrepareReference(ref, cfg, problem_size, d_x.get(), d_w.get(), d_ref.get(),
                                        input_size, filter_size, output_size);
  if (st != cutlass::Status::kSuccess || ref.run() != cutlass::Status::kSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "CUTLASS reference failed: " << cutlassGetStatusString(st) << std::endl;
    return false;
  }
  KhHalfOutCompare cmp{d_ref.get(), d_golden.get(), d_abs.get(), kTolScale, kOutputRoundTol};
  if (!ReportCheck("Reference check (CUTLASS vs FP64 golden)", kh_compare(cmp, out_count))) {
    std::cerr << "CUTLASS reference incorrect vs FP64 golden model." << std::endl;
    return false;
  }

#ifdef KH_TEST_SOLUTION
  // Poisons the solution's output buffer (all bits set = FP16 NaN): unwritten elements fail.
  auto prepare_output = [&]() { cudaMemset(d_sol.get(), 0xFF, out_count * sizeof(cutlass::half_t)); };
  auto check_solution = [&](char const *what) {
    KhHalfOutCompare c{d_sol.get(), d_golden.get(), d_abs.get(), kTolScale, kOutputRoundTol};
    return ReportCheck(what, kh_compare(c, out_count));
  };
  auto call_solution = [&]() {
    return TransposedConv2d(
      reinterpret_cast<__half const *>(d_x.get()), reinterpret_cast<__half const *>(d_w.get()),
      reinterpret_cast<__half *>(d_sol.get()), N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  };

  {
    prepare_output();
    cudaDeviceSynchronize();
    cudaError_t result = call_solution();
    cudaError_t sync = cudaDeviceSynchronize();
    if (result != cudaSuccess || sync != cudaSuccess) {
      std::cerr << "Solution kernel failed: "
                << cudaGetErrorString(result != cudaSuccess ? result : sync) << std::endl;
      return false;
    }
    if (!check_solution("Solution check")) {
      std::cerr << "Solution incorrect." << std::endl;
      return false;
    }
  }
#endif

  std::cout << "Passed" << std::endl;
  if (iterations <= 0) return true;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Profile reference (kernel launch only; the CUTLASS operator is initialized once above).
  for (int i = 0; i < 3; ++i) ref.run();
  cudaDeviceSynchronize();
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    ref.run();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ref_total_ms += ms;
    if (ms < ref_min_ms) ref_min_ms = ms;
  }
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms) [config %s]\n",
          ref_total_ms / iterations, iterations, ref_min_ms, ref.name.c_str());

#ifdef KH_TEST_SOLUTION
  {
  // Warmup solution
  for (int i = 0; i < 3; i++) {
    prepare_output();
    call_solution();
  }
  cudaDeviceSynchronize();

  // Profile solution; every timed call is verified as well (untimed).
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    prepare_output();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    call_solution();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sol_total_ms += ms;
    if (ms < sol_min_ms) sol_min_ms = ms;
    KhHalfOutCompare c{d_sol.get(), d_golden.get(), d_abs.get(), kTolScale, kOutputRoundTol};
    KhCheckResult r = kh_compare(c, out_count);
    if (!r.ok()) {
      std::cerr << "Solution incorrect on timed call " << iter << " (max err/tol = " << r.max_ratio << ")." << std::endl;
      return false;
    }
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol_total_ms / iterations, iterations, sol_min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
  }
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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

  bool explicit_size = (argc >= 6);
  std::vector<TestConfig> configs;
  int iterations = 20;

  if (explicit_size) {
    // ./test N H W C K [R S stride pad iterations]   (defaults: 4x4 filter, stride 2, pad 1)
    int N = 1, H = 32, W = 32, C = 64, K = 64, R = 4, S = 4, stride = 2, pad = 1;
    std::stringstream(arg[1]) >> N;
    std::stringstream(arg[2]) >> H;
    std::stringstream(arg[3]) >> W;
    std::stringstream(arg[4]) >> C;
    std::stringstream(arg[5]) >> K;
    if (argc > 6) std::stringstream(arg[6]) >> R;
    if (argc > 7) std::stringstream(arg[7]) >> S;
    if (argc > 8) std::stringstream(arg[8]) >> stride;
    if (argc > 9) std::stringstream(arg[9]) >> pad;
    if (argc > 10) std::stringstream(arg[10]) >> iterations;
    configs.push_back({"custom", N, H, W, C, K, R, S, pad, pad, stride, stride});
  } else {
    // Transposed conv: 4x4 filter, stride 2, pad 1 = exact 2x upsampling (U-Net / GAN generator /
    // VAE-decoder style upsampling). H, W, C describe the input; K is the number of output channels.
    configs = {
      {"unet_up_32to64",      32,  32,  32, 512, 256, 4, 4, 1, 1, 2, 2},
      {"gan_up_64to128",      16,  64,  64, 256, 128, 4, 4, 1, 1, 2, 2},
      {"decoder_up_128to256",  8, 128, 128, 128,  64, 4, 4, 1, 1, 2, 2},
      {"decoder_up_256to512",  4, 256, 256,  64,  32, 4, 4, 1, 1, 2, 2},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: N=%d, H=%d, W=%d, C=%d, K=%d, R=%d, S=%d, pad=%d, stride=%d, H_out=%d, W_out=%d ===\n",
            cfg.label, cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S, cfg.pad_h, cfg.stride_h,
            (cfg.H - 1) * cfg.stride_h - 2 * cfg.pad_h + cfg.R,
            (cfg.W - 1) * cfg.stride_w - 2 * cfg.pad_w + cfg.S);
    if (!RunSize(cfg, iterations)) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
