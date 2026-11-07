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

This example shows how to fuse per channel scale+bias+relu of the activations
into the fprop mainloop.

Compared with original fprop kernel, this example has two more vectors, one for
the scale and one for the bias.  The length of the vectors are the same as the
activation channel number.  This kernels loads the vectors when the associated
activation channels are loaded in the mainloop.  Between reading the
activations and scale/bias data from the shared memory and calling tensor core
instructions, scale+bias+relu is computed in the register file.

This example is customized for Ampere 16816 fp16 tensor core instruction.
Changing to different data types or different tensor core instruction require
source code changing.  See
include/cutlass/conv/threadblock/implicit_gemm_fprop_fusion_multistage.h for more
technical details.

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
#include "cutlass/conv/kernel/default_conv2d_fprop_fusion.h"
#include "cutlass/conv/device/implicit_gemm_convolution_fusion.h"
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

/// FP16 activation after the fused per-channel transform: relu(scale * x + bias), computed exactly
/// and rounded once to FP16 (the precision of the tensor-core operand).
__device__ inline double kh_transform(double x, double scale, double bias) {
  return fmax(0.0, double(__half2float(__double2half(scale * x + bias))));
}

/// Golden problem: implicit GEMM view of the fused fprop (M = N*P*Q, N = K, K-dim = R*S*C).
/// Zero padding applies to the transformed activations.
struct KhFpropFusionProblem {
  cutlass::half_t const *x, *w, *scale, *bias;
  int N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w;
  int splits, ncols;
  __device__ int rows(int) const { return N * P * Q; }
  __device__ int depth(int) const { return R * S * C; }
  __device__ double a(int, int m, int k) const {
    int c = k % C, rs = k / C, s = rs % S, r = rs / S;
    int q = m % Q, t = m / Q, p = t % P, n = t / P;
    int h = p * stride_h + r - pad_h, wi = q * stride_w + s - pad_w;
    if (h < 0 || h >= H || wi < 0 || wi >= W) return 0.0;
    return kh_transform(double(float(x[((size_t(n) * H + h) * W + wi) * C + c])),
                        double(float(scale[c])), double(float(bias[c])));
  }
  __device__ double b(int, int k, int n) const { return double(float(w[size_t(n) * R * S * C + k])); }
  __device__ size_t out(int, int m, int n) const { return size_t(m) * K + n; }
};

/// expected = conv(transformed, w);  tol = kTolScale * sum|t*w| + kTransformTol * sqrt(sum (t*w)^2)
struct KhFusionCompare {
  float const *got; double const *G; float const *Aabs; float const *Asq;
  double tol_scale, transform_tol;
  __device__ void operator()(size_t i, double &g, double &e, double &tol) const {
    g = got[i];
    e = G[i];
    tol = tol_scale * Aabs[i] + transform_tol * sqrt(double(Asq[i]));
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

using ElementInputScaleBias = cutlass::half_t;     // Data type of elements in input scale and bias vectors
using LayoutInputScaleBias = cutlass::layout::RowMajor;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized memory access.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

/// CUTLASS Ampere tensor-core fprop with the scale+bias+relu of the activations fused into the
/// mainloop (F16 x F16 -> F32 accumulate -> F32), for a given tile configuration.
template <int TbM, int TbN, int TbK, int WM, int WN, int WK, int Stages>
struct FpropConfig {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropFusion<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementInputScaleBias, LayoutInputScaleBias,
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
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolutionFusion<Kernel>;
};

// Tile configurations; the per-size choice comes from an offline sweep on the target GPU.
using Fprop_128x128x64_s3 = FpropConfig<128, 128, 64, 64, 64, 64, 3>;   // also the default (custom sizes)
using Fprop_128x64x32_s4  = FpropConfig<128,  64, 32, 64, 32, 32, 4>;
using Fprop_128x64x64_s3  = FpropConfig<128,  64, 64, 64, 32, 64, 3>;

/// A prepared (initialized) reference convolution; run() only launches the kernel.
struct RefConv {
  std::function<cutlass::Status()> run;
  std::string name;
};

template <typename Op>
cutlass::Status MakeRefConv(RefConv &rc, char const *name,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    cutlass::half_t const *input, cutlass::half_t const *filter,
    cutlass::half_t const *scale, cutlass::half_t const *bias, float *D,
    cutlass::Tensor4DCoord input_size, cutlass::Tensor4DCoord filter_size, cutlass::Tensor4DCoord output_size) {
  int C = input_size.c();
  typename Op::Arguments arguments{
    problem_size,
    {const_cast<cutlass::half_t *>(input), LayoutInputA::packed(input_size)},
    {const_cast<cutlass::half_t *>(filter), LayoutInputB::packed(filter_size)},
    {const_cast<cutlass::half_t *>(scale), LayoutInputScaleBias(C)},
    {const_cast<cutlass::half_t *>(bias), LayoutInputScaleBias(C)},
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

struct TestConfig {
  const char* label;
  int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
};

/// Per-size reference kernel selection (benchmark sizes); anything else uses the default tile.
cutlass::Status PrepareReference(RefConv &rc, TestConfig const &cfg,
    cutlass::conv::Conv2dProblemSize const &ps, cutlass::half_t const *x, cutlass::half_t const *w,
    cutlass::half_t const *scale, cutlass::half_t const *bias, float *D,
    cutlass::Tensor4DCoord is, cutlass::Tensor4DCoord fs, cutlass::Tensor4DCoord os) {
  std::string l = cfg.label;
  if (l == "resnet50_conv2_b256")
    return MakeRefConv<Fprop_128x64x32_s4::Op>(rc, "128x64x32_s4", ps, x, w, scale, bias, D, is, fs, os);
  if (l == "sdxl_unet_128")
    return MakeRefConv<Fprop_128x64x64_s3::Op>(rc, "128x64x64_s3", ps, x, w, scale, bias, D, is, fs, os);
  return MakeRefConv<Fprop_128x128x64_s3::Op>(rc, "128x128x64_s3", ps, x, w, scale, bias, D, is, fs, os);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// The solution is included after all reference / verification code so that its macros cannot
// change the harness.
#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Tolerance: |out - expected| <= kTolScale * sum|t*w| + kTransformTol * sqrt(sum (t*w)^2), where
// `expected` is the FP64 golden result with t = relu(scale*x + bias) rounded once to FP16.
// The first term covers FP32 accumulation; the second covers a different (or no) rounding of the
// transformed activation t: an independent relative error of up to 2^-11 per term has an RMS of
// 2^-11/sqrt(3) * sqrt(sum (t*w)^2), and the bound allows 7x that. FP16 accumulation and
// BF16/FP8/INT8 operands exceed it.
constexpr double kTolScale = 5.0e-6;
constexpr double kTransformTol = 7.0 * 4.8828125e-4 / 1.7320508075688772;

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

  cutlass::device_memory::allocation<cutlass::half_t> d_x(in_count), d_w(f_count), d_scale(C), d_bias(C);
  cutlass::device_memory::allocation<float> d_ref(out_count), d_sol(out_count);
  cutlass::device_memory::allocation<double> d_golden(out_count);
  cutlass::device_memory::allocation<float> d_abs(out_count), d_sq(out_count);

  // Random inputs with full FP16 mantissa: activations and filter uniform in [-1, 1),
  // per-channel scale uniform in [0.25, 2), bias uniform in [-1, 1).
  kh_fill_uniform(d_x.get(), in_count, 11u, -1.0, 1.0);
  kh_fill_uniform(d_w.get(), f_count, 23u, -1.0, 1.0);
  kh_fill_uniform(d_scale.get(), size_t(C), 37u, 0.25, 2.0);
  kh_fill_uniform(d_bias.get(), size_t(C), 41u, -1.0, 1.0);

  // FP64 golden model on the device.
  KhFpropFusionProblem pb{d_x.get(), d_w.get(), d_scale.get(), d_bias.get(),
                          N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, 1, K};
  kh_golden(pb, N * P * Q, 1, d_golden.get(), d_abs.get(), d_sq.get(), out_count);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Golden model failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Reference (CUTLASS).
  RefConv ref;
  cutlass::Status st = PrepareReference(ref, cfg, problem_size, d_x.get(), d_w.get(), d_scale.get(), d_bias.get(),
                                        d_ref.get(), input_size, filter_size, output_size);
  if (st != cutlass::Status::kSuccess || ref.run() != cutlass::Status::kSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "CUTLASS reference failed: " << cutlassGetStatusString(st) << std::endl;
    return false;
  }
  KhFusionCompare cmp{d_ref.get(), d_golden.get(), d_abs.get(), d_sq.get(), kTolScale, kTransformTol};
  if (!ReportCheck("Reference check (CUTLASS vs FP64 golden)", kh_compare(cmp, out_count))) {
    std::cerr << "CUTLASS reference incorrect vs FP64 golden model." << std::endl;
    return false;
  }

#ifdef KH_TEST_SOLUTION
  // Poisons the solution's output buffer (all bits set = NaN): unwritten elements fail the check.
  auto prepare_output = [&]() { cudaMemset(d_sol.get(), 0xFF, out_count * sizeof(float)); };
  auto check_solution = [&](char const *what) {
    KhFusionCompare c{d_sol.get(), d_golden.get(), d_abs.get(), d_sq.get(), kTolScale, kTransformTol};
    return ReportCheck(what, kh_compare(c, out_count));
  };
  auto call_solution = [&]() {
    return Conv2dFpropScaleBias(
      reinterpret_cast<__half const *>(d_x.get()), reinterpret_cast<__half const *>(d_w.get()),
      reinterpret_cast<__half const *>(d_scale.get()), reinterpret_cast<__half const *>(d_bias.get()),
      d_sol.get(), N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
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
    KhFusionCompare c{d_sol.get(), d_golden.get(), d_abs.get(), d_sq.get(), kTolScale, kTransformTol};
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

  bool explicit_size = (argc >= 8);
  std::vector<TestConfig> configs;
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
    if (argc > 8) std::stringstream(arg[8]) >> iterations;
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
    if (!RunSize(cfg, iterations)) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
