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
  This example is almost the same as example 27 which uses 3xTF32 to run GEMM.  The only
  difference is that this example uses 3xtf32 on complex gemm.

  To enable this feature, the only change needs to make is to change OpMultiplyAddComplex
  to OpMultiplyAddComplexFastF32.
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS complex GEMM with 3xTF32 tensor-core math (OpMultiplyAddComplexFastF32,
// FP32-level accuracy), A column-major, B/C/D row-major, with a per-size configuration chosen by
// an offline sweep on the benchmark GPU (RTX PRO 6000 Blackwell, sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using Cf = cutlass::complex<float>;
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Cf, 1, Cf, Cf>;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages>
using Gemm3xTF32T = cutlass::gemm::device::GemmComplex<
    Cf, LayoutInputA, Cf, LayoutInputB, Cf, LayoutOutput, Cf,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 8>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
    cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddComplexFastF32,
    true /*SplitKSerial*/>;

// Per-size configurations (offline sweep):
//   0: 64x64x16 threadblock, 32x32x16 warp, 3 stages               (lu_trailing_update, subspace_proj)
//   1: 64x128x16 threadblock, 32x32x16 warp, 3 stages, serial split-K (mimo_beamform; correlator
//      with 4 K-slices: only 256 output tiles for 188 SMs)
using Gemm3xTF32_0 = Gemm3xTF32T<64, 64, 16, 32, 32, 16, 3>;
using Gemm3xTF32_1 = Gemm3xTF32T<64, 128, 16, 32, 32, 16, 3>;

struct RefPlan { int config; int split_k; };

static RefPlan RefPlanFor(int M, int N, int K) {
  if (M == 512 && N == 131072 && K == 512)    return {1, 1};   // mimo_beamform
  if (M == 1024 && N == 1024 && K == 131072)  return {1, 4};   // correlator
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  long long tiles = (long long)((M + 63) / 64) * ((N + 63) / 64);
  int split = int(std::min<long long>(4, std::max<long long>(1, (2LL * sms + tiles - 1) / tiles)));
  return {0, std::max(1, std::min(split, K / 1024))};
}


/// Persistent workspace for the split-K semaphores (allocated once, outside the timed region).
static void *g_ref_workspace = nullptr;
static size_t g_ref_workspace_bytes = 0;

template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, Cf alpha, float const *A, int lda,
                              float const *B, int ldb, Cf beta, float const *C, int ldc,
                              float *D, int ldd, int split_k) {
  typename Gemm::Arguments args(
      {M, N, K},
      {reinterpret_cast<Cf const *>(A), lda}, {reinterpret_cast<Cf const *>(B), ldb},
      {reinterpret_cast<Cf const *>(C), ldc}, {reinterpret_cast<Cf *>(D), ldd},
      {alpha, beta}, split_k);
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_ref_workspace_bytes) {
    if (g_ref_workspace) cudaFree(g_ref_workspace);
    if (cudaMalloc(&g_ref_workspace, ws) != cudaSuccess) return cudaErrorMemoryAllocation;
    g_ref_workspace_bytes = ws;
  }
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (gemm_op.initialize(args, g_ref_workspace) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm_op() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaGetLastError();
}

/// Reference 3xTF32 complex GEMM: D = alpha * A * B + beta * C (interleaved (re, im) floats).
cudaError_t Cutlass3xTF32ComplexGemm(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const *A, int lda,
    float const *B, int ldb,
    float beta_real, float beta_imag,
    float const *C, int ldc,
    float *D, int ldd) {
  Cf alpha(alpha_real, alpha_imag), beta(beta_real, beta_imag);
  RefPlan p = RefPlanFor(M, N, K);
  if (p.config == 1)
    return RunCutlass<Gemm3xTF32_1>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, p.split_k);
  return RunCutlass<Gemm3xTF32_0>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, p.split_k);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Verification kit: random full-mantissa inputs, golden GEMM on the device (FP64, or FP32 for
// 16-bit inputs whose products are exact in FP32), NaN-safe comparison of every output element
// (max error and relative Frobenius error vs the golden).
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define KH_CHECK(call)                                                                         \
  do {                                                                                         \
    cudaError_t kh_err_ = (call);                                                              \
    if (kh_err_ != cudaSuccess) {                                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(kh_err_) << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                                      \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

__host__ __device__ inline unsigned long long kh_hash64(unsigned long long i, unsigned int seed) {
  unsigned long long x = i * 0x9E3779B97F4A7C15ull + (seed + 1ull) * 0xD1B54A32D192ED03ull;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 33;
  return x;
}

/// Uniform in [-1, 1): a random 25-bit signed integer times 2^-24. Every value with |v| >= 0.5
/// carries a full 24-bit significand, so rounding inputs to TF32/BF16/FP16 changes the result.
__host__ __device__ inline float kh_uniform(unsigned long long i, unsigned int seed) {
  return float(int(unsigned(kh_hash64(i, seed) >> 32)) >> 7) * (1.0f / 16777216.0f);
}

__device__ inline double kh_to_double(float x) { return double(x); }
__device__ inline double kh_to_double(__half x) { return double(__half2float(x)); }
__device__ inline double kh_to_double(__nv_bfloat16 x) { return double(__bfloat162float(x)); }
template <typename T> __device__ inline T kh_from_float(float x);
template <> __device__ inline float kh_from_float<float>(float x) { return x; }
template <> __device__ inline __half kh_from_float<__half>(float x) { return __float2half_rn(x); }
template <> __device__ inline __nv_bfloat16 kh_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

/// Random values: uniform in [-scale, scale) rounded to T (a full T significand).
template <typename T>
__global__ void kh_fill_kernel(T *p, size_t n, unsigned int seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_from_float<T>(kh_uniform(i, seed) * scale);
}

template <typename T>
__global__ void kh_fill_value_kernel(T *p, size_t n, float v) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_from_float<T>(v);
}

template <typename T>
static void kh_fill(T *p, size_t n, unsigned int seed, float scale = 1.0f) {
  kh_fill_kernel<T><<<4096, 256>>>(p, n, seed, scale);
  KH_CHECK(cudaGetLastError());
}

/// Finite poison for outputs that the operation may also read with beta = 0 (0 * poison = 0):
/// any element the solution leaves unwritten keeps this value and fails the comparison
/// (65504 = largest finite FP16 value; 3e38 for FP32 outputs).
static const float kPoison16 = 65504.0f;
static const float kPoison32 = 3.0e38f;

template <typename T>
static void kh_fill_value(T *p, size_t n, float v) {
  kh_fill_value_kernel<T><<<4096, 256>>>(p, n, v);
  KH_CHECK(cudaGetLastError());
}

/// Golden GEMM: G(i,j) = alpha * sum_k A(i,k) B(k,j) + beta * C(i,j), G row-major M x N (FP64),
/// accumulated in Acc (double, or float for 16-bit inputs whose products are exact in FP32).
/// Element (i,k) of A is A[i*sam + k*sak]; B(k,j) = B[k*sbk + j*sbn]; C(i,j) = C[i*scm + j*scn].
template <typename Acc, typename TA, typename TB, typename TC>
__global__ void __launch_bounds__(256) kh_golden_gemm_kernel(
    int M, int N, int K, double alpha,
    TA const *A, long long sam, long long sak,
    TB const *B, long long sbk, long long sbn,
    double beta, TC const *C, long long scm, long long scn,
    double *G) {
  __shared__ Acc As[16][65];
  __shared__ Acc Bs[16][65];
  int const tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  int const m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  Acc acc[4][4];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) acc[r][c] = Acc(0);
  bool const a_m_fast = (sam == 1);
  bool const b_n_fast = (sbn == 1);
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int e = threadIdx.x; e < 16 * 64; e += 256) {
      int kk = a_m_fast ? e / 64 : e % 16;
      int mm = a_m_fast ? e % 64 : e / 16;
      int gi = m0 + mm, gk = k0 + kk;
      As[kk][mm] = (gi < M && gk < K) ? Acc(kh_to_double(A[gi * sam + gk * sak])) : Acc(0);
      int kb = b_n_fast ? e / 64 : e % 16;
      int nn = b_n_fast ? e % 64 : e / 16;
      int gj = n0 + nn, gkb = k0 + kb;
      Bs[kb][nn] = (gj < N && gkb < K) ? Acc(kh_to_double(B[gkb * sbk + gj * sbn])) : Acc(0);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      Acc a[4], b[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) a[r] = As[kk][ty + 16 * r];
#pragma unroll
      for (int c = 0; c < 4; ++c) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
    }
    __syncthreads();
  }
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      int i = m0 + ty + 16 * r, j = n0 + tx + 16 * c;
      if (i < M && j < N) {
        double v = alpha * double(acc[r][c]);
        if (beta != 0.0) v += beta * kh_to_double(C[i * scm + j * scn]);
        G[size_t(i) * N + j] = v;
      }
    }
}

template <typename Acc, typename TA, typename TB, typename TC>
static void kh_golden_gemm(int M, int N, int K, double alpha,
                           TA const *A, long long sam, long long sak,
                           TB const *B, long long sbk, long long sbn,
                           double beta, TC const *C, long long scm, long long scn, double *G) {
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  kh_golden_gemm_kernel<Acc, TA, TB, TC><<<grid, 256>>>(M, N, K, alpha, A, sam, sak, B, sbk, sbn,
                                                       beta, C, scm, scn, G);
  KH_CHECK(cudaGetLastError());
}

struct KhStats {
  double sum_g2, sum_d2;
  unsigned long long max_d_bits, nonfinite, nbad;
};

__device__ inline void kh_block_add(double v, double *dst) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(dst, v);
}

__global__ void kh_sumsq_kernel(double const *G, size_t n, KhStats *st) {
  double s = 0.0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    s += G[i] * G[i];
  kh_block_add(s, &st->sum_g2);
}

/// Compares X(i,j) = X[i*sxm + j*sxn] with G (row-major M x N). Counts non-finite outputs and
/// elements with !(|x - g| <= tol_abs) (NaN-safe), accumulates sum (x-g)^2 and max |x-g|.
template <typename TX>
__global__ void kh_compare_kernel(TX const *X, long long sxm, long long sxn, double const *G,
                                  int M, int N, double tol_abs, double tol_rel, KhStats *st) {
  size_t const n = size_t(M) * N;
  double s = 0.0, mx = 0.0;
  unsigned long long nf = 0, nb = 0;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < n;
       idx += size_t(gridDim.x) * blockDim.x) {
    long long i = (long long)(idx / N), j = (long long)(idx % N);
    double x = kh_to_double(X[i * sxm + j * sxn]);
    double d = fabs(x - G[idx]);
    if (!isfinite(x)) { ++nf; continue; }
    if (!(d <= tol_abs + tol_rel * fabs(G[idx]))) ++nb;
    s += d * d;
    mx = fmax(mx, d);
  }
  kh_block_add(s, &st->sum_d2);
  for (int o = 16; o > 0; o >>= 1) {
    mx = fmax(mx, __shfl_down_sync(0xffffffffu, mx, o));
    nf += __shfl_down_sync(0xffffffffu, nf, o);
    nb += __shfl_down_sync(0xffffffffu, nb, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(&st->max_d_bits, (unsigned long long)__double_as_longlong(mx));
    atomicAdd(&st->nonfinite, nf);
    atomicAdd(&st->nbad, nb);
  }
}

struct KhResult {
  double rms, max_err_rel, fro_rel;
  unsigned long long nonfinite, nbad;
  bool pass;
};

/// Full check of an M x N output against the golden: every element must be finite and within
/// tol_max * RMS(golden) (+ tol_rel * |golden|) of the golden, and the relative Frobenius error
/// must be <= tol_fro.
template <typename TX>
static KhResult kh_check(TX const *X, long long sxm, long long sxn, double const *G,
                         int M, int N, double tol_max, double tol_fro, double tol_rel = 0.0) {
  KhStats *st;
  KH_CHECK(cudaMalloc(&st, sizeof(KhStats)));
  KH_CHECK(cudaMemset(st, 0, sizeof(KhStats)));
  size_t n = size_t(M) * N;
  kh_sumsq_kernel<<<2048, 256>>>(G, n, st);
  KH_CHECK(cudaGetLastError());
  KhStats h;
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  double rms = std::sqrt(h.sum_g2 / double(n));
  kh_compare_kernel<TX><<<2048, 256>>>(X, sxm, sxn, G, M, N, tol_max * rms, tol_rel, st);
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  KH_CHECK(cudaFree(st));
  double max_d;
  std::memcpy(&max_d, &h.max_d_bits, sizeof(max_d));
  KhResult r;
  r.rms = rms;
  r.max_err_rel = rms > 0 ? max_d / rms : INFINITY;
  r.fro_rel = h.sum_g2 > 0 ? std::sqrt(h.sum_d2 / h.sum_g2) : INFINITY;
  r.nonfinite = h.nonfinite;
  r.nbad = h.nbad;
  r.pass = (rms > 0) && h.nonfinite == 0 && h.nbad == 0 && r.fro_rel <= tol_fro;
  return r;
}

static void kh_print(char const *who, KhResult const &r, double tol_max, double tol_fro) {
  std::printf("  %s: max|err|/rms = %.3e (tol %.1e), rel_fro_err = %.3e (tol %.1e), "
              "non-finite = %llu, elements over tol = %llu -> %s\n",
              who, r.max_err_rel, tol_max, r.fro_rel, tol_fro, r.nonfinite, r.nbad,
              r.pass ? "ok" : "FAIL");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(status) KH_CHECK(status)

/// Golden complex GEMM in FP64, written as an M x 2N row-major array of doubles (re, im
/// interleaved): A column-major (lda), B/C row-major (ldb/ldc), units of complex elements.
__global__ void GoldenComplexGemm_kernel(int M, int N, int K, double alpha_re, double alpha_im,
                                         float const *A, int lda, float const *B, int ldb,
                                         double beta_re, double beta_im, float const *C, int ldc,
                                         double *G) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= M || j >= N) return;
  double re = 0.0, im = 0.0;
  for (int k = 0; k < K; ++k) {
    float2 a = *reinterpret_cast<float2 const *>(A + (size_t(i) + size_t(k) * lda) * 2);
    float2 b = *reinterpret_cast<float2 const *>(B + (size_t(k) * ldb + j) * 2);
    re += double(a.x) * double(b.x) - double(a.y) * double(b.y);
    im += double(a.x) * double(b.y) + double(a.y) * double(b.x);
  }
  double r_re = alpha_re * re - alpha_im * im;
  double r_im = alpha_re * im + alpha_im * re;
  if (beta_re != 0.0 || beta_im != 0.0) {
    float2 c = *reinterpret_cast<float2 const *>(C + (size_t(i) * ldc + j) * 2);
    r_re += beta_re * double(c.x) - beta_im * double(c.y);
    r_im += beta_re * double(c.y) + beta_im * double(c.x);
  }
  G[(size_t(i) * N + j) * 2] = r_re;
  G[(size_t(i) * N + j) * 2 + 1] = r_im;
}

// FP32-level (3xTF32) tolerances, relative to the RMS of the exact (FP64) result and scaled with
// sqrt(2K) (each real output component accumulates 2K real products) like FP32 accumulation
// error. A correct FP32 / 3xTF32 result stays >= 4x below both bounds, while single-pass TF32
// (relative Frobenius error >= 2.6e-4) or BF16/FP16 operands exceed the Frobenius bound >= 4x.
static double TolMax(int K) { return 1.5e-6 * std::sqrt(2.0 * K); }
static double TolFro(int K) { return 1.2e-7 * std::sqrt(2.0 * K); }

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against an FP64
/// golden result computed on the device. Every real and imaginary output component is checked.
cudaError_t TestCorrectness(int M, int N, int K,
    float alpha_real, float alpha_imag,
    float beta_real, float beta_imag) {
  int lda = M, ldb = N, ldc = N, ldd = N;
  size_t count_A = size_t(lda) * K * 2;
  size_t count_B = size_t(K) * ldb * 2;
  size_t count_C = size_t(M) * ldc * 2;
  double const tol_max = TolMax(K), tol_fro = TolFro(K);
  Cf alpha(alpha_real, alpha_imag), beta(beta_real, beta_imag);

  float *d_A, *d_B, *d_C, *d_D_ref;
  double *G;
  CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D_ref, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&G, size_t(M) * N * 2 * sizeof(double)));
  kh_fill(d_A, count_A, 1u);
  kh_fill(d_B, count_B, 2u);
  kh_fill(d_C, count_C, 3u);
  CUDA_CHECK(cudaMemset(d_D_ref, 0xFF, count_C * sizeof(float)));   // NaN poison

  {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    GoldenComplexGemm_kernel<<<grid, block>>>(M, N, K, alpha_real, alpha_imag, d_A, lda, d_B, ldb,
                                              beta_real, beta_imag, d_C, ldc, G);
    CUDA_CHECK(cudaGetLastError());
  }

  cudaError_t result = Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag, d_A, lda, d_B,
                                                ldb, beta_real, beta_imag, d_C, ldc, d_D_ref, ldd);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS 3xTF32 complex GEMM failed: " << cudaGetErrorString(result) << std::endl;
  } else {
    KhResult r = kh_check(d_D_ref, 2LL * ldd, 1, G, M, 2 * N, tol_max, tol_fro);
    kh_print("reference", r, tol_max, tol_fro);
    if (!r.pass) {
      std::cerr << "CUTLASS reference incorrect vs FP64 golden." << std::endl;
      result = cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    // Solution output, poisoned (0xFF bytes = NaN) so unwritten elements fail the check
    float *d_D_solution;
    CUDA_CHECK(cudaMalloc(&d_D_solution, count_C * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_D_solution, 0xFF, count_C * sizeof(float)));
    result = ComplexGemm(M, N, K, alpha_real, alpha_imag, d_A, lda, d_B, ldb,
                         beta_real, beta_imag, d_C, ldc, d_D_solution, ldd);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(d_D_solution, 2LL * ldd, 1, G, M, 2 * N, tol_max, tol_fro);
      kh_print("solution ", r, tol_max, tol_fro);
      if (!r.pass) {
        std::cerr << "Solution incorrect vs FP64 golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", components over tolerance = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
    cudaFree(d_D_solution);
  }
#endif

  cudaFree(G);
  cudaFree(d_D_ref);
  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K,
    float alpha_real, float alpha_imag,
    float beta_real, float beta_imag, int iterations) {
  int lda = M, ldb = N, ldc = N, ldd = N;
  size_t count_A = size_t(lda) * K * 2;
  size_t count_B = size_t(K) * ldb * 2;
  size_t count_C = size_t(M) * ldc * 2;

  float *d_A, *d_B, *d_C, *d_D;
  CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, count_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D, count_C * sizeof(float)));
  kh_fill(d_A, count_A, 1u);
  kh_fill(d_B, count_B, 2u);
  kh_fill(d_C, count_C, 3u);
  CUDA_CHECK(cudaMemset(d_D, 0, count_C * sizeof(float)));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++)
    Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Cutlass3xTF32ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ref_total_ms += ms;
    if (ms < ref_min_ms) ref_min_ms = ms;
  }
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref_total_ms / iterations, iterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
  // Warmup solution
  cudaMemset(d_D, 0, count_C * sizeof(float));
  for (int i = 0; i < 3; i++)
    ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    ComplexGemm(M, N, K, alpha_real, alpha_imag,
      d_A, lda, d_B, ldb, beta_real, beta_imag, d_C, ldc, d_D, ldd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sol_total_ms += ms;
    if (ms < sol_min_ms) sol_min_ms = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol_total_ms / iterations, iterations, sol_min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Check for SM >= 80
  bool notSupported = false;
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80." << std::endl;
    notSupported = true;
  }
  if (notSupported) return 0;

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha_real = 1.0f, alpha_imag = 0.0f;
  float beta_real = 0.0f, beta_imag = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 256, N = 256, K = 128;
    std::stringstream ss1(arg[1]); ss1 >> M;
    std::stringstream ss2(arg[2]); ss2 >> N;
    std::stringstream ss3(arg[3]); ss3 >> K;
    if (argc > 4) { std::stringstream ss(arg[4]); ss >> alpha_real; }
    if (argc > 5) { std::stringstream ss(arg[5]); ss >> alpha_imag; }
    if (argc > 6) { std::stringstream ss(arg[6]); ss >> beta_real; }
    if (argc > 7) { std::stringstream ss(arg[7]); ss >> beta_imag; }
    if (argc > 8) { std::stringstream ss(arg[8]); ss >> iterations; }
    configs.push_back({"custom", M, N, K});
  } else {
    // Complex-valued signal-processing / scientific-computing shapes:
    configs = {
      // massive-MIMO beamforming: 512 beams x 512-antenna array applied to a 128K-sample block
      {"mimo_beamform",        512, 131072,    512},
      // blocked complex LU (ZGETRF-style) trailing-matrix update, 16384^2 matrix, panel width 512
      {"lu_trailing_update", 16384,  16384,    512},
      // subspace projection: 16384^2 complex operator applied to a 512-vector block (block Krylov)
      {"subspace_proj",      16384,    512,  16384},
      // radio-astronomy correlator: 1024 inputs (512 dual-pol stations) integrated over 128K samples
      {"correlator",          1024,   1024, 131072},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k,
      alpha_real, alpha_imag, beta_real, beta_imag);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, alpha_real, alpha_imag, beta_real, beta_imag, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
