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

// Standard Library includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS TF32 tensor-core GEMM (float in/out, TF32 multiply, FP32 accumulate),
// A row-major, B column-major, C/D row-major, with a per-size configuration chosen by an
// offline sweep on the benchmark GPU (RTX PRO 6000 Blackwell, sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue>;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, int Swizzle = 1>
using Tf32Gemm_t = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 8>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages, 4, 4,
    true /*SplitKSerial*/>;

// Per-size configurations (offline sweep). All use >= 3 stages: CUTLASS 2.x silently runs
// 2-stage float tensor-op GEMMs with FP16 math (OpMultiplyAddFastF16), which is not TF32.
//   0: 128x128x16 threadblock, 64x64x16 warp, 4 stages            (decode_lm_head)
//   1: 128x256x16 threadblock, 64x64x16 warp, 3 stages, swizzle 2 (16K-token prefill shapes)
//   2: 256x128x16 threadblock, 64x64x16 warp, 3 stages            (prefill_mlp_down)
//   3:  64x64x32  threadblock, 32x32x32 warp, 5 stages, serial split-K (decode_qkv)
using Tf32Gemm0 = Tf32Gemm_t<128, 128, 16, 64, 64, 16, 4>;
using Tf32Gemm1 = Tf32Gemm_t<128, 256, 16, 64, 64, 16, 3, 2>;
using Tf32Gemm2 = Tf32Gemm_t<256, 128, 16, 64, 64, 16, 3>;
using Tf32Gemm3 = Tf32Gemm_t<64, 64, 32, 32, 32, 32, 5>;

struct RefPlan { int config; int split_k; };

static RefPlan RefPlanFor(int M, int N, int K) {
  if (M == 128 && N == 10240 && K == 5120)    return {3, 4};   // decode_qkv
  if (M == 128 && N == 151936 && K == 5120)   return {0, 1};   // decode_lm_head
  if (M == 16384 && N == 5120 && K == 25600)  return {2, 1};   // prefill_mlp_down
  if (M >= 4096) return {1, 1};                                 // other prefill shapes
  if (M <= 512) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    long long tiles = (long long)((M + 63) / 64) * ((N + 63) / 64);
    int split = int(std::min<long long>(8, std::max<long long>(1, (2LL * sms + tiles - 1) / tiles)));
    return {3, std::max(1, std::min(split, K / 256))};
  }
  return {0, 1};
}


/// Persistent workspace for the split-K semaphores (allocated once, outside the timed region).
static void *g_ref_workspace = nullptr;
static size_t g_ref_workspace_bytes = 0;

template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha, float const *A, int lda,
                              float const *B, int ldb, float beta, float *C, int ldc, int split_k) {
  typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
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

/// CUTLASS TF32 GEMM wrapper.
/// A row-major [M,K] (lda=K), B col-major [K,N] (ldb=K), C/D row-major [M,N] (ldc=N)
/// D = alpha * A * B + beta * C
cudaError_t CutlassTf32Gemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float *C, int ldc) {
  RefPlan p = RefPlanFor(M, N, K);
  switch (p.config) {
    case 1: return RunCutlass<Tf32Gemm1>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 1);
    case 2: return RunCutlass<Tf32Gemm2>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 1);
    case 3: return RunCutlass<Tf32Gemm3>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, p.split_k);
    default: return RunCutlass<Tf32Gemm0>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 1);
  }
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

// TF32 tolerances, relative to the RMS of the exact (FP64) result. Rounding both operands to
// TF32 gives a relative Frobenius error of 2.6e-4 (round-to-nearest, the reference: 4.6x below
// the bound) to 6.9e-4 (truncation, 1.7x below); BF16 operands give 2.1e-3 and fail. A has
// entries beyond the FP16 range (|A| up to 2^17) and B entries below its normal range
// (|B| < 2^-17), so FP16 operands overflow/underflow and fail as well.
static double TolMax(int) { return 7e-3; }
static double TolFro(int) { return 1.2e-3; }
static const float kScaleA = 131072.0f;          // 2^17
static const float kScaleB = 1.0f / 131072.0f;   // 2^-17

/// Row-major M x N buffer filled with the initial C: random data when beta != 0, otherwise a
/// finite poison value (C is not needed when beta == 0, and unwritten elements must fail).
static void InitC(float *C, size_t count, float beta) {
  if (beta != 0.0f) kh_fill(C, count, 3u);
  else kh_fill_value(C, count, kPoison32);
}

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against an FP64
/// golden result computed on the device. Every output element is checked.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta,
                            bool skip_reference_verification) {
  // A row-major [M,K] (lda = K), B col-major [K,N] (ldb = K), C row-major [M,N] (ldc = N)
  int lda = K, ldb = K, ldc = N;
  size_t count_A = size_t(M) * lda, count_B = size_t(N) * ldb, count_C = size_t(M) * ldc;
  double const kTolMax = TolMax(K), kTolFro = TolFro(K);

  float *A, *B, *C_ref;
  double *G;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(float)));
  KH_CHECK(cudaMalloc(&C_ref, count_C * sizeof(float)));
  KH_CHECK(cudaMalloc(&G, size_t(M) * N * sizeof(double)));
  kh_fill(A, count_A, 1u, kScaleA);
  kh_fill(B, count_B, 2u, kScaleB);
  InitC(C_ref, count_C, beta);

  // Golden (FP64 accumulation) from the initial C, before anything overwrites it.
  kh_golden_gemm<double>(M, N, K, alpha, A, lda, 1, B, 1, ldb, beta, C_ref, ldc, 1, G);

  cudaError_t result = cudaSuccess;
#ifdef KH_TEST_SOLUTION
  float *C_sol;
  KH_CHECK(cudaMalloc(&C_sol, count_C * sizeof(float)));
  KH_CHECK(cudaMemcpy(C_sol, C_ref, count_C * sizeof(float), cudaMemcpyDeviceToDevice));
#endif

  // Reference
  result = CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
  } else if (!skip_reference_verification) {
    KhResult r = kh_check(C_ref, ldc, 1, G, M, N, kTolMax, kTolFro);
    kh_print("reference", r, kTolMax, kTolFro);
    if (!r.pass) {
      std::cerr << "CUTLASS reference incorrect vs FP64 golden." << std::endl;
      result = cudaErrorUnknown;
    }
  } else {
    std::cout << "Baseline reference verification skipped." << std::endl;
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    result = Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C_sol, ldc);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(C_sol, ldc, 1, G, M, N, kTolMax, kTolFro);
      kh_print("solution ", r, kTolMax, kTolFro);
      if (!r.pass) {
        std::cerr << "Solution incorrect vs FP64 golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", elements over tolerance = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
  }
  cudaFree(C_sol);
#endif

  cudaFree(G);
  cudaFree(C_ref);
  cudaFree(B);
  cudaFree(A);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = K, ldc = N;
  size_t count_A = size_t(M) * lda, count_B = size_t(N) * ldb, count_C = size_t(M) * ldc;

  float *A, *B, *C;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(float)));
  KH_CHECK(cudaMalloc(&C, count_C * sizeof(float)));
  kh_fill(A, count_A, 1u, kScaleA);
  kh_fill(B, count_B, 2u, kScaleB);
  kh_fill(C, count_C, 3u);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassTf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
  kh_fill(C, count_C, 3u);
  for (int i = 0; i < 3; i++) {
    Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    Tf32Gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k;
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

  // Check for explicit size on command line
  bool explicit_size = (positional.size() >= 3);

  if (explicit_size) {
    int M = 128, N = 128, K = 128;
    std::stringstream(positional[0]) >> M;
    std::stringstream(positional[1]) >> N;
    std::stringstream(positional[2]) >> K;
    if (positional.size() > 3) std::stringstream(positional[3]) >> alpha;
    if (positional.size() > 4) std::stringstream(positional[4]) >> beta;
    if (positional.size() > 5) std::stringstream(positional[5]) >> iterations;
    configs.push_back({"custom", M, N, K});
  } else {
    // Qwen3-32B linear layers on one GPU (hidden 5120, 64 q / 8 kv heads x 128 -> fused QKV
    // 10240, FFN 25600, vocab 151936); M = tokens: 16K-token prefill chunk, 128-sequence decode.
    configs = {
      {"decode_qkv",        128,  10240,  5120},   // fused QKV proj, 128 decode tokens
      {"prefill_qkv",     16384,  10240,  5120},   // fused QKV proj, 16K-token prefill chunk
      {"prefill_o_proj",  16384,   5120,  8192},   // attention output proj (64 heads x 128)
      {"prefill_mlp_up",  16384,  51200,  5120},   // fused gate+up proj (2 x 25600)
      {"prefill_mlp_down",16384,   5120, 25600},   // down proj
      {"decode_lm_head",    128, 151936,  5120},   // LM head, 128 decode tokens
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    cudaError_t result = TestCorrectness(
      cfg.m, cfg.n, cfg.k, alpha, beta, skip_reference_verification);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, alpha, beta, iterations);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
