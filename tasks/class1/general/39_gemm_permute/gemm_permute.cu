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

// GEMM Permute example — half precision, row-major
// Based on CUTLASS example 39_gemm_permute, restructured for CUDA-Hercules testing.
// D = alpha * A * B + beta * D  (row-major, FP16 with FP32 accumulation)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_fp16.h>

#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference: CUTLASS FP16 tensor-core GEMM (FP32 accumulation, FP16 output), all row-major,
// with a per-size configuration chosen by an offline sweep on the benchmark GPU
// (RTX PRO 6000 Blackwell, sm_120).
///////////////////////////////////////////////////////////////////////////////////////////////////

using RowMajor = cutlass::layout::RowMajor;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, int Swizzle = 1>
using HgemmT = cutlass::gemm::device::Gemm<
    cutlass::half_t, RowMajor,    // A
    cutlass::half_t, RowMajor,    // B
    cutlass::half_t, RowMajor,    // C/D
    float,                        // ElementAccumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages, 8, 8>;

// Per-size configurations (offline sweep). No serial split-K: it would round the partial sums to
// FP16 in D, i.e. not accumulate in FP32.
//   0: 128x128x32 threadblock, 64x64x32 warp, 3 stages            (decode_lm_head, other shapes)
//   1: 128x256x32 threadblock, 64x64x32 warp, 3 stages, swizzle 2 (16K-token prefill shapes)
//   2:  64x128x32 threadblock, 32x64x32 warp, 6 stages            (decode_qkv, small M)
using Hgemm0 = HgemmT<128, 128, 32, 64, 64, 32, 3>;
using Hgemm1 = HgemmT<128, 256, 32, 64, 64, 32, 3, 2>;
using Hgemm2 = HgemmT<64, 128, 32, 32, 64, 32, 6>;

static int RefConfigFor(int M, int N, int K) {
  if (M == 128 && N == 151936 && K == 5120) return 0;   // decode_lm_head
  if (M >= 4096) return 1;                               // prefill shapes
  if (M <= 512) return 2;                                // decode_qkv and other small-M shapes
  return 0;
}


template <typename Gemm>
static cudaError_t RunCutlass(int M, int N, int K, float alpha, cutlass::half_t const *A, int lda,
                              cutlass::half_t const *B, int ldb, float beta, cutlass::half_t *D,
                              int ldd) {
  typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {D, ldd}, {D, ldd}, {alpha, beta});
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (gemm_op.initialize(args, nullptr) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm_op() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaGetLastError();
}

/// Run CUTLASS GEMM: D = alpha * A * B + beta * D (row-major, half precision)
cudaError_t CutlassHgemm(
    int M, int N, int K,
    float alpha,
    cutlass::half_t const *A, int lda,
    cutlass::half_t const *B, int ldb,
    float beta,
    cutlass::half_t *D, int ldd) {
  switch (RefConfigFor(M, N, K)) {
    case 1: return RunCutlass<Hgemm1>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    case 2: return RunCutlass<Hgemm2>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    default: return RunCutlass<Hgemm0>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
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

// FP16 output of an FP32-accumulated product: a correct result is within half an FP16 ulp
// (2^-11 relative) of the exact value plus a small FP32 accumulation error. Element bound:
// |D - exact| <= kTolRel * |exact| + kTolMax * RMS(exact) (3x the FP16 rounding bound, >= 4x
// the FP32 accumulation error); relative Frobenius error <= kTolFro (~3x a correct result).
// FP16 accumulation (~1e-2) or FP8/INT8 operands (> 1e-2) fail both bounds.
static const double kTolRel = 1.5e-3;
static const double kTolMax = 2e-4;
static const double kTolFro = 7.5e-4;

/// Test correctness of the reference and (if KH_TEST_SOLUTION) the solution against a golden
/// result computed on the device (FP16 products are exact in FP32). Every element is checked.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  int lda = K, ldb = N, ldd = N;  // row-major
  size_t count_A = size_t(M) * lda, count_B = size_t(K) * ldb, count_D = size_t(M) * ldd;

  __half *A, *B, *D_ref;
  double *G;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__half)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__half)));
  KH_CHECK(cudaMalloc(&D_ref, count_D * sizeof(__half)));
  KH_CHECK(cudaMalloc(&G, count_D * sizeof(double)));
  kh_fill(A, count_A, 1u);
  kh_fill(B, count_B, 2u);
  // Initial D: random when beta != 0; otherwise a finite poison (0 * poison = 0) so that
  // elements the solution never writes fail.
  if (beta != 0.0f) kh_fill(D_ref, count_D, 3u);
  else kh_fill_value(D_ref, count_D, kPoison16);

  kh_golden_gemm<float>(M, N, K, alpha, A, lda, 1, B, ldb, 1, beta, D_ref, ldd, 1, G);

  cudaError_t result = cudaSuccess;
#ifdef KH_TEST_SOLUTION
  __half *D_sol;
  KH_CHECK(cudaMalloc(&D_sol, count_D * sizeof(__half)));
  KH_CHECK(cudaMemcpy(D_sol, D_ref, count_D * sizeof(__half), cudaMemcpyDeviceToDevice));
#endif

  result = CutlassHgemm(M, N, K, alpha, reinterpret_cast<cutlass::half_t const *>(A), lda,
                        reinterpret_cast<cutlass::half_t const *>(B), ldb, beta,
                        reinterpret_cast<cutlass::half_t *>(D_ref), ldd);
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
  } else {
    KhResult r = kh_check(D_ref, ldd, 1, G, M, N, kTolMax, kTolFro, kTolRel);
    kh_print("reference", r, kTolMax, kTolFro);
    if (!r.pass) {
      std::cerr << "CUTLASS reference incorrect vs golden." << std::endl;
      result = cudaErrorUnknown;
    }
  }

#ifdef KH_TEST_SOLUTION
  if (result == cudaSuccess) {
    result = GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D_sol, ldd);
    cudaError_t sync = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = sync;
    if (result != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    } else {
      KhResult r = kh_check(D_sol, ldd, 1, G, M, N, kTolMax, kTolFro, kTolRel);
      kh_print("solution ", r, kTolMax, kTolFro);
      if (!r.pass) {
        std::cerr << "Solution incorrect vs golden: max|err|/rms = " << r.max_err_rel
                  << ", rel_fro_err = " << r.fro_rel << ", non-finite = " << r.nonfinite
                  << ", elements over tolerance (" << kTolRel << "*|ref| + " << kTolMax
                  << "*rms) = " << r.nbad << std::endl;
        result = cudaErrorUnknown;
      }
    }
  }
  cudaFree(D_sol);
#endif

  cudaFree(G);
  cudaFree(D_ref);
  cudaFree(B);
  cudaFree(A);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = N, ldd = N;
  size_t count_A = size_t(M) * lda, count_B = size_t(K) * ldb, count_D = size_t(M) * ldd;

  __half *A, *B, *D;
  KH_CHECK(cudaMalloc(&A, count_A * sizeof(__half)));
  KH_CHECK(cudaMalloc(&B, count_B * sizeof(__half)));
  KH_CHECK(cudaMalloc(&D, count_D * sizeof(__half)));
  kh_fill(A, count_A, 1u);
  kh_fill(B, count_B, 2u);
  kh_fill(D, count_D, 3u);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto ref = [&]() {
    CutlassHgemm(M, N, K, alpha, reinterpret_cast<cutlass::half_t const *>(A), lda,
                 reinterpret_cast<cutlass::half_t const *>(B), ldb, beta,
                 reinterpret_cast<cutlass::half_t *>(D), ldd);
  };

  // Warmup reference
  for (int i = 0; i < 3; i++) ref();
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    ref();
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
  kh_fill(D, count_D, 3u);
  for (int i = 0; i < 3; i++) {
    GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    GemmPermute(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
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
  cudaFree(D);
  cudaFree(B);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 128, N = 128, K = 128;
    std::stringstream(arg[1]) >> M;
    std::stringstream(arg[2]) >> N;
    std::stringstream(arg[3]) >> K;
    if (argc > 4) std::stringstream(arg[4]) >> alpha;
    if (argc > 5) std::stringstream(arg[5]) >> beta;
    if (argc > 6) std::stringstream(arg[6]) >> iterations;
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

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, alpha, beta);
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
