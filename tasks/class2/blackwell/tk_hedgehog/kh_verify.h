#pragma once
// kh_verify.h -- shared verification / timing helpers for CUDA-Hercules class-1 harnesses.
//
// This file is the single source of truth. scripts/sync_kh_verify.sh copies it into every task
// directory that uses it (the runner copies only the task directory, so each task must be
// self-contained). Do not edit the per-task copies.
//
// A harness uses it like this (solution.h must be included BEFORE this header so the solution
// cannot see these helpers):
//
//   inputs:   kh_fill_uniform(ptr, n, seed, scale)        full-mantissa random data
//   outputs:  kh_poison(ptr, bytes)                       0xFF fill: NaN for fp16/bf16/fp32
//   oracle:   the task's reference kernel writes ref_out; the solution writes sol_out (a
//             different, poisoned buffer)
//   check:    kh_compare(sol_out, ref_out, n, tol)        every element; NaN/Inf fail
//   timing:   kh_time(...)                                L2 flushed before each trial
//   report:   kh_print_time("Ref"/"Kernel", t)            format parsed by the runner
//
// The reference itself must be validated offline against an independent computation when the
// task is authored; it is not re-derived at run time.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define KH_CHECK(call)                                                                         \
  do {                                                                                         \
    cudaError_t kh_err_ = (call);                                                              \
    if (kh_err_ != cudaSuccess) {                                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(kh_err_) << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                                      \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

// ---------------------------------------------------------------------------------------------
// Deterministic full-mantissa random data
// ---------------------------------------------------------------------------------------------

__host__ __device__ inline unsigned long long kh_hash64(unsigned long long i, unsigned int seed) {
  unsigned long long x = i * 0x9E3779B97F4A7C15ull + (seed + 1ull) * 0xD1B54A32D192ED03ull;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 33;
  return x;
}

/// Uniform in [-1, 1): a random 25-bit signed integer times 2^-24, so values with |v| >= 0.5 carry
/// a full 24-bit significand (rounding to a narrower type changes the result).
__host__ __device__ inline float kh_uniform(unsigned long long i, unsigned int seed) {
  return float(int(unsigned(kh_hash64(i, seed) >> 32)) >> 7) * (1.0f / 16777216.0f);
}

/// Random positive value in [lo, hi), used for scale factors (never 0, never 1.0 by construction).
__host__ __device__ inline float kh_uniform_pos(unsigned long long i, unsigned int seed,
                                                float lo, float hi) {
  float u = float(unsigned(kh_hash64(i, seed) >> 40)) * (1.0f / 16777216.0f);  // [0,1)
  return lo + (hi - lo) * u;
}

template <typename T> __device__ inline T kh_from_float(float x);
template <> __device__ inline float kh_from_float<float>(float x) { return x; }
template <> __device__ inline __half kh_from_float<__half>(float x) { return __float2half_rn(x); }
template <> __device__ inline __nv_bfloat16 kh_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}
template <> __device__ inline __nv_fp8_e4m3 kh_from_float<__nv_fp8_e4m3>(float x) {
  return __nv_fp8_e4m3(x);
}
template <> __device__ inline __nv_fp8_e5m2 kh_from_float<__nv_fp8_e5m2>(float x) {
  return __nv_fp8_e5m2(x);
}

template <typename T>
__global__ void kh_fill_kernel(T *p, size_t n, unsigned int seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_from_float<T>(scale * kh_uniform(i, seed));
}

/// Fill p[0..n) with uniform values in [-scale, scale), rounded to T.
template <typename T>
static void kh_fill_uniform(T *p, size_t n, unsigned int seed, float scale = 1.0f) {
  kh_fill_kernel<T><<<4096, 256>>>(p, n, seed, scale);
  KH_CHECK(cudaGetLastError());
}

/// Fill raw bytes with a fixed pattern (default 0xFF: NaN for fp16 / bf16 / fp32 outputs). Used to
/// pre-fill every output buffer before the solution runs, so elements it fails to write cannot
/// be mistaken for correct results.
static void kh_poison(void *p, size_t bytes, int byte = 0xFF) {
  KH_CHECK(cudaMemset(p, byte, bytes));
}

// ---------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------

__device__ inline double kh_to_double(float x) { return double(x); }
__device__ inline double kh_to_double(__half x) { return double(__half2float(x)); }
__device__ inline double kh_to_double(__nv_bfloat16 x) { return double(__bfloat162float(x)); }
__device__ inline double kh_to_double(__nv_fp8_e4m3 x) { return double(float(x)); }
__device__ inline double kh_to_double(__nv_fp8_e5m2 x) { return double(float(x)); }

struct KhTol {
  /// Every element must satisfy |x - ref| <= max_rms * RMS(ref) + rel * |ref|.
  double max_rms;
  /// Relative Frobenius error ||x - ref|| / ||ref|| must be <= fro.
  double fro;
  double rel;
  KhTol(double max_rms_, double fro_, double rel_ = 0.0) : max_rms(max_rms_), fro(fro_), rel(rel_) {}
};

struct KhStats {
  double sum_r2, sum_d2;
  unsigned long long max_d_bits, nonfinite, nbad;
};

__device__ inline void kh_block_add(double v, double *dst) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(dst, v);
}

template <typename TR>
__global__ void kh_sumsq_kernel(TR const *R, size_t n, KhStats *st) {
  double s = 0.0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    double r = kh_to_double(R[i]);
    if (isfinite(r)) s += r * r;
  }
  kh_block_add(s, &st->sum_r2);
}

template <typename TX, typename TR>
__global__ void kh_compare_kernel(TX const *X, TR const *R, size_t n, double tol_abs,
                                  double tol_rel, KhStats *st) {
  double s = 0.0, mx = 0.0;
  unsigned long long nf = 0, nb = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    double x = kh_to_double(X[i]);
    double r = kh_to_double(R[i]);
    if (!isfinite(x) || !isfinite(r)) { ++nf; continue; }   // NaN / Inf never pass
    double d = fabs(x - r);
    if (!(d <= tol_abs + tol_rel * fabs(r))) ++nb;
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
  double rms_ref, max_err_rel, fro_rel;   // max_err_rel = max|x-ref| / RMS(ref)
  unsigned long long nonfinite, nbad;
  bool pass;
};

/// Compare x (solution or reference under test) with ref over n elements, flat index. Both
/// buffers must hold the same logical layout. Every element is checked.
template <typename TX, typename TR>
static KhResult kh_compare(TX const *x, TR const *ref, size_t n, KhTol tol) {
  KhStats *st;
  KH_CHECK(cudaMalloc(&st, sizeof(KhStats)));
  KH_CHECK(cudaMemset(st, 0, sizeof(KhStats)));
  kh_sumsq_kernel<TR><<<2048, 256>>>(ref, n, st);
  KH_CHECK(cudaGetLastError());
  KhStats h;
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  double rms = std::sqrt(h.sum_r2 / double(n));
  kh_compare_kernel<TX, TR><<<2048, 256>>>(x, ref, n, tol.max_rms * rms, tol.rel, st);
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  KH_CHECK(cudaFree(st));
  double max_d;
  std::memcpy(&max_d, &h.max_d_bits, sizeof(max_d));
  KhResult r;
  r.rms_ref = rms;
  r.max_err_rel = rms > 0 ? max_d / rms : INFINITY;
  r.fro_rel = h.sum_r2 > 0 ? std::sqrt(h.sum_d2 / h.sum_r2) : INFINITY;
  r.nonfinite = h.nonfinite;
  r.nbad = h.nbad;
  // rms == 0 means the reference output is all zero / non-finite: the test itself is broken.
  r.pass = (rms > 0) && h.nonfinite == 0 && h.nbad == 0 && r.fro_rel <= tol.fro;
  return r;
}

// ---------------------------------------------------------------------------------------------
// Ragged comparison: row r of a row-major [rows, stride] matrix contributes only columns
// [0, len[r]) -- used for outputs with a masked / compressed / padded region that is not defined.
// Same criteria as kh_compare, evaluated over the contributing elements only (RMS(ref) too).
// ---------------------------------------------------------------------------------------------

template <typename TR>
__global__ void kh_sumsq_ragged_kernel(TR const *R, int rows, size_t stride, int const *len,
                                       KhStats *st) {
  double s = 0.0;
  for (int r = blockIdx.x; r < rows; r += gridDim.x)
    for (int c = threadIdx.x; c < len[r]; c += blockDim.x) {
      double v = kh_to_double(R[r * stride + c]);
      if (isfinite(v)) s += v * v;
    }
  kh_block_add(s, &st->sum_r2);
}

template <typename TX, typename TR>
__global__ void kh_compare_ragged_kernel(TX const *X, TR const *R, int rows, size_t stride,
                                         int const *len, double tol_abs, double tol_rel,
                                         KhStats *st) {
  double s = 0.0, mx = 0.0;
  unsigned long long nf = 0, nb = 0;
  for (int r = blockIdx.x; r < rows; r += gridDim.x)
    for (int c = threadIdx.x; c < len[r]; c += blockDim.x) {
      double x = kh_to_double(X[r * stride + c]);
      double ref = kh_to_double(R[r * stride + c]);
      if (!isfinite(x) || !isfinite(ref)) { ++nf; continue; }
      double d = fabs(x - ref);
      if (!(d <= tol_abs + tol_rel * fabs(ref))) ++nb;
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

/// Compares columns [0, len[r]) of every row r (len is a DEVICE array, `total` = sum of len on the
/// host). Elements outside that region are not looked at.
template <typename TX, typename TR>
static KhResult kh_compare_ragged(TX const *x, TR const *ref, int rows, size_t stride,
                                  int const *len, size_t total, KhTol tol) {
  KhStats *st;
  KH_CHECK(cudaMalloc(&st, sizeof(KhStats)));
  KH_CHECK(cudaMemset(st, 0, sizeof(KhStats)));
  kh_sumsq_ragged_kernel<TR><<<2048, 256>>>(ref, rows, stride, len, st);
  KH_CHECK(cudaGetLastError());
  KhStats h;
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  double rms = std::sqrt(h.sum_r2 / double(total));
  kh_compare_ragged_kernel<TX, TR><<<2048, 256>>>(x, ref, rows, stride, len, tol.max_rms * rms,
                                                  tol.rel, st);
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaMemcpy(&h, st, sizeof(h), cudaMemcpyDeviceToHost));
  KH_CHECK(cudaFree(st));
  double max_d;
  std::memcpy(&max_d, &h.max_d_bits, sizeof(max_d));
  KhResult r;
  r.rms_ref = rms;
  r.max_err_rel = rms > 0 ? max_d / rms : INFINITY;
  r.fro_rel = h.sum_r2 > 0 ? std::sqrt(h.sum_d2 / h.sum_r2) : INFINITY;
  r.nonfinite = h.nonfinite;
  r.nbad = h.nbad;
  r.pass = (rms > 0) && h.nonfinite == 0 && h.nbad == 0 && r.fro_rel <= tol.fro;
  return r;
}

static void kh_print_result(char const *who, KhResult const &r, KhTol tol) {
  std::printf("  %s: max|err|/rms = %.3e (tol %.1e), rel_fro_err = %.3e (tol %.1e), "
              "non-finite = %llu, elements over tol = %llu -> %s\n",
              who, r.max_err_rel, tol.max_rms, r.fro_rel, tol.fro, r.nonfinite, r.nbad,
              r.pass ? "ok" : "FAIL");
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------
// Timing: L2 flushed before every trial; reference and solution are timed identically
// ---------------------------------------------------------------------------------------------

struct KhTime { float avg_ms, min_ms; int trials; };

class KhL2Flusher {
 public:
  KhL2Flusher() {
    int dev = 0, l2 = 0;
    KH_CHECK(cudaGetDevice(&dev));
    KH_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    bytes_ = size_t(l2) * 2;
    if (bytes_ < (size_t(256) << 20)) bytes_ = size_t(256) << 20;
    KH_CHECK(cudaMalloc(&buf_, bytes_));
  }
  ~KhL2Flusher() { cudaFree(buf_); }
  void flush() { KH_CHECK(cudaMemsetAsync(buf_, 0xA5, bytes_)); }
 private:
  void *buf_ = nullptr;
  size_t bytes_ = 0;
};

/// Times f() with CUDA events: 3 warm-up calls, then `trials` timed calls, each preceded by an L2
/// flush that is outside the timed region. `prep` (optional) runs before every call, also outside
/// the timed region (use it to re-poison outputs or reset workspaces). Host-side setup that is not
/// part of the operation (descriptor / plan construction) must happen before kh_time.
template <typename F, typename P>
static KhTime kh_time(F &&f, int trials, P &&prep) {
  static KhL2Flusher flusher;
  for (int i = 0; i < 3; ++i) { prep(); f(); }
  KH_CHECK(cudaDeviceSynchronize());
  cudaEvent_t s, e;
  KH_CHECK(cudaEventCreate(&s));
  KH_CHECK(cudaEventCreate(&e));
  float total = 0.f, mn = 1e30f;
  for (int i = 0; i < trials; ++i) {
    prep();
    flusher.flush();
    KH_CHECK(cudaEventRecord(s));
    f();
    KH_CHECK(cudaEventRecord(e));
    KH_CHECK(cudaEventSynchronize(e));
    float ms;
    KH_CHECK(cudaEventElapsedTime(&ms, s, e));
    total += ms;
    mn = std::min(mn, ms);
  }
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaEventDestroy(s));
  KH_CHECK(cudaEventDestroy(e));
  return {total / trials, mn, trials};
}

template <typename F>
static KhTime kh_time(F &&f, int trials) { return kh_time(f, trials, [] {}); }

/// Prints "<label> time: X ms (avg over N trials, min: Y ms)" (label "Ref" or "Kernel").
static void kh_print_time(char const *label, KhTime const &t) {
  std::printf("%s time: %.4f ms (avg over %d trials, min: %.4f ms)\n", label, t.avg_ms, t.trials,
              t.min_ms);
  std::fflush(stdout);
}

static void kh_print_speedup(KhTime const &ref, KhTime const &sol) {
  std::printf("Speedup: %.4fx (ref_min / kernel_min)\n", ref.min_ms / sol.min_ms);
  std::fflush(stdout);
}

/// Iterations for timing: KH_BENCHMARK env var (set by the runner), default 20; 0 disables.
static int kh_trials() {
  char const *env = std::getenv("KH_BENCHMARK");
  return env ? std::atoi(env) : 20;
}
