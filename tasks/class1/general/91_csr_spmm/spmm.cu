// CSR SpMM harness: C = A B with an FP32 CSR matrix A (num_rows x num_cols) and FP32 dense
// row-major matrices B (num_cols x K) and C (num_rows x K), for K in {8, 32, 128, 256}.
//
// The implementation provides three functions (solution.h):
//   void* spmm_preprocess(row_ptr, col_ind, values, num_rows, num_cols, nnz, K)
//       NOT timed. Receives only A (and K); may convert A's storage format or build auxiliary
//       data over A.
//   void  spmm_compute(ctx, row_ptr, col_ind, values, B, C, num_rows, num_cols, nnz, K)
//       The ONLY timed function. C is zero-filled before every call (outside the timed region),
//       so partial sums may be accumulated into it.
//   void  spmm_cleanup(ctx)
//       Releases everything preprocess allocated.
//
// For every (matrix, K) case the harness checks and times a tuned reference implementation, then
// the solution. B changes between calls (two input sets with seeds drawn at run time), C is zeroed
// before every call, and every timed solution call is verified against an FP64 golden result.
#include <random>
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

#include "kh_sparse.h"

void *spmm_preprocess(const int *d_row_ptr, const int *d_col_ind, const float *d_values, int num_rows,
                      int num_cols, int nnz, int K);
void spmm_compute(void *ctx, const int *d_row_ptr, const int *d_col_ind, const float *d_values,
                  const float *d_B, float *d_C, int num_rows, int num_cols, int nnz, int K);
void spmm_cleanup(void *ctx);

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

typedef void *(*SpmmPre)(const int *, const int *, const float *, int, int, int, int);
typedef void (*SpmmCompute)(void *, const int *, const int *, const float *, const float *, float *, int, int,
                            int, int);
typedef void (*SpmmCleanup)(void *);

static int const kSpmmK[] = {8, 32, 128, 256};

/// FP64 golden of C(r, k) for one row per blockDim.y lane group, B recomputed from its seed.
__global__ void kh_spmm_check_kernel(KhDevCsr A, float const *C, int K, uint64_t seed, KhCheckCounters *c) {
  long long const r0 = blockIdx.x * (long long)blockDim.y + threadIdx.y;
  long long const rstride = gridDim.x * (long long)blockDim.y;
  for (long long r = r0; r < A.rows; r += rstride) {
    int const b = A.rp[r], e = A.rp[r + 1];
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      double g = 0.0;
      for (int j = b; j < e; ++j)
        g += double(A.v[j]) * double(kh_hash_unit(seed, uint64_t(A.ci[j]) * K + k));
      float const t = C[size_t(r) * K + k];
      if (kh_mismatch(t, g)) atomicAdd(&c->mismatches, 1ull);
      if (isfinite(t)) kh_atomic_max_nonneg(&c->max_diff, fabs(double(t) - g));
    }
  }
}

static KhVerdict kh_spmm_check(KhDevCsr const &A, float const *C, int K, uint64_t seed) {
  KhCheckCounters *c = nullptr;
  kh_sparse_cuda(cudaMalloc(&c, sizeof(KhCheckCounters)), "check");
  kh_sparse_cuda(cudaMemset(c, 0, sizeof(KhCheckCounters)), "check");
  dim3 const block(32, 8);
  kh_spmm_check_kernel<<<8192, block>>>(A, C, K, seed, c);
  KhCheckCounters h;
  kh_sparse_cuda(cudaMemcpy(&h, c, sizeof(h), cudaMemcpyDeviceToHost), "check");
  cudaFree(c);
  return kh_verdict(h.mismatches, (unsigned long long)A.rows * K, h.max_diff);
}

struct SpmmImpl {
  SpmmPre pre;
  SpmmCompute compute;
  SpmmCleanup cleanup;
};

/// Checks one implementation on input set 0, then (iters > 0) times it. Returns false if the
/// untimed check fails; a failing timed call aborts the run inside kh_bench.
static bool run_impl(char const *label, SpmmImpl const &impl, KhDevCsr const &P, KhDevCsr const &W, int K,
                     float *B, float *C, uint64_t const seeds[2], int warmup, int iters, bool verify_timed) {
  void *ctx = impl.pre(W.rp, W.ci, W.v, W.rows, W.cols, W.nnz, K);
  kh_sparse_cuda(cudaDeviceSynchronize(), "preprocess");
  auto prep = [&](int s) {
    kh_fill_hash(B, size_t(W.cols) * K, seeds[s]);
    kh_zero(C, size_t(W.rows) * K * sizeof(float));
  };
  auto call = [&] { impl.compute(ctx, W.rp, W.ci, W.v, B, C, W.rows, W.cols, W.nnz, K); };
  prep(0);
  kh_sparse_cuda(cudaDeviceSynchronize(), "input setup");
  call();
  kh_sparse_cuda(cudaDeviceSynchronize(), "compute");
  KhVerdict const v = kh_spmm_check(P, C, K, seeds[0]);
  kh_print_verdict(label, v);
  if (!v.pass) return false;
  if (iters > 0) {
    auto check = [&](int s) {
      if (!verify_timed) return true;
      KhVerdict const t = kh_spmm_check(P, C, K, seeds[s]);
      if (!t.pass) kh_print_verdict(label, t);
      return t.pass;
    };
    bool const is_ref = !std::strcmp(label, "reference");
    KhTiming const r = kh_bench(is_ref ? "Ref time" : "Kernel time", warmup, iters, prep, call, check);
    std::printf("%s time: %.4f ms (avg over %d trials, min: %.4f ms)\n", is_ref ? "Ref" : "Kernel", r.avg_ms,
                iters, r.min_ms);
  }
  impl.cleanup(ctx);
  kh_sparse_cuda(cudaDeviceSynchronize(), "cleanup");
  return true;
}

int main() {
  int const warmup = kh_sparse_warmup(3), iters = kh_sparse_iters(20);
  std::random_device rd;
  for (char const *name : kSparseMatrices) {
    if (!kh_sparse_selected(name)) continue;
    KhHostCsr const h = kh_load_matrix(name);
    KhDevCsr const P = kh_upload(h);  // pristine copy, used by the checks only
    KhDevCsr const W = kh_upload(h);  // the buffers handed to the implementations
    for (int K : kSpmmK) {
      std::printf("\n=== %s_K%d ===\nrows=%d, cols=%d, nnz=%d, K=%d\n", name, K, h.rows, h.cols, h.nnz, K);
      std::fflush(stdout);
      float *B = nullptr, *C = nullptr;
      kh_sparse_cuda(cudaMalloc(&B, size_t(h.cols) * K * sizeof(float)), "alloc B");
      kh_sparse_cuda(cudaMalloc(&C, size_t(h.rows) * K * sizeof(float)), "alloc C");
      uint64_t const seeds[2] = {(uint64_t(rd()) << 32) | rd(), (uint64_t(rd()) << 32) | rd()};

      SpmmImpl ref;
      kh_ref_open(std::string("spmm_") + name + "_k" + std::to_string(K), (void **)&ref.pre,
                  (void **)&ref.compute, (void **)&ref.cleanup);
      if (!run_impl("reference", ref, P, W, K, B, C, seeds, warmup, iters, false))
        kh_sparse_die("reference implementation fails the correctness check (task setup error)");
      kh_restore(W, P);
#ifdef KH_TEST_SOLUTION
      SpmmImpl sol = {spmm_preprocess, spmm_compute, spmm_cleanup};
      if (!run_impl("solution", sol, P, W, K, B, C, seeds, warmup, iters, true))
        kh_fail("solution: incorrect result");
      kh_restore(W, P);
#endif
      std::printf("Passed\n");
      std::fflush(stdout);
      cudaFree(B);
      cudaFree(C);
    }
    KhDevCsr p = P, w = W;
    kh_free(p);
    kh_free(w);
  }
  return 0;
}
