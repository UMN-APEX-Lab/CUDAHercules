// CSR SpGEMM harness: C = A A for a square FP32 CSR matrix A; C is returned in CSR.
//
// The implementation provides three functions (solution.h):
//   void* spgemm_preprocess(row_ptr, col_ind, values, num_rows, num_cols, nnz)
//       NOT timed. May ONLY convert A's storage format or build auxiliary data over A; it must not
//       compute anything about C (structure or values).
//   void  spgemm_compute(ctx, row_ptr, col_ind, values, &c_row_ptr, &c_col_ind, &c_values, &c_nnz,
//                        num_rows, num_cols, nnz)
//       The ONLY timed function. Must produce the complete CSR C (structure and values) on every
//       call. The output arrays are device memory owned by the implementation (allocate once,
//       reuse across calls); the harness only reads them. Column indices within a row may be in
//       any order, but every (row, column) may appear only once, and entries that are absent
//       compare as 0.
//   void  spgemm_cleanup(ctx)
//       Releases everything preprocess and compute allocated.
//
// For every test matrix the harness checks and times a tuned reference implementation, then the
// solution. Before every call the arrays returned by the previous call are poisoned, and every
// timed solution call is verified against an FP64 golden product.
#include <omp.h>
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

void *spgemm_preprocess(const int *d_row_ptr, const int *d_col_ind, const float *d_values, int num_rows,
                        int num_cols, int nnz);
void spgemm_compute(void *ctx, const int *d_row_ptr, const int *d_col_ind, const float *d_values,
                    int **d_c_row_ptr, int **d_c_col_ind, float **d_c_values, int *c_nnz, int num_rows,
                    int num_cols, int nnz);
void spgemm_cleanup(void *ctx);

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

typedef void *(*SpgemmPre)(const int *, const int *, const float *, int, int, int);
typedef void (*SpgemmCompute)(void *, const int *, const int *, const float *, int **, int **, float **, int *,
                              int, int, int);
typedef void (*SpgemmCleanup)(void *);

/// The SpGEMM test set (a subset of kSparseMatrices).
static const char *const kSpgemmMatrices[] = {
    "asia_osm", "delaunay_n24", "CoupCons3D", "case39", "gupta3", "in-2004", "TSOPF_FS_b300_c2",
};

struct GoldenCsr {
  std::vector<long long> rp;
  std::vector<int> ci;
  std::vector<double> v;
};

/// Per-thread workspace for one row of A*A in FP64: a dense accumulator over the columns when
/// they are few (<= 4M), otherwise an (column, value) list that is sorted and merged.
struct GoldenRowWs {
  std::vector<double> dense;
  std::vector<char> seen;
  std::vector<int> touched;
  std::vector<std::pair<int, double>> acc;  // the row, sorted by column
};

static void golden_row(KhHostCsr const &A, int r, GoldenRowWs &w) {
  w.acc.clear();
  if (A.cols <= (4 << 20)) {
    if (w.dense.empty()) {
      w.dense.assign(size_t(A.cols), 0.0);
      w.seen.assign(size_t(A.cols), 0);
    }
    w.touched.clear();
    for (int j = A.rp[r]; j < A.rp[r + 1]; ++j) {
      double const a = A.v[j];
      int const k = A.ci[j];
      for (int l = A.rp[k]; l < A.rp[k + 1]; ++l) {
        int const c = A.ci[l];
        if (!w.seen[c]) {
          w.seen[c] = 1;
          w.touched.push_back(c);
        }
        w.dense[c] += a * double(A.v[l]);
      }
    }
    std::sort(w.touched.begin(), w.touched.end());
    for (int c : w.touched) {
      w.acc.emplace_back(c, w.dense[c]);
      w.dense[c] = 0.0;
      w.seen[c] = 0;
    }
    return;
  }
  for (int j = A.rp[r]; j < A.rp[r + 1]; ++j) {
    double const a = A.v[j];
    int const k = A.ci[j];
    for (int l = A.rp[k]; l < A.rp[k + 1]; ++l) w.acc.emplace_back(A.ci[l], a * double(A.v[l]));
  }
  std::sort(w.acc.begin(), w.acc.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
  size_t n = 0;
  for (size_t i = 0; i < w.acc.size(); ++i) {
    if (n > 0 && w.acc[n - 1].first == w.acc[i].first) w.acc[n - 1].second += w.acc[i].second;
    else w.acc[n++] = w.acc[i];
  }
  w.acc.resize(n);
}

static GoldenCsr golden_product(KhHostCsr const &A) {
  GoldenCsr G;
  G.rp.assign(size_t(A.rows) + 1, 0);
#pragma omp parallel
  {
    GoldenRowWs w;
#pragma omp for schedule(dynamic, 64)
    for (int r = 0; r < A.rows; ++r) {
      golden_row(A, r, w);
      G.rp[size_t(r) + 1] = (long long)w.acc.size();
    }
  }
  for (int r = 0; r < A.rows; ++r) G.rp[size_t(r) + 1] += G.rp[r];
  G.ci.resize(size_t(G.rp[A.rows]));
  G.v.resize(size_t(G.rp[A.rows]));
#pragma omp parallel
  {
    GoldenRowWs w;
#pragma omp for schedule(dynamic, 64)
    for (int r = 0; r < A.rows; ++r) {
      golden_row(A, r, w);
      size_t o = size_t(G.rp[r]);
      for (auto const &e : w.acc) {
        G.ci[o] = e.first;
        G.v[o++] = e.second;
      }
    }
  }
  return G;
}

/// Device CSR returned by one call.
struct SpgemmOut {
  int *rp = nullptr, *ci = nullptr;
  float *v = nullptr;
  int nnz = -1;
};

/// Copies a returned C to the host and compares it with the golden product: structure checks
/// (row_ptr[0] = 0, non-decreasing, row_ptr[rows] = c_nnz, columns in range, no duplicates), then
/// the element criterion of kh_sparse.h over the union of both structures.
static KhVerdict check_product(GoldenCsr const &G, SpgemmOut const &o, int rows, int cols, char const *who) {
  KhVerdict bad = kh_verdict(1, 1, 0.0);
  if (o.nnz < 0 || !o.rp || (o.nnz > 0 && (!o.ci || !o.v))) {
    std::printf("  %s: invalid output (c_nnz %d, null arrays)\n", who, o.nnz);
    return bad;
  }
  std::vector<int> rp(size_t(rows) + 1), ci(size_t(o.nnz));
  std::vector<float> v(size_t(o.nnz));
  kh_sparse_cuda(cudaMemcpy(rp.data(), o.rp, rp.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy C");
  if (o.nnz > 0) {
    kh_sparse_cuda(cudaMemcpy(ci.data(), o.ci, ci.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy C");
    kh_sparse_cuda(cudaMemcpy(v.data(), o.v, v.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
  }
  if (rp[0] != 0 || rp[rows] != o.nnz) {
    std::printf("  %s: malformed row_ptr (row_ptr[0] = %d, row_ptr[rows] = %d, c_nnz = %d)\n", who, rp[0],
                rp[rows], o.nnz);
    return bad;
  }
  for (int r = 0; r < rows; ++r)
    if (rp[r + 1] < rp[r]) {
      std::printf("  %s: row_ptr decreases at row %d\n", who, r);
      return bad;
    }
  unsigned long long mism = 0, total = 0, dups = 0, out_of_range = 0;
  double max_diff = 0.0;
#pragma omp parallel reduction(+ : mism, total, dups, out_of_range) reduction(max : max_diff)
  {
    std::vector<std::pair<int, float>> row;
#pragma omp for schedule(dynamic, 256)
    for (int r = 0; r < rows; ++r) {
      row.clear();
      for (int i = rp[r]; i < rp[r + 1]; ++i) {
        if (ci[i] < 0 || ci[i] >= cols) ++out_of_range;
        else row.emplace_back(ci[i], v[i]);
      }
      std::sort(row.begin(), row.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
      for (size_t i = 1; i < row.size(); ++i) dups += row[i].first == row[i - 1].first;
      size_t i = size_t(G.rp[r]), ie = size_t(G.rp[r + 1]), j = 0;
      while (i < ie || j < row.size()) {
        double g = 0.0;
        float t = 0.0f;
        if (j >= row.size() || (i < ie && G.ci[i] < row[j].first)) {
          g = G.v[i++];
        } else if (i >= ie || row[j].first < G.ci[i]) {
          t = row[j++].second;
        } else {
          g = G.v[i++];
          t = row[j++].second;
        }
        ++total;
        double const d = std::fabs(double(t) - g);
        if (!std::isfinite(t) || (d > kAtol && d / (std::fabs(g) + 1e-8) > kRtol)) ++mism;
        if (std::isfinite(t)) max_diff = std::max(max_diff, d);
      }
    }
  }
  if (out_of_range || dups) {
    std::printf("  %s: %llu column indices out of range, %llu duplicate (row, column) entries\n", who,
                out_of_range, dups);
    return bad;
  }
  return kh_verdict(mism, total, max_diff);
}

struct SpgemmImpl {
  SpgemmPre pre;
  SpgemmCompute compute;
  SpgemmCleanup cleanup;
};

/// Checks one implementation, then (iters > 0) times it. Returns false if the untimed check fails;
/// a failing timed call aborts the run inside kh_bench.
static bool run_impl(char const *label, SpgemmImpl const &impl, GoldenCsr const &G, KhDevCsr const &W,
                     int warmup, int iters, bool verify_timed) {
  void *ctx = impl.pre(W.rp, W.ci, W.v, W.rows, W.cols, W.nnz);
  kh_sparse_cuda(cudaDeviceSynchronize(), "preprocess");
  SpgemmOut out;
  // Poison what the previous call returned: a call that hands back the same arrays without
  // rewriting them returns NaN values / -1 indices and fails the check.
  auto prep = [&](int) {
    if (out.nnz >= 0 && out.rp) kh_poison(out.rp, (size_t(W.rows) + 1) * sizeof(int));
    if (out.nnz > 0 && out.ci) kh_poison(out.ci, size_t(out.nnz) * sizeof(int));
    if (out.nnz > 0 && out.v) kh_poison(out.v, size_t(out.nnz) * sizeof(float));
  };
  auto call = [&] {
    out.nnz = -1;
    impl.compute(ctx, W.rp, W.ci, W.v, &out.rp, &out.ci, &out.v, &out.nnz, W.rows, W.cols, W.nnz);
  };
  call();  // first call: output arrays are allocated here
  kh_sparse_cuda(cudaDeviceSynchronize(), "compute");
  prep(0);
  kh_sparse_cuda(cudaDeviceSynchronize(), "output poison");
  call();  // second call must rewrite the poisoned arrays
  kh_sparse_cuda(cudaDeviceSynchronize(), "compute");
  KhVerdict const v = check_product(G, out, W.rows, W.cols, label);
  kh_print_verdict(label, v);
  if (!v.pass) return false;
  if (iters > 0) {
    auto check = [&](int) {
      if (!verify_timed) return true;
      KhVerdict const t = check_product(G, out, W.rows, W.cols, label);
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
  int const warmup = kh_sparse_warmup(2), iters = kh_sparse_iters(10);
  for (char const *name : kSpgemmMatrices) {
    if (!kh_sparse_selected(name)) continue;
    KhHostCsr const h = kh_load_matrix(name);
    if (h.rows != h.cols) kh_sparse_die("SpGEMM test matrices must be square");
    double const t0 = kh_host_ms();
    GoldenCsr const G = golden_product(h);
    std::printf("\n=== %s ===\nrows=%d, nnz(A)=%d, nnz(C)=%lld\n", name, h.rows, h.nnz, G.rp[h.rows]);
    std::fprintf(stderr, "[%s] FP64 golden product in %.1f s\n", name, (kh_host_ms() - t0) / 1000.0);
    std::fflush(stdout);
    KhDevCsr const P = kh_upload(h);  // pristine copy, restores W after each implementation
    KhDevCsr const W = kh_upload(h);  // the buffers handed to the implementations

    SpgemmImpl ref;
    kh_ref_open(std::string("spgemm_") + name, (void **)&ref.pre, (void **)&ref.compute, (void **)&ref.cleanup);
    if (!run_impl("reference", ref, G, W, warmup, iters, false))
      kh_sparse_die("reference implementation fails the correctness check (task setup error)");
    kh_restore(W, P);
#ifdef KH_TEST_SOLUTION
    SpgemmImpl sol = {spgemm_preprocess, spgemm_compute, spgemm_cleanup};
    if (!run_impl("solution", sol, G, W, warmup, iters, true)) kh_fail("solution: incorrect result");
#endif
    std::printf("Passed\n");
    std::fflush(stdout);
    KhDevCsr p = P, w = W;
    kh_free(p);
    kh_free(w);
  }
  return 0;
}
