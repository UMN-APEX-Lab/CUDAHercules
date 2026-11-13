// Shared support code of the CSR sparse harnesses (SpMV / SpMM / SpGEMM): matrix loading,
// deterministic input generation, reference-library loading and the correctness criteria.
// This file is identical in every sparse task directory.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------------------------------------
// Test matrices. Each one is a public sparse matrix (SuiteSparse / OGB / DGL) stored as
//   int32 rows, int32 cols, int32 nnz, int32 row_ptr[rows + 1], int32 col_ind[nnz],
//   float32 values[nnz], float32 x[cols]          (x is not used by the harnesses)
// in $HERCULES_DATA_DIR/sparse_matrices/<name>.bin (see scripts/prepare_sparse_data.py).
// Pattern-only matrices carry 1.0.
// ------------------------------------------------------------------------------------------------
static const char *const kSparseMatrices[] = {
    "af_shell10",   "CoupCons3D", "delaunay_n24",     "asia_osm",      "human_gene1", "mouse_gene",
    "gupta3",       "in-2004",    "ogbn-proteins",    "reddit",        "case39",      "TSOPF_FS_b300_c2",
};

struct KhHostCsr {
  int rows = 0, cols = 0, nnz = 0;
  std::vector<int> rp, ci;
  std::vector<float> v;
};

struct KhDevCsr {
  int rows = 0, cols = 0, nnz = 0;
  int *rp = nullptr, *ci = nullptr;
  float *v = nullptr;
};

[[noreturn]] static void kh_sparse_die(char const *what) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s\n", what);
  std::printf("Incorrect\n");
  std::exit(1);
}

static void kh_sparse_cuda(cudaError_t e, char const *what) {
  if (e != cudaSuccess) {
    char msg[512];
    std::snprintf(msg, sizeof(msg), "%s: CUDA error: %s", what, cudaGetErrorString(e));
    kh_sparse_die(msg);
  }
}

static std::string kh_sparse_data_dir() {
  char const *d = std::getenv("HERCULES_DATA_DIR");
  if (!d || !*d) kh_sparse_die("HERCULES_DATA_DIR is not set (see scripts/prepare_sparse_data.py)");
  return std::string(d) + "/sparse_matrices";
}

static KhHostCsr kh_load_matrix(char const *name) {
  std::string const path = kh_sparse_data_dir() + "/" + name + ".bin";
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::string msg = "cannot open " + path + " (see scripts/prepare_sparse_data.py)";
    kh_sparse_die(msg.c_str());
  }
  KhHostCsr m;
  int hdr[3];
  bool ok = std::fread(hdr, sizeof(int), 3, f) == 3;
  if (ok) {
    m.rows = hdr[0];
    m.cols = hdr[1];
    m.nnz = hdr[2];
    m.rp.resize(size_t(m.rows) + 1);
    m.ci.resize(size_t(m.nnz));
    m.v.resize(size_t(m.nnz));
    ok = std::fread(m.rp.data(), sizeof(int), m.rp.size(), f) == m.rp.size() &&
         std::fread(m.ci.data(), sizeof(int), m.ci.size(), f) == m.ci.size() &&
         std::fread(m.v.data(), sizeof(float), m.v.size(), f) == m.v.size();
  }
  std::fclose(f);
  if (!ok || m.rp[0] != 0 || m.rp[m.rows] != m.nnz) {
    std::string msg = "malformed matrix file " + path;
    kh_sparse_die(msg.c_str());
  }
  return m;
}

static KhDevCsr kh_upload(KhHostCsr const &m) {
  KhDevCsr d;
  d.rows = m.rows;
  d.cols = m.cols;
  d.nnz = m.nnz;
  kh_sparse_cuda(cudaMalloc(&d.rp, m.rp.size() * sizeof(int)), "matrix upload");
  kh_sparse_cuda(cudaMalloc(&d.ci, std::max<size_t>(1, m.ci.size()) * sizeof(int)), "matrix upload");
  kh_sparse_cuda(cudaMalloc(&d.v, std::max<size_t>(1, m.v.size()) * sizeof(float)), "matrix upload");
  kh_sparse_cuda(cudaMemcpy(d.rp, m.rp.data(), m.rp.size() * sizeof(int), cudaMemcpyHostToDevice),
                 "matrix upload");
  kh_sparse_cuda(cudaMemcpy(d.ci, m.ci.data(), m.ci.size() * sizeof(int), cudaMemcpyHostToDevice),
                 "matrix upload");
  kh_sparse_cuda(cudaMemcpy(d.v, m.v.data(), m.v.size() * sizeof(float), cudaMemcpyHostToDevice),
                 "matrix upload");
  return d;
}

/// Restores the CSR buffers handed to an implementation from the pristine copy (an implementation
/// may reorder or overwrite its inputs; the next implementation and the checks must not see that).
static void kh_restore(KhDevCsr const &dst, KhDevCsr const &src) {
  kh_sparse_cuda(cudaMemcpy(dst.rp, src.rp, (size_t(src.rows) + 1) * sizeof(int), cudaMemcpyDeviceToDevice),
                 "matrix restore");
  kh_sparse_cuda(cudaMemcpy(dst.ci, src.ci, size_t(src.nnz) * sizeof(int), cudaMemcpyDeviceToDevice),
                 "matrix restore");
  kh_sparse_cuda(cudaMemcpy(dst.v, src.v, size_t(src.nnz) * sizeof(float), cudaMemcpyDeviceToDevice),
                 "matrix restore");
}

static void kh_free(KhDevCsr &d) {
  cudaFree(d.rp);
  cudaFree(d.ci);
  cudaFree(d.v);
  d = KhDevCsr();
}

// ------------------------------------------------------------------------------------------------
// Dense inputs (x, B) are a pure function of (seed, index): 24-bit uniforms in [-1, 1), exactly
// representable in FP32, so the checks recompute them instead of reading the (writable) buffers.
// ------------------------------------------------------------------------------------------------
__host__ __device__ inline float kh_hash_unit(uint64_t seed, uint64_t i) {
  uint64_t z = seed * 0x9E3779B97F4A7C15ull + i + 0x632BE59BD9B4E019ull;
  z ^= z >> 30;
  z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 27;
  z *= 0x94D049BB133111EBull;
  z ^= z >> 31;
  return float(z >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

__global__ void kh_fill_hash_kernel(float *p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = kh_hash_unit(seed, i);
}

static void kh_fill_hash(float *p, size_t n, uint64_t seed) {
  kh_fill_hash_kernel<<<2048, 256>>>(p, n, seed);
  kh_sparse_cuda(cudaGetLastError(), "input fill");
}

/// Outputs are zero-filled before every call (outside the timed region): implementations may
/// accumulate partial sums into them.
static void kh_zero(void *p, size_t bytes) {
  kh_sparse_cuda(cudaMemsetAsync(p, 0, bytes), "output zero-fill");
}

/// SpGEMM output poison: all-ones bytes (NaN values, -1 indices) in the arrays a call returned,
/// so a later call that hands back the same arrays without rewriting them is detected.
static void kh_poison(void *p, size_t bytes) {
  kh_sparse_cuda(cudaMemsetAsync(p, 0xFF, bytes), "output poison");
}

// ------------------------------------------------------------------------------------------------
// Correctness criterion (applied to every output element against an FP64 golden value g):
//   an element t matches if it is finite and (|t - g| <= atol or |t - g| / (|g| + 1e-8) <= rtol);
//   a result is correct if at most max_mismatch_rate of its elements mismatch.
// ------------------------------------------------------------------------------------------------
constexpr double kAtol = 0.01, kRtol = 0.01, kMaxMismatchRate = 0.001;

struct KhCheckCounters {
  unsigned long long mismatches;
  unsigned long long total;
  double max_diff;
};

__device__ inline bool kh_mismatch(float t, double g) {
  if (!isfinite(t)) return true;
  double const d = fabs(double(t) - g);
  return d > kAtol && d / (fabs(g) + 1e-8) > kRtol;
}

__device__ inline void kh_atomic_max_nonneg(double *addr, double v) {
  // Non-negative doubles order like their bit patterns.
  atomicMax(reinterpret_cast<unsigned long long *>(addr), __double_as_longlong(v));
}

struct KhVerdict {
  bool pass;
  unsigned long long mismatches, total;
  double max_diff;
};

static void kh_print_verdict(char const *who, KhVerdict const &v) {
  std::printf("  %s: %llu / %llu mismatched elements (%.4f%%, limit %.2f%%), max |diff| %.3e -> %s\n", who,
              v.mismatches, v.total, v.total ? 100.0 * double(v.mismatches) / double(v.total) : 0.0,
              100.0 * kMaxMismatchRate, v.max_diff, v.pass ? "ok" : "FAIL");
}

static KhVerdict kh_verdict(unsigned long long mismatches, unsigned long long total, double max_diff) {
  KhVerdict v;
  v.mismatches = mismatches;
  v.total = total;
  v.max_diff = max_diff;
  v.pass = total > 0 && double(mismatches) <= kMaxMismatchRate * double(total);
  return v;
}

// ------------------------------------------------------------------------------------------------
// Reference implementations: one tuned kernel per test matrix (and per K for SpMM), each built as
// its own shared library (KH_REF_DIR/<op>_<matrix>[_k<K>].so) exporting kh_ref_preprocess /
// kh_ref_compute / kh_ref_cleanup with the same signatures as the solution's functions.
// ------------------------------------------------------------------------------------------------
#ifndef KH_REF_DIR
#define KH_REF_DIR "ref_build"
#endif

static void *kh_ref_open(std::string const &lib, void **pre, void **comp, void **clean) {
  std::string const path = std::string(KH_REF_DIR) + "/" + lib + ".so";
  void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::string msg = "cannot load reference " + path + ": " + dlerror();
    kh_sparse_die(msg.c_str());
  }
  *pre = dlsym(h, "kh_ref_preprocess");
  *comp = dlsym(h, "kh_ref_compute");
  *clean = dlsym(h, "kh_ref_cleanup");
  if (!*pre || !*comp || !*clean) {
    std::string msg = "reference " + path + " lacks the kh_ref_* entry points";
    kh_sparse_die(msg.c_str());
  }
  return h;  // never closed: unloading a library with registered CUDA kernels is not safe
}

static int kh_sparse_iters(int dflt) {
  char const *b = std::getenv("KH_BENCHMARK");
  if (b && *b && std::atoi(b) == 0) return 0;  // correctness-only run
  char const *v = std::getenv("KH_ITERS");
  return (v && *v) ? std::atoi(v) : dflt;
}

static int kh_sparse_warmup(int dflt) {
  char const *v = std::getenv("KH_WARMUP");
  return (v && *v) ? std::atoi(v) : dflt;
}

/// KH_SPARSE_ONLY=<name>[,<name>...] restricts a run to some matrices (development aid).
static bool kh_sparse_selected(char const *name) {
  char const *only = std::getenv("KH_SPARSE_ONLY");
  if (!only || !*only) return true;
  std::string s = std::string(",") + only + ",";
  return s.find(std::string(",") + name + ",") != std::string::npos;
}
