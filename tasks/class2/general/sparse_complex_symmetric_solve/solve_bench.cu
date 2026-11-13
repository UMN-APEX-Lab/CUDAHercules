// Harness of the complex-symmetric sparse solve task: X = A^-1 B for large sparse complex-symmetric
// A (FEM discretizations of 3D electromagnetic problems) and a few right-hand sides.
//
// The solution provides (solution.cu):
//   extern "C" void sparse_solve(int n, long long nnz, const int* row_ptr, const int* col_idx,
//                                const cuDoubleComplex* values, int nrhs, const cuDoubleComplex* B,
//                                cuDoubleComplex* X);
// A is given as the upper triangle (col >= row, diagonal included, columns sorted within a row) in
// CSR; B and X are n x nrhs, column-major. All pointers are device pointers. The call is timed
// from entry to device idle (wall clock), and must deliver ||B_k - A X_k|| / ||B_k|| <= 1e-8 for
// every column k (checked in extended precision).
//
// Every call gets a fresh random symmetric permutation P A P^T (and P B) of the test system, so no
// work (ordering, analysis, factors) can be carried over between calls. The reference solver runs
// as two processes on the same permuted systems; its time is the sum of the phases it reports,
// excluding process start-up, CUDA context creation, file reading and its own checks.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef KH_TEST_SOLUTION
extern "C" void sparse_solve(int n, long long nnz, const int *row_ptr, const int *col_idx,
                             const cuDoubleComplex *values, int nrhs, const cuDoubleComplex *B,
                             cuDoubleComplex *X);
#endif

#ifndef KH_REF_BIN_DIR
#define KH_REF_BIN_DIR "."
#endif

typedef std::complex<double> cplx;
constexpr double kRelResidualLimit = 1e-8;

[[noreturn]] static void die(std::string const &what) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s\n", what.c_str());
  std::printf("Incorrect\n");
  std::exit(1);
}

static void cuda_ok(cudaError_t e, char const *what) {
  if (e != cudaSuccess) die(std::string(what) + ": CUDA error: " + cudaGetErrorString(e));
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------
// Test systems: $HERCULES_DATA_DIR/sparse_solver/<system>/<case>/{A.bin,B.bin}. Both files are
// COO: a 32-byte header {uint32 info; (pad); int64 rows, cols, nnz}, int32 row[nnz], int32
// col[nnz], complex128 val[nnz], then the CRC32 of everything before it. A holds the upper
// triangle of the complex-symmetric matrix, B the nonzeros of the right-hand sides (0-based).
// ------------------------------------------------------------------------------------------------
struct TestCase {
  char const *label, *system, *dir;
};
static TestCase const kCases[] = {
    {"em_7M_1MHz", "em_7M", "1000000.000000"},
    {"em_7M_1GHz", "em_7M", "1000000000.000000"},
    {"em_18M_5MHz", "em_18M", "5000000.000000"},
    {"em_18M_142MHz", "em_18M", "142000000.000000"},
};

struct Coo {
  unsigned char header[32];
  int64_t rows = 0, cols = 0, nnz = 0;
  std::vector<int> r, c;
  std::vector<cplx> v;
};

static Coo read_coo(std::string const &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) die("cannot open " + path + " (see scripts/prepare_solver_data.py)");
  Coo m;
  bool ok = std::fread(m.header, 1, 32, f) == 32;
  if (ok) {
    std::memcpy(&m.rows, m.header + 8, 8);
    std::memcpy(&m.cols, m.header + 16, 8);
    std::memcpy(&m.nnz, m.header + 24, 8);
    m.r.resize(size_t(m.nnz));
    m.c.resize(size_t(m.nnz));
    m.v.resize(size_t(m.nnz));
    ok = std::fread(m.r.data(), 4, m.r.size(), f) == m.r.size() &&
         std::fread(m.c.data(), 4, m.c.size(), f) == m.c.size() &&
         std::fread(m.v.data(), 16, m.v.size(), f) == m.v.size();
  }
  std::fclose(f);
  if (!ok) die("malformed file " + path);
  return m;
}

static void write_coo(std::string const &path, Coo const &m) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) die("cannot write " + path);
  uLong crc = crc32(0L, Z_NULL, 0);
  auto put = [&](void const *p, size_t bytes) {
    if (std::fwrite(p, 1, bytes, f) != bytes) die("write failed: " + path);
    for (size_t o = 0; o < bytes; o += (size_t(1) << 30)) {
      size_t const len = std::min(bytes - o, size_t(1) << 30);
      crc = crc32(crc, static_cast<Bytef const *>(p) + o, uInt(len));
    }
  };
  put(m.header, 32);
  put(m.r.data(), m.r.size() * 4);
  put(m.c.data(), m.c.size() * 4);
  put(m.v.data(), m.v.size() * 16);
  uint32_t const fin = uint32_t(crc);
  if (std::fwrite(&fin, 4, 1, f) != 1) die("write failed: " + path);
  std::fclose(f);
}

/// Upper-triangle CSR (columns sorted within each row).
struct UpperCsr {
  int n = 0;
  std::vector<int> rp, ci;
  std::vector<cplx> v;
};

/// Builds the upper CSR of P A P^T from A's upper COO; perm[i] = new index of row i.
static UpperCsr permuted_upper(Coo const &A, std::vector<int> const &perm) {
  UpperCsr M;
  M.n = int(A.rows);
  size_t const nnz = size_t(A.nnz);
  std::vector<int> nr(nnz), nc(nnz);
  std::vector<int> cnt(size_t(M.n) + 1, 0);
#pragma omp parallel for
  for (size_t e = 0; e < nnz; ++e) {
    int const a = perm[A.r[e]], b = perm[A.c[e]];
    nr[e] = std::min(a, b);
    nc[e] = std::max(a, b);
#pragma omp atomic
    ++cnt[size_t(nr[e]) + 1];
  }
  for (int i = 0; i < M.n; ++i) cnt[size_t(i) + 1] += cnt[i];
  M.rp = cnt;
  M.ci.resize(nnz);
  M.v.resize(nnz);
  std::vector<int> cur(M.rp.begin(), M.rp.end() - 1);
#pragma omp parallel for
  for (size_t e = 0; e < nnz; ++e) {
    int pos;
#pragma omp atomic capture
    pos = cur[nr[e]]++;
    M.ci[pos] = nc[e];
    M.v[pos] = A.v[e];
  }
#pragma omp parallel
  {
    std::vector<std::pair<int, cplx>> row;
#pragma omp for schedule(dynamic, 4096)
    for (int i = 0; i < M.n; ++i) {
      row.clear();
      for (int j = M.rp[i]; j < M.rp[i + 1]; ++j) row.emplace_back(M.ci[j], M.v[j]);
      std::sort(row.begin(), row.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
      for (size_t k = 0; k < row.size(); ++k) {
        M.ci[size_t(M.rp[i]) + k] = row[k].first;
        M.v[size_t(M.rp[i]) + k] = row[k].second;
      }
    }
  }
  return M;
}

/// Full (both triangles) CSR of the original A, for the residual check.
struct FullCsr {
  int n = 0;
  std::vector<int64_t> rp;
  std::vector<int> ci;
  std::vector<cplx> v;
};

[[maybe_unused]] static FullCsr full_csr(Coo const &A) {
  FullCsr F;
  F.n = int(A.rows);
  std::vector<int64_t> cnt(size_t(F.n) + 1, 0);
  for (int64_t e = 0; e < A.nnz; ++e) {
    ++cnt[size_t(A.r[e]) + 1];
    if (A.r[e] != A.c[e]) ++cnt[size_t(A.c[e]) + 1];
  }
  for (int i = 0; i < F.n; ++i) cnt[size_t(i) + 1] += cnt[i];
  F.rp = cnt;
  F.ci.resize(size_t(F.rp[F.n]));
  F.v.resize(size_t(F.rp[F.n]));
  std::vector<int64_t> cur(F.rp.begin(), F.rp.end() - 1);
  for (int64_t e = 0; e < A.nnz; ++e) {
    int const i = A.r[e], j = A.c[e];
    F.ci[size_t(cur[i])] = j;
    F.v[size_t(cur[i]++)] = A.v[e];
    if (i != j) {
      F.ci[size_t(cur[j])] = i;
      F.v[size_t(cur[j]++)] = A.v[e];
    }
  }
  return F;
}

/// ||b_k - A x_k|| / ||b_k|| per column, accumulated in long double; non-finite x -> +inf.
[[maybe_unused]] static std::vector<double> rel_residuals(FullCsr const &A, std::vector<cplx> const &b, std::vector<cplx> const &x,
                                         int nrhs) {
  std::vector<double> out(static_cast<size_t>(nrhs));
  size_t const n = size_t(A.n);
  for (int k = 0; k < nrhs; ++k) {
    long double rr = 0.0L, bb = 0.0L;
    bool finite = true;
#pragma omp parallel for reduction(+ : rr, bb) reduction(&& : finite) schedule(dynamic, 8192)
    for (int i = 0; i < A.n; ++i) {
      long double re = b[k * n + size_t(i)].real(), im = b[k * n + size_t(i)].imag();
      bb += re * re + im * im;
      for (int64_t j = A.rp[i]; j < A.rp[i + 1]; ++j) {
        cplx const a = A.v[size_t(j)], xv = x[k * n + size_t(A.ci[size_t(j)])];
        if (!std::isfinite(xv.real()) || !std::isfinite(xv.imag())) finite = false;
        re -= (long double)a.real() * xv.real() - (long double)a.imag() * xv.imag();
        im -= (long double)a.real() * xv.imag() + (long double)a.imag() * xv.real();
      }
      rr += re * re + im * im;
    }
    out[size_t(k)] = finite ? double(std::sqrt(rr / bb)) : INFINITY;
  }
  return out;
}

// ------------------------------------------------------------------------------------------------
// Reference: two processes on the permuted system written to a scratch directory.
// ------------------------------------------------------------------------------------------------
static std::string run_capture(std::string const &cmd) {
  FILE *p = popen((cmd + " 2>&1").c_str(), "r");
  if (!p) die("cannot run " + cmd);
  std::string out;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), p)) out += buf;
  int const rc = pclose(p);
  if (rc != 0) die("reference command failed (" + std::to_string(rc) + "): " + cmd + "\n" + out.substr(out.size() > 4000 ? out.size() - 4000 : 0));
  return out;
}

static double field(std::string const &text, std::string const &key, char const *after = nullptr) {
  size_t pos = after ? text.find(after) : 0;
  if (pos == std::string::npos) die(std::string("reference report lacks ") + after);
  pos = text.find(key, pos);
  if (pos == std::string::npos) die("reference report lacks '" + key + "'");
  return std::atof(text.c_str() + pos + key.size());
}

/// Reference time in seconds for one permuted system in dir.
static double reference_solve(std::string const &dir, double *ordering, double *solver) {
  std::string const bin = KH_REF_BIN_DIR;
  std::string const o = run_capture(bin + "/ref_order " + dir + " --threads 64 --gpu -o " + dir + "/plan");
  size_t const p = o.find("ordering total");
  if (p == std::string::npos) die("reference ordering report lacks 'ordering total'\n" + o);
  *ordering = std::atof(o.c_str() + o.find(':', p) + 1);
  std::string const s = run_capture(bin + "/ref_solve --whole --reps 1 --streams 16 --gmres --mp-tol 1e-8 "
                                          "--mp-iters 200 --no-verify --user-perm " + dir + "/plan.perm " +
                                          "--user-tree " + dir + "/plan.rowtree " + dir);
  char const *once = "[testbench] once";
  double const init = field(s, " init ", once), read = field(s, " read ", once),
               verify = field(s, " verify ", once), wall = field(s, " wall ", once),
               analysis = field(s, " analysis ", once);
  char const *freq = "[testbench] freq";
  std::printf("    (reference phases: analysis %.2f s, factor %.2f s, solve %.2f s, %d preconditioner solves)\n",
              analysis, field(s, " factor ", freq), field(s, " solve ", freq), int(field(s, " solves ", freq)));
  *solver = wall - read - init - verify;
  return *ordering + *solver;
}

// ------------------------------------------------------------------------------------------------

int main() {
  char const *data_env = std::getenv("HERCULES_DATA_DIR");
  if (!data_env || !*data_env) die("HERCULES_DATA_DIR is not set (see scripts/prepare_solver_data.py)");
  std::string const root = std::string(data_env) + "/sparse_solver";
  // HERCULES_SOLVER_CASES: "dev" = only the em_7M cases, or a comma-separated list of case labels.
  char const *set_env = std::getenv("HERCULES_SOLVER_CASES");
  std::string const set = set_env ? set_env : "";
  auto selected = [&](TestCase const &tc) {
    if (set.empty()) return true;
    if (set == "dev") return !std::strcmp(tc.system, "em_7M");
    return ("," + set + ",").find(std::string(",") + tc.label + ",") != std::string::npos;
  };
  int const timed = 2;  // timed calls per case (plus one warm-up call), each on a new permutation
  std::random_device rd;
  char scratch_tmpl[] = "/dev/shm/kh_solver_XXXXXX";
  if (!mkdtemp(scratch_tmpl)) die("cannot create scratch directory under /dev/shm");
  std::string const scratch = scratch_tmpl;

  for (TestCase const &tc : kCases) {
    if (!selected(tc)) continue;
    std::string const dir = root + "/" + tc.system + "/" + tc.dir;
    Coo const A = read_coo(dir + "/A.bin");
    Coo const Bc = read_coo(dir + "/B.bin");
    int const n = int(A.rows), nrhs = int(Bc.cols);
    std::printf("\n=== %s ===\nn=%d, nnz(upper)=%lld, nrhs=%d\n", tc.label, n, (long long)A.nnz, nrhs);
    std::fflush(stdout);
#ifdef KH_TEST_SOLUTION
    FullCsr const F = full_csr(A);
    std::vector<cplx> b(size_t(n) * nrhs, cplx(0, 0));
    for (int64_t e = 0; e < Bc.nnz; ++e) b[size_t(Bc.c[e]) * n + size_t(Bc.r[e])] = Bc.v[e];
#endif

    // One permutation per call: call 0 is the warm-up.
    std::vector<std::vector<int>> perms(timed + 1, std::vector<int>(size_t(n)));
    for (auto &p : perms) {
      std::iota(p.begin(), p.end(), 0);
      std::mt19937_64 g((uint64_t(rd()) << 32) | rd());
      std::shuffle(p.begin(), p.end(), g);
    }

    // Reference.
    double ref_min = 1e30, ref_sum = 0;
    for (int call = 0; call <= timed; ++call) {
      UpperCsr const M = permuted_upper(A, perms[call]);
      Coo PA;
      std::memcpy(PA.header, A.header, 32);
      PA.rows = A.rows;
      PA.cols = A.cols;
      PA.nnz = A.nnz;
      PA.r.resize(size_t(A.nnz));
      for (int i = 0; i < n; ++i)
        for (int j = M.rp[i]; j < M.rp[i + 1]; ++j) PA.r[size_t(j)] = i;
      PA.c = M.ci;
      PA.v = M.v;
      Coo PB = Bc;
      for (auto &r : PB.r) r = perms[call][size_t(r)];
      std::string const cdir = scratch + "/" + tc.label;
      if (system(("mkdir -p '" + cdir + "'").c_str()) != 0) die("mkdir failed");
      write_coo(cdir + "/A.bin", PA);
      write_coo(cdir + "/B.bin", PB);
      double ord = 0, sol = 0;
      double const t = reference_solve(cdir, &ord, &sol);
      std::printf("  reference call %d: %.3f s (ordering %.3f s, analysis+factor+solve %.3f s)\n", call, t, ord, sol);
      std::fflush(stdout);
      if (call > 0) {
        ref_min = std::min(ref_min, t);
        ref_sum += t;
      }
      if (system(("rm -rf '" + cdir + "'").c_str()) != 0) die("rm failed");
    }
    std::printf("Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n", 1e3 * ref_sum / timed, timed, 1e3 * ref_min);
    std::fflush(stdout);

#ifdef KH_TEST_SOLUTION
    double sol_min = 1e30, sol_sum = 0;
    for (int call = 0; call <= timed; ++call) {
      std::vector<int> const &perm = perms[call];
      UpperCsr const M = permuted_upper(A, perm);
      std::vector<cplx> pb(size_t(n) * nrhs);
      for (int k = 0; k < nrhs; ++k)
        for (int i = 0; i < n; ++i) pb[size_t(k) * n + size_t(perm[i])] = b[size_t(k) * n + size_t(i)];
      int *d_rp = nullptr, *d_ci = nullptr;
      cuDoubleComplex *d_v = nullptr, *d_b = nullptr, *d_x = nullptr;
      cuda_ok(cudaMalloc(&d_rp, (size_t(n) + 1) * sizeof(int)), "alloc");
      cuda_ok(cudaMalloc(&d_ci, M.ci.size() * sizeof(int)), "alloc");
      cuda_ok(cudaMalloc(&d_v, M.v.size() * sizeof(cuDoubleComplex)), "alloc");
      cuda_ok(cudaMalloc(&d_b, pb.size() * sizeof(cuDoubleComplex)), "alloc");
      cuda_ok(cudaMalloc(&d_x, pb.size() * sizeof(cuDoubleComplex)), "alloc");
      cuda_ok(cudaMemcpy(d_rp, M.rp.data(), (size_t(n) + 1) * sizeof(int), cudaMemcpyHostToDevice), "upload");
      cuda_ok(cudaMemcpy(d_ci, M.ci.data(), M.ci.size() * sizeof(int), cudaMemcpyHostToDevice), "upload");
      cuda_ok(cudaMemcpy(d_v, M.v.data(), M.v.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "upload");
      cuda_ok(cudaMemcpy(d_b, pb.data(), pb.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "upload");
      cuda_ok(cudaMemset(d_x, 0xFF, pb.size() * sizeof(cuDoubleComplex)), "poison X");  // NaN
      cuda_ok(cudaDeviceSynchronize(), "upload");

      double const t0 = now_s();
      sparse_solve(n, (long long)M.ci.size(), d_rp, d_ci, d_v, nrhs, d_b, d_x);
      cudaError_t const e = cudaDeviceSynchronize();
      double const t = now_s() - t0;
      cuda_ok(e != cudaSuccess ? e : cudaGetLastError(), "sparse_solve");

      std::vector<cplx> px(pb.size()), x(pb.size());
      cuda_ok(cudaMemcpy(px.data(), d_x, px.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "download");
      for (int k = 0; k < nrhs; ++k)
        for (int i = 0; i < n; ++i) x[size_t(k) * n + size_t(i)] = px[size_t(k) * n + size_t(perm[i])];
      std::vector<double> const res = rel_residuals(F, b, x, nrhs);
      bool ok = true;
      std::printf("  solution call %d: %.3f s, relative residuals", call, t);
      for (double r : res) {
        std::printf(" %.3e", r);
        ok = ok && r <= kRelResidualLimit;
      }
      std::printf(" (limit %.0e) -> %s\n", kRelResidualLimit, ok ? "ok" : "FAIL");
      std::fflush(stdout);
      cudaFree(d_rp);
      cudaFree(d_ci);
      cudaFree(d_v);
      cudaFree(d_b);
      cudaFree(d_x);
      if (!ok) die("solution: relative residual above the limit");
      if (call > 0) {
        sol_min = std::min(sol_min, t);
        sol_sum += t;
      }
    }
    std::printf("Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n", 1e3 * sol_sum / timed, timed, 1e3 * sol_min);
#endif
    std::printf("Passed\n");
    std::fflush(stdout);
  }
  if (system(("rm -rf '" + scratch + "'").c_str()) != 0) std::fprintf(stderr, "could not remove %s\n", scratch.c_str());
  return 0;
}
