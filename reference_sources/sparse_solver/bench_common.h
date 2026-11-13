// Shared helpers for the solver benchmarks: CSR conversion, residual check,
// dense RHS loading.
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include "COOBinaryIO.h"
#include "report.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

inline double since(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

inline void die(const char* msg)
{
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

// Frequency in Hz from a directory name such as data/1000000.000000/
inline double dirFrequency(std::string d)
{
    while (!d.empty() && d.back() == '/') d.pop_back();
    return std::atof(d.substr(d.find_last_of('/') + 1).c_str());
}

struct Csr
{
    int64_t n = 0, nnz = 0;
    std::vector<int32_t> rowPtr, col;
    std::vector<Complex> val;
};

// [off, off + bytes) of an open file into dst, read in parallel 64 MB pieces
inline bool preadAll(int fd, int64_t off, void* dst, int64_t bytes)
{
    const int64_t piece = int64_t(64) << 20, np = (bytes + piece - 1) / piece;
    bool ok = true;
#pragma omp parallel for schedule(dynamic, 1) reduction(&& : ok)
    for (int64_t p = 0; p < np; ++p) {
        char* d = static_cast<char*>(dst) + p * piece;
        const int64_t len = std::min(piece, bytes - p * piece);
        for (int64_t done = 0; done < len;) {
            const ssize_t r = pread(fd, d + done, len - done, off + p * piece + done);
            if (r <= 0) {
                ok = false;
                break;
            }
            done += r;
        }
    }
    return ok;
}

// zlib CRC-32 of consecutive buffers, in parallel 16 MB pieces joined with crc32_combine
inline uint32_t crc32Parallel(const std::vector<std::pair<const void*, int64_t>>& parts)
{
    struct Piece { const Bytef* p; uInt len; uLong crc; };
    std::vector<Piece> pieces;
    const int64_t piece = int64_t(16) << 20;
    for (const auto& part : parts)
        for (int64_t o = 0; o < part.second; o += piece)
            pieces.push_back({static_cast<const Bytef*>(part.first) + o, (uInt)std::min(piece, part.second - o), 0});
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)pieces.size(); ++i) pieces[i].crc = crc32(crc32(0L, Z_NULL, 0), pieces[i].p, pieces[i].len);
    uLong crc = crc32(0L, Z_NULL, 0);
    for (const Piece& pc : pieces) crc = crc32_combine(crc, pc.crc, (z_off_t)pc.len);
    return (uint32_t)crc;
}

// An A.bin straight into upper CSR, with the same checks as readMatrixABinary (header, exact size,
// CRC). The file holds the upper triangle row by row with increasing columns, so the file's index arrays are CSR:
// the columns and values are read in place and the row pointers come from the row indices. A file in any
// other order is rejected (dies). rowPtrSeconds (if given) receives the time of the order check and the row
// pointers, after reading.
inline bool readMatrixACsr(const std::string& path, Csr& A, int& sym, double* rowPtrSeconds = nullptr)
{
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    MatrixFileHeader h{};
    struct stat st{};
    bool ok = pread(fd, &h, sizeof(h), 0) == (ssize_t)sizeof(h) && fstat(fd, &st) == 0;
    int dataType = 0, indexType = 0, indexBase = 0, compressFormat = 0, storeFormat = 0, symType = 0;
    if (ok) {
        parseMatrixInfo(h.matrixInfo, dataType, indexType, indexBase, compressFormat, storeFormat, symType);
        ok = dataType == DT_COMPLEX128 && indexType == IT_INT32 && compressFormat == CF_SPARSE_COO && h.rows >= 0 &&
             h.cols >= 0 && h.nnz >= 0 && h.rows <= INT_MAX && h.cols <= INT_MAX && h.nnz <= INT_MAX &&
             (int64_t)st.st_size == (int64_t)sizeof(h) + h.nnz * (int64_t)(2 * sizeof(int32_t) + sizeof(Complex)) + 4;
    }
    const int64_t n = ok ? h.rows : 0, nnz = ok ? h.nnz : 0;
    std::vector<int32_t> row;
    uint32_t stored = 0;
    if (ok) {
        A = Csr();
        A.n = n;
        A.nnz = nnz;
        row.resize(nnz);
        A.col.resize(nnz);
        A.val.resize(nnz);
        const int64_t o = sizeof(h);
        ok = preadAll(fd, o, row.data(), nnz * 4) && preadAll(fd, o + nnz * 4, A.col.data(), nnz * 4) &&
             preadAll(fd, o + nnz * 8, A.val.data(), nnz * (int64_t)sizeof(Complex)) &&
             pread(fd, &stored, 4, o + nnz * (8 + (int64_t)sizeof(Complex))) == 4;
    }
    close(fd);
    if (!ok || crc32Parallel({{&h, (int64_t)sizeof(h)}, {row.data(), nnz * 4}, {A.col.data(), nnz * 4},
                              {A.val.data(), nnz * (int64_t)sizeof(Complex)}}) != stored)
        return false;
    sym = symType == ST_SYM ? 2 : symType == ST_SYMPOS ? 1 : 0;
    const auto tc = Clock::now();
    bool ordered = true;  // rows nondecreasing, columns increasing within a row, indices in range
#pragma omp parallel for reduction(&& : ordered)
    for (int64_t k = 0; k < nnz; ++k)
        ordered = ordered && row[k] >= 0 && row[k] < n && A.col[k] >= 0 && A.col[k] < h.cols &&
                  (k == 0 || row[k] > row[k - 1] || (row[k] == row[k - 1] && A.col[k] > A.col[k - 1]));
    if (!ordered) die(("not in CSR order (upper triangle row by row, columns increasing): " + path).c_str());
    A.rowPtr.assign(n + 1, (int32_t)nnz);
#pragma omp parallel for
    for (int64_t k = 0; k < nnz; ++k)
        for (int64_t r = k == 0 ? 0 : row[k - 1] + 1; r <= row[k]; ++r) A.rowPtr[r] = (int32_t)k;
    if (rowPtrSeconds) *rowPtrSeconds = since(tc);
    return true;
}

// Another frequency of the same dataset with every check (CRC, and that its pattern is A's): its values in
// A's CSR order
inline void readSamePattern(const std::string& path, const Csr& A, std::vector<Complex>& val)
{
    Csr B;
    int sym = 0;
    if (!readMatrixACsr(path, B, sym)) die(("read failed (CRC?): " + path).c_str());
    if (B.rowPtr != A.rowPtr || B.col != A.col) die("sparsity pattern differs between frequencies");
    val = std::move(B.val);
}

// Values of another frequency of the same dataset, in CSR order (the file's): the index arrays are known to
// equal the first frequency's, so only the header and the value block are read (no CRC check).
inline void readMatrixAValues(const std::string& path, int64_t nnz, Complex* out)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die(("cannot open " + path).c_str());
    MatrixFileHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 || h.nnz != nnz) die("A header does not match the dataset");
    if (fseeko(f, (off_t)sizeof(h) + (off_t)nnz * 8, SEEK_SET) != 0 ||
        std::fread(out, sizeof(Complex), nnz, f) != (size_t)nnz)
        die("short read of A values");
    std::fclose(f);
}

// B (COO, nports columns) -> dense column-major n x nports; rows (optional) receives the rows where B has
// an entry (the port rows), sorted, straight from the COO data.
// Both parts of B are read. REF_B_REAL_ONLY=1 drops the imaginary part.
inline std::vector<Complex> loadDenseB(const std::string& dir, int64_t n, int& nports,
                                       std::vector<int64_t>* rows = nullptr)
{
    COOMatrix bm;
    if (!readMatrixBBinary(dir + "/B.bin", bm, nports)) die("read B failed");
    const char* ro = getenv("REF_B_REAL_ONLY");
    const bool realOnly = ro && ro[0] == '1';
    std::vector<Complex> b(n * nports, Complex(0));
    for (int k = 0; k < bm.m_nnz; ++k)
        b[(int64_t)bm.col_indices[k] * n + bm.row_indices[k]] = realOnly ? Complex(bm.data[k].real(), 0) : bm.data[k];
    if (rows) {
        rows->clear();
        for (int k = 0; k < bm.m_nnz; ++k)
            if ((realOnly ? bm.data[k].real() : bm.data[k]) != 0.0) rows->push_back(bm.row_indices[k]);
        std::sort(rows->begin(), rows->end());
        rows->erase(std::unique(rows->begin(), rows->end()), rows->end());
    }
    return b;
}

// ||b - A x|| / ||b|| per column, A symmetric stored as upper CSR
inline std::vector<double> residual(const Csr& A, const std::vector<Complex>& val,
                                    const std::vector<Complex>& x, const std::vector<Complex>& b, int nrhs)
{
    std::vector<double> rel(nrhs);
    std::vector<Complex> y(A.n);
    for (int c = 0; c < nrhs; ++c) {
        const Complex* xc = x.data() + c * A.n;
        const Complex* bc = b.data() + c * A.n;
        std::fill(y.begin(), y.end(), Complex(0));
        for (int64_t i = 0; i < A.n; ++i)
            for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q) {
                int32_t j = A.col[q];
                y[i] += val[q] * xc[j];
                if (j != i) y[j] += val[q] * xc[i];
            }
        double rn = 0, bn = 0;
        for (int64_t i = 0; i < A.n; ++i) {
            rn += std::norm(bc[i] - y[i]);
            bn += std::norm(bc[i]);
        }
        rel[c] = std::sqrt(rn / bn);
    }
    return rel;
}

// Same as residual() with long double accumulation, so the result is the true residual of the
// stored (double) x rather than one inflated by rounding in the residual computation itself
inline std::vector<double> residualExtended(const Csr& A, const std::vector<Complex>& val,
                                            const std::vector<Complex>& x, const std::vector<Complex>& b, int nrhs)
{
    using LC = std::complex<long double>;
    std::vector<double> rel(nrhs);
    std::vector<LC> y(A.n);
    for (int c = 0; c < nrhs; ++c) {
        const Complex* xc = x.data() + c * A.n;
        const Complex* bc = b.data() + c * A.n;
        std::fill(y.begin(), y.end(), LC(0));
        for (int64_t i = 0; i < A.n; ++i)
            for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q) {
                int32_t j = A.col[q];
                LC a(val[q].real(), val[q].imag());
                y[i] += a * LC(xc[j].real(), xc[j].imag());
                if (j != i) y[j] += a * LC(xc[i].real(), xc[i].imag());
            }
        long double rn = 0, bn = 0;
        for (int64_t i = 0; i < A.n; ++i) {
            rn += std::norm(LC(bc[i].real(), bc[i].imag()) - y[i]);
            bn += std::norm(LC(bc[i].real(), bc[i].imag()));
        }
        rel[c] = (double)std::sqrt(rn / bn);
    }
    return rel;
}

// Extended-precision iterative refinement (CPU prototype): x is accumulated in long double,
// r = b - A x is computed in long double, and each correction A d = r is solved in working
// precision by the caller's solve(r) -> d. Per step it prints the residual of the accumulated
// x, the residual of x rounded to double (both summed in long double), the floor that any
// double x is subject to, u * || |A||x| || / ||b||, and the relative change of x at B's
// nonzero rows. On return x holds the accumulated solution rounded to double.
template <class Solve>
inline void extendedRefinement(const Csr& A, const std::vector<Complex>& val, std::vector<Complex>& x,
                               const std::vector<Complex>& b, int nrhs, int steps, Solve solve)
{
    using LC = std::complex<long double>;
    const int64_t n = A.n, len = n * nrhs;
    std::vector<LC> xl(x.begin(), x.end()), r(len);
    std::vector<int64_t> ports;
    for (int64_t i = 0; i < n; ++i)
        for (int c = 0; c < nrhs; ++c)
            if (b[c * n + i] != Complex(0)) { ports.push_back(i); break; }

    // r = b - A xl per column, returns ||r_c|| / ||b_c||; |A||x| norms for the floor on request
    auto resid = [&](const std::vector<LC>& xv, std::vector<double>* floorRel) {
        std::vector<double> rel(nrhs);
        std::vector<long double> s;
        for (int c = 0; c < nrhs; ++c) {
            const LC* xc = xv.data() + c * n;
            LC* rc = r.data() + c * n;
            for (int64_t i = 0; i < n; ++i) rc[i] = LC(b[c * n + i].real(), b[c * n + i].imag());
            if (floorRel) s.assign(n, 0);
            for (int64_t i = 0; i < n; ++i)
                for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q) {
                    int32_t j = A.col[q];
                    LC a(val[q].real(), val[q].imag());
                    rc[i] -= a * xc[j];
                    if (j != i) rc[j] -= a * xc[i];
                    if (floorRel) {
                        long double aa = std::abs(a);
                        s[i] += aa * std::abs(xc[j]);
                        if (j != i) s[j] += aa * std::abs(xc[i]);
                    }
                }
            long double rn = 0, bn = 0, sn = 0;
            for (int64_t i = 0; i < n; ++i) {
                rn += std::norm(rc[i]);
                bn += std::norm(LC(b[c * n + i].real(), b[c * n + i].imag()));
                if (floorRel) sn += s[i] * s[i];
            }
            rel[c] = (double)std::sqrt(rn / bn);
            if (floorRel) (*floorRel)[c] = (double)(std::ldexp(1.0L, -53) * std::sqrt(sn / bn));
        }
        return rel;
    };
    auto rounded = [&](const std::vector<LC>& xv) {
        std::vector<LC> xr(len);
        for (int64_t k = 0; k < len; ++k)
            xr[k] = LC((double)xv[k].real(), (double)xv[k].imag());
        return xr;
    };
    auto portVals = [&](const std::vector<LC>& xv) {
        std::vector<LC> p;
        for (int c = 0; c < nrhs; ++c)
            for (int64_t i : ports) p.push_back(xv[c * n + i]);
        return p;
    };
    auto print = [](const char* label, const std::vector<double>& v) {
        std::printf(" %s", label);
        for (double e : v) std::printf(" %.3e", e);
    };

    std::vector<LC> prevPorts = portVals(xl);
    for (int k = 0;; ++k) {
        std::vector<double> floorRel(nrhs);
        std::vector<double> rr = resid(rounded(xl), &floorRel);  // residual of x rounded to double
        std::vector<double> re = resid(xl, nullptr);             // leaves r = b - A xl for the solve
        std::printf("  ext-IR step %d            :", k);
        print("resid(x_ext)", re);
        print("| resid(double x)", rr);
        print("| floor", floorRel);
        if (k > 0) {
            std::vector<LC> p = portVals(xl);
            long double dn = 0, pn = 0;
            for (size_t m = 0; m < p.size(); ++m) { dn += std::norm(p[m] - prevPorts[m]); pn += std::norm(p[m]); }
            std::printf(" | port change %.2e", (double)std::sqrt(dn / pn));
            prevPorts = p;
        }
        std::printf("\n");
        if (k == steps) break;
        std::vector<Complex> r64(len);
        for (int64_t m = 0; m < len; ++m) r64[m] = Complex((double)r[m].real(), (double)r[m].imag());
        std::vector<Complex> d = solve(r64);
        for (int64_t m = 0; m < len; ++m) xl[m] += LC(d[m].real(), d[m].imag());
    }
    for (int64_t m = 0; m < len; ++m) x[m] = Complex((double)xl[m].real(), (double)xl[m].imag());
}

// REF_EXT_RESIDUAL=1 also prints the long double residual (slower, CPU). Returns the relative
// residual of every column.
inline std::vector<double> printResidual(const Csr& A, const std::vector<Complex>& val,
                                         const std::vector<Complex>& x, const std::vector<Complex>& b, int nrhs)
{
    auto rel = residual(A, val, x, b, nrhs);
    std::printf("  ||b-Ax||/||b||           :");
    for (double r : rel) std::printf(" %.3e", r);
    std::printf("\n");
    const char* ext = std::getenv("REF_EXT_RESIDUAL");
    if (ext && *ext == '1') {
        auto er = residualExtended(A, val, x, b, nrhs);
        std::printf("  residual, long double sum:");
        for (double r : er) std::printf(" %.3e", r);
        std::printf("\n");
    }
    return rel;
}

// x at the excited rows (B's nonzeros), full precision, so runs can be compared
inline void printPortSolution(const std::vector<Complex>& b, const std::vector<Complex>& x, int64_t n,
                              int nrhs, const char* label)
{
    std::printf("  port solution:\n");
    for (int c = 0; c < nrhs; ++c)
        for (int64_t i = 0; i < n; ++i)
            if (b[c * n + i] != Complex(0))
                std::printf("    x[%lld,%d] %-6s % .15e %+.15ej\n", (long long)i, c, label,
                            x[c * n + i].real(), x[c * n + i].imag());
}

#endif
