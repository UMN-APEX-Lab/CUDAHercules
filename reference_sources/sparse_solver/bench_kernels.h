// Small CUDA kernels for mixed-precision iterative refinement:
// complex double residual with the upper-triangular symmetric CSR, precision
// conversion with scaling, and column norms. All run on the default stream.
#ifndef BENCH_KERNELS_H
#define BENCH_KERNELS_H

#include <cuComplex.h>
#include <cstddef>
#include <cstdint>

// out[k] = (cuComplex)(in[k] * scale), k < n
void convertToC32(const cuDoubleComplex* in, cuComplex* out, int64_t n, double scale);

// x[k] += scale * d[k], k < n
void axpyFromC32(cuDoubleComplex* x, const cuComplex* d, int64_t n, double scale);

// out[k] = (cuDoubleComplex)in[k], k < n
void convertToC64(const cuComplex* in, cuDoubleComplex* out, int64_t n);

// y = A x for one column of length n; A symmetric, upper CSR
void spmvSymUpper(int64_t n, const int32_t* rowPtr, const int32_t* col, const cuDoubleComplex* val,
                  const cuDoubleComplex* x, cuDoubleComplex* y);

// r = b - A x for nrhs column-major columns of length n; A symmetric, upper CSR
void residualSymUpper(int64_t n, const int32_t* rowPtr, const int32_t* col, const cuDoubleComplex* val,
                      const cuDoubleComplex* x, const cuDoubleComplex* b, cuDoubleComplex* r, int nrhs);

// norms[c] = ||v(:, c)||_2 for nrhs columns of length n (norms is a host array)
void columnNorms(const cuDoubleComplex* v, int64_t n, int nrhs, double* norms);

// out[c] = max_k |v(k, c)| for nrhs columns of length n (out is a host array)
void columnMaxAbs(const cuDoubleComplex* v, int64_t n, int nrhs, double* out);

// --- double-double refinement: x = xHi + xLo, every row of the symmetric A whole ---

// A symmetric matrix whose values are stored once, in its upper CSR (the driver's arrays), plus an index of
// its strictly lower part: row i's lower entries are the upper entries (j, i), j < i, at columns
// colL[rowL[i] .. rowL[i + 1]) in ascending order, with values valU[srcL[...]]. The kernels below see row i
// as its lower part followed by its upper row, the layout of a full CSR, without a second copy of the values.
struct SymCsr
{
    int64_t n = 0;
    const int32_t *rowU = nullptr, *colU = nullptr;
    const cuDoubleComplex* valU = nullptr;
    const int64_t* rowL = nullptr;
    const int32_t *colL = nullptr, *srcL = nullptr;
};

// r = b - A (xHi + xLo) accumulated in double-double and rounded to double, nrhs columns.
// Pass xLo = nullptr to get the accurately computed residual of xHi alone.
void residualDD(const SymCsr& A, const cuDoubleComplex* xHi, const cuDoubleComplex* xLo, const cuDoubleComplex* b,
                cuDoubleComplex* r, int nrhs);

// y = A x for one column, products accumulated in double-double and rounded to double (accurate
// even when |A||x| is much larger than |A x|)
void spmvDD(const SymCsr& A, const cuDoubleComplex* x, cuDoubleComplex* y);

// Plain FP64 y = A x and r = b - A x for ncol / nrhs columns at stride n (whole rows: no atomics, unlike the
// upper-storage versions above), two columns per pass over A. With two columns and a scratch of 2 n values,
// x is interleaved first, so each x_j the rows gather is one 32-byte sector instead of two.
void spmvSym(const SymCsr& A, const cuDoubleComplex* x, cuDoubleComplex* y, int ncol = 1, cuDoubleComplex* scratch = nullptr);
void residualSym(const SymCsr& A, const cuDoubleComplex* x, const cuDoubleComplex* b, cuDoubleComplex* r, int nrhs,
                 cuDoubleComplex* scratch = nullptr);


// dst[c * P + k] = src[c * n + rows[k]] for c < ncol, k < P (values at a few rows, e.g. the ports)
void gatherRows(const cuDoubleComplex* src, int64_t n, int ncol, const int64_t* rows, int64_t P,
                cuDoubleComplex* dst);

// (xHi, xLo) += d in double-double, k < n
void accumulateDD(cuDoubleComplex* xHi, cuDoubleComplex* xLo, const cuDoubleComplex* d, int64_t n);

// Gram-Schmidt over vectors that need not be contiguous: vecs is a device array of nb pointers, and vector i
// is vecs[i] + off, k < n (off picks a column of n x ncol blocks). Deterministic (fixed summation order).
// h[i] = vecs_i^H w; partial: device scratch of multiDotScratch(nb) values
size_t multiDotScratch(int nb);
void multiDotc(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* w, int64_t n,
               cuDoubleComplex* h, cuDoubleComplex* partial);
// w += alpha * sum_i vecs_i h[i] (h on the device)
void multiAxpy(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* h, double alpha,
               cuDoubleComplex* w, int64_t n);

#endif
