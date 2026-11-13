#include "bench_kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define KCHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

static int blocks(int64_t n, int t) { return (int)((n + t - 1) / t); }

__global__ void convertKer(const cuDoubleComplex* in, cuComplex* out, int64_t n, double s)
{
    int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (k < n) out[k] = make_cuComplex((float)(in[k].x * s), (float)(in[k].y * s));
}

__global__ void axpyKer(cuDoubleComplex* x, const cuComplex* d, int64_t n, double s)
{
    int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (k < n) {
        x[k].x += s * d[k].x;
        x[k].y += s * d[k].y;
    }
}

// One thread per row: the stored upper part contributes a_ij x_j to row i, and the
// mirrored lower part a_ij x_i to row j (atomics). r must already hold b.
__global__ void symUpperKer(int64_t n, const int32_t* rowPtr, const int32_t* col, const cuDoubleComplex* val,
                            const cuDoubleComplex* x, cuDoubleComplex* r)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const cuDoubleComplex xi = x[i];
    double accRe = 0, accIm = 0;
    for (int32_t q = rowPtr[i]; q < rowPtr[i + 1]; ++q) {
        const int32_t j = col[q];
        const cuDoubleComplex a = val[q];
        const cuDoubleComplex xj = x[j];
        accRe += a.x * xj.x - a.y * xj.y;
        accIm += a.x * xj.y + a.y * xj.x;
        if (j != i) {
            atomicAdd(&r[j].x, -(a.x * xi.x - a.y * xi.y));
            atomicAdd(&r[j].y, -(a.x * xi.y + a.y * xi.x));
        }
    }
    atomicAdd(&r[i].x, -accRe);
    atomicAdd(&r[i].y, -accIm);
}

// sum of |v_k|^2: one partial sum per block (partial[blockIdx.x]), added up by sumBlocksKer in a fixed
// order, so the norm is the same on every run (an atomicAdd per block made FGMRES's histories differ)
__global__ void normKer(const cuDoubleComplex* v, int64_t n, double* partial)
{
    __shared__ double part[256];
    double s = 0;
    for (int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; k < n; k += (int64_t)gridDim.x * blockDim.x)
        s += v[k].x * v[k].x + v[k].y * v[k].y;
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = part[0];
}

// out[blockIdx.x] = sum of partial[blockIdx.x * nblk .. + nblk), in a fixed order
__global__ void sumBlocksKer(const double* partial, int nblk, double* out)
{
    __shared__ double part[256];
    double s = 0;
    for (int k = threadIdx.x; k < nblk; k += blockDim.x) s += partial[(int64_t)blockIdx.x * nblk + k];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}

void convertToC32(const cuDoubleComplex* in, cuComplex* out, int64_t n, double scale)
{
    convertKer<<<blocks(n, 256), 256>>>(in, out, n, scale);
    KCHECK(cudaGetLastError());
}

void axpyFromC32(cuDoubleComplex* x, const cuComplex* d, int64_t n, double scale)
{
    axpyKer<<<blocks(n, 256), 256>>>(x, d, n, scale);
    KCHECK(cudaGetLastError());
}

__global__ void toC64Ker(const cuComplex* in, cuDoubleComplex* out, int64_t n)
{
    int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (k < n) out[k] = make_cuDoubleComplex(in[k].x, in[k].y);
}

__global__ void negateKer(cuDoubleComplex* y, int64_t n)
{
    int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (k < n) y[k] = make_cuDoubleComplex(-y[k].x, -y[k].y);
}

void convertToC64(const cuComplex* in, cuDoubleComplex* out, int64_t n)
{
    toC64Ker<<<blocks(n, 256), 256>>>(in, out, n);
    KCHECK(cudaGetLastError());
}

void spmvSymUpper(int64_t n, const int32_t* rowPtr, const int32_t* col, const cuDoubleComplex* val,
                  const cuDoubleComplex* x, cuDoubleComplex* y)
{
    // symUpperKer accumulates r -= A x, so start from zero and flip the sign
    KCHECK(cudaMemset(y, 0, n * sizeof(cuDoubleComplex)));
    symUpperKer<<<blocks(n, 256), 256>>>(n, rowPtr, col, val, x, y);
    negateKer<<<blocks(n, 256), 256>>>(y, n);
    KCHECK(cudaGetLastError());
}

void residualSymUpper(int64_t n, const int32_t* rowPtr, const int32_t* col, const cuDoubleComplex* val,
                      const cuDoubleComplex* x, const cuDoubleComplex* b, cuDoubleComplex* r, int nrhs)
{
    KCHECK(cudaMemcpy(r, b, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    for (int c = 0; c < nrhs; ++c)
        symUpperKer<<<blocks(n, 256), 256>>>(n, rowPtr, col, val, x + c * n, r + c * n);
    KCHECK(cudaGetLastError());
}

// ---- double-double arithmetic (error-free transformations, FMA based) ----
struct DD { double h, l; };

__device__ __forceinline__ DD ddAdd(DD a, DD b)
{
    double s = a.h + b.h, bb = s - a.h;
    double e = (a.h - (s - bb)) + (b.h - bb) + a.l + b.l;
    double h = s + e;
    return {h, e - (h - s)};
}

// a * (xh + xl) as a double-double
__device__ __forceinline__ DD ddMul(double a, double xh, double xl)
{
    double p = a * xh;
    double e = __fma_rn(a, xh, -p) + a * xl;
    double h = p + e;
    return {h, e - (h - p)};
}

// The whole-row kernels below give each row kLanes threads that stride over its entries and then combine
// their partial sums with shuffles: rows have about 15 entries, so one thread per row reads the matrix
// with poor coalescing. Double-double sums are accurate in any order.
constexpr int kLanes = 8;

// Row i of a SymCsr as one sequence: its lower entries, then its upper row
struct SymRow
{
    int64_t lb, ub;  // first lower entry, first upper entry
    int nl, len;     // lower entries, all entries
};

__device__ __forceinline__ SymRow symRow(const SymCsr& A, int64_t i)
{
    const int64_t lb = A.rowL[i], ub = A.rowU[i];
    const int nl = (int)(A.rowL[i + 1] - lb);
    return {lb, ub, nl, nl + (int)(A.rowU[i + 1] - ub)};
}

__device__ __forceinline__ void symEntry(const SymCsr& A, const SymRow& r, int k, int32_t& j, cuDoubleComplex& a)
{
    if (k < r.nl) {
        j = A.colL[r.lb + k];
        a = A.valU[A.srcL[r.lb + k]];
    } else {
        j = A.colU[r.ub + k - r.nl];
        a = A.valU[r.ub + k - r.nl];
    }
}

// Compensated accumulation (Ogita, Rump, Oishi, "Dot2"): every product is split exactly with an FMA,
// every sum with TwoSum, and all the rounding errors go to one compensation term. The result is as
// accurate as a sum in twice the working precision, with about two thirds of the operations of
// renormalizing a double-double after each term.
struct Acc { double s, c; };

__device__ __forceinline__ void accAdd(Acc& a, double p, double e)  // a += p (+ e, already small)
{
    const double s = a.s + p, z = s - a.s;
    a.c += ((a.s - (s - z)) + (p - z)) + e;
    a.s = s;
}

// a += m * (xh + xl): m * xh exactly (product and FMA error), m * xl rounded (it is ~1e-16 of the rest)
__device__ __forceinline__ void accMul(Acc& a, double m, double xh, double xl)
{
    const double p = m * xh;
    accAdd(a, p, __fma_rn(m, xh, -p) + m * xl);
}

__device__ __forceinline__ Acc accShflDown(Acc a, int o)
{
    return {__shfl_down_sync(0xffffffffu, a.s, o, kLanes), __shfl_down_sync(0xffffffffu, a.c, o, kLanes)};
}

__device__ __forceinline__ void accMerge(Acc& a, Acc b)
{
    accAdd(a, b.s, b.c);
}

// r = b - A (xHi + xLo) for ncol (1 or 2) columns at stride n, sharing the matrix reads
__global__ void residualDDKer(const SymCsr A, const cuDoubleComplex* xHi, const cuDoubleComplex* xLo,
                              const cuDoubleComplex* b, cuDoubleComplex* r, int ncol)
{
    const int64_t n = A.n;
    const int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) / kLanes;
    const int lane = threadIdx.x % kLanes;
    Acc re[2] = {{0, 0}, {0, 0}}, im[2] = {{0, 0}, {0, 0}};
    if (i < n) {
        if (lane == 0)
            for (int c = 0; c < ncol; ++c) {
                re[c].s = b[c * n + i].x;
                im[c].s = b[c * n + i].y;
            }
        const SymRow row = symRow(A, i);
        for (int k = lane; k < row.len; k += kLanes) {
            int32_t j;
            cuDoubleComplex a;
            symEntry(A, row, k, j, a);
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                if (c >= ncol) break;
                const cuDoubleComplex xh = xHi[c * n + j];
                const cuDoubleComplex xl = xLo ? xLo[c * n + j] : make_cuDoubleComplex(0, 0);
                // (a.x + i a.y)(xr + i xi) = (a.x xr - a.y xi) + i (a.x xi + a.y xr)
                accMul(re[c], -a.x, xh.x, xl.x);
                accMul(re[c], a.y, xh.y, xl.y);
                accMul(im[c], -a.x, xh.y, xl.y);
                accMul(im[c], -a.y, xh.x, xl.x);
            }
        }
    }
#pragma unroll
    for (int o = kLanes / 2; o > 0; o >>= 1)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            accMerge(re[c], accShflDown(re[c], o));
            accMerge(im[c], accShflDown(im[c], o));
        }
    if (i < n && lane == 0)
        for (int c = 0; c < ncol; ++c) r[c * n + i] = make_cuDoubleComplex(re[c].s + re[c].c, im[c].s + im[c].c);
}

__global__ void spmvDDKer(const SymCsr A, const cuDoubleComplex* x, cuDoubleComplex* y)
{
    const int64_t n = A.n;
    const int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) / kLanes;
    const int lane = threadIdx.x % kLanes;
    Acc re = {0, 0}, im = {0, 0};
    if (i < n) {
        const SymRow row = symRow(A, i);
        for (int k = lane; k < row.len; k += kLanes) {
            int32_t j;
            cuDoubleComplex a;
            symEntry(A, row, k, j, a);
            const cuDoubleComplex xj = x[j];
            accMul(re, a.x, xj.x, 0.0);
            accMul(re, -a.y, xj.y, 0.0);
            accMul(im, a.x, xj.y, 0.0);
            accMul(im, a.y, xj.x, 0.0);
        }
    }
#pragma unroll
    for (int o = kLanes / 2; o > 0; o >>= 1) {
        accMerge(re, accShflDown(re, o));
        accMerge(im, accShflDown(im, o));
    }
    if (i < n && lane == 0) y[i] = make_cuDoubleComplex(re.s + re.c, im.s + im.c);
}

// Plain FP64 y = A x (residual = false) or y = b - A x (residual = true) for ncol (1 or 2) columns at stride n,
// whole rows, sharing the matrix reads (the lower part's values are scattered reads, the expensive part).
// Inter: x is interleaved (x[2 j + c], two columns), so the gather of x_j is one 32-byte sector
template <bool Residual, bool Inter = false>
__global__ void symKer(const SymCsr A, const cuDoubleComplex* x, const cuDoubleComplex* b, cuDoubleComplex* y, int ncol)
{
    const int64_t n = A.n;
    const int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) / kLanes;
    const int lane = threadIdx.x % kLanes;
    double re[2] = {0, 0}, im[2] = {0, 0};
    if (i < n) {
        const SymRow row = symRow(A, i);
        for (int k = lane; k < row.len; k += kLanes) {
            int32_t j;
            cuDoubleComplex a;
            symEntry(A, row, k, j, a);
            if constexpr (Inter) {
                const double4 xj = reinterpret_cast<const double4*>(x)[j];  // both columns at once
                re[0] += a.x * xj.x - a.y * xj.y;
                im[0] += a.x * xj.y + a.y * xj.x;
                re[1] += a.x * xj.z - a.y * xj.w;
                im[1] += a.x * xj.w + a.y * xj.z;
            } else {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    if (c >= ncol) break;
                    const cuDoubleComplex xj = x[c * n + j];
                    re[c] += a.x * xj.x - a.y * xj.y;
                    im[c] += a.x * xj.y + a.y * xj.x;
                }
            }
        }
    }
#pragma unroll
    for (int o = kLanes / 2; o > 0; o >>= 1)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            re[c] += __shfl_down_sync(0xffffffffu, re[c], o, kLanes);
            im[c] += __shfl_down_sync(0xffffffffu, im[c], o, kLanes);
        }
    if (i < n && lane == 0)
        for (int c = 0; c < ncol; ++c)
            y[c * n + i] = Residual ? make_cuDoubleComplex(b[c * n + i].x - re[c], b[c * n + i].y - im[c])
                                    : make_cuDoubleComplex(re[c], im[c]);
}

__global__ void interleave2Ker(const cuDoubleComplex* x, int64_t n, double4* xi)
{
    const int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (j < n) {
        const cuDoubleComplex a = x[j], b = x[n + j];
        xi[j] = make_double4(a.x, a.y, b.x, b.y);
    }
}

void spmvSym(const SymCsr& A, const cuDoubleComplex* x, cuDoubleComplex* y, int ncol, cuDoubleComplex* scratch)
{
    const int64_t n = A.n;
    if (ncol == 2 && scratch) {
        interleave2Ker<<<blocks(n, 256), 256>>>(x, n, reinterpret_cast<double4*>(scratch));
        symKer<false, true><<<blocks(n * kLanes, 256), 256>>>(A, scratch, nullptr, y, 2);
        KCHECK(cudaGetLastError());
        return;
    }
    for (int c = 0; c < ncol; c += 2)
        symKer<false><<<blocks(n * kLanes, 256), 256>>>(A, x + c * n, nullptr, y + c * n, ncol - c < 2 ? 1 : 2);
    KCHECK(cudaGetLastError());
}

void residualSym(const SymCsr& A, const cuDoubleComplex* x, const cuDoubleComplex* b, cuDoubleComplex* r, int nrhs,
                 cuDoubleComplex* scratch)
{
    const int64_t n = A.n;
    if (nrhs == 2 && scratch) {
        interleave2Ker<<<blocks(n, 256), 256>>>(x, n, reinterpret_cast<double4*>(scratch));
        symKer<true, true><<<blocks(n * kLanes, 256), 256>>>(A, scratch, b, r, 2);
        KCHECK(cudaGetLastError());
        return;
    }
    for (int c = 0; c < nrhs; c += 2)
        symKer<true><<<blocks(n * kLanes, 256), 256>>>(A, x + c * n, b + c * n, r + c * n, nrhs - c < 2 ? 1 : 2);
    KCHECK(cudaGetLastError());
}

void spmvDD(const SymCsr& A, const cuDoubleComplex* x, cuDoubleComplex* y)
{
    spmvDDKer<<<blocks(A.n * kLanes, 256), 256>>>(A, x, y);
    KCHECK(cudaGetLastError());
}

__global__ void accumulateDDKer(cuDoubleComplex* xHi, cuDoubleComplex* xLo, const cuDoubleComplex* d, int64_t n)
{
    int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    DD re = ddAdd({xHi[k].x, xLo[k].x}, {d[k].x, 0});
    DD im = ddAdd({xHi[k].y, xLo[k].y}, {d[k].y, 0});
    xHi[k] = make_cuDoubleComplex(re.h, im.h);
    xLo[k] = make_cuDoubleComplex(re.l, im.l);
}


__global__ void gatherRowsKer(const cuDoubleComplex* src, int64_t n, int ncol, const int64_t* rows, int64_t P,
                              cuDoubleComplex* dst)
{
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t < ncol * P) dst[t] = src[(t / P) * n + rows[t % P]];
}

void gatherRows(const cuDoubleComplex* src, int64_t n, int ncol, const int64_t* rows, int64_t P,
                cuDoubleComplex* dst)
{
    gatherRowsKer<<<blocks(ncol * P, 256), 256>>>(src, n, ncol, rows, P, dst);
    KCHECK(cudaGetLastError());
}

void residualDD(const SymCsr& A, const cuDoubleComplex* xHi, const cuDoubleComplex* xLo, const cuDoubleComplex* b,
                cuDoubleComplex* r, int nrhs)
{
    const int64_t n = A.n;
    for (int c = 0; c < nrhs; c += 2)
        residualDDKer<<<blocks(n * kLanes, 256), 256>>>(A, xHi + c * n, xLo ? xLo + c * n : nullptr, b + c * n,
                                                        r + c * n, nrhs - c < 2 ? 1 : 2);
    KCHECK(cudaGetLastError());
}

void accumulateDD(cuDoubleComplex* xHi, cuDoubleComplex* xLo, const cuDoubleComplex* d, int64_t n)
{
    accumulateDDKer<<<blocks(n, 256), 256>>>(xHi, xLo, d, n);
    KCHECK(cudaGetLastError());
}

// non-negative doubles order like their bit patterns, so atomicMax on the bits gives the max
__global__ void maxAbsKer(const cuDoubleComplex* v, int64_t n, unsigned long long* out)
{
    __shared__ double part[256];
    double m = 0;
    for (int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; k < n; k += (int64_t)gridDim.x * blockDim.x)
        m = fmax(m, hypot(v[k].x, v[k].y));
    part[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] = fmax(part[threadIdx.x], part[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out, (unsigned long long)__double_as_longlong(part[0]));
}

void columnMaxAbs(const cuDoubleComplex* v, int64_t n, int nrhs, double* out)
{
    unsigned long long* d;
    KCHECK(cudaMalloc(&d, nrhs * sizeof(unsigned long long)));
    KCHECK(cudaMemset(d, 0, nrhs * sizeof(unsigned long long)));
    for (int c = 0; c < nrhs; ++c) maxAbsKer<<<1024, 256>>>(v + c * n, n, d + c);
    std::vector<unsigned long long> h(nrhs);
    KCHECK(cudaMemcpy(h.data(), d, nrhs * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int c = 0; c < nrhs; ++c) std::memcpy(&out[c], &h[c], sizeof(double));
    KCHECK(cudaFree(d));
}

void columnNorms(const cuDoubleComplex* v, int64_t n, int nrhs, double* norms)
{
    constexpr int kBlocks = 1024;
    double* d;
    KCHECK(cudaMalloc(&d, (size_t)nrhs * (kBlocks + 1) * sizeof(double)));
    for (int c = 0; c < nrhs; ++c) normKer<<<kBlocks, 256>>>(v + c * n, n, d + (size_t)c * kBlocks);
    sumBlocksKer<<<nrhs, 256>>>(d, kBlocks, d + (size_t)nrhs * kBlocks);
    KCHECK(cudaGetLastError());
    KCHECK(cudaMemcpy(norms, d + (size_t)nrhs * kBlocks, nrhs * sizeof(double), cudaMemcpyDeviceToHost));
    for (int c = 0; c < nrhs; ++c) norms[c] = std::sqrt(norms[c]);
    KCHECK(cudaFree(d));
}

// h[i] = sum_k conj(vecs[i][off + k]) w[k]: G vectors per blockIdx.y, w re-read once per group; partial sums
// per block, then summed in a fixed order, so the result does not depend on the schedule
namespace {
constexpr int kDotGroup = 8, kDotBlocks = 1024, kDotThreads = 256;

__global__ void multiDotKer(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* __restrict__ w,
                            int64_t n, cuDoubleComplex* partial)
{
    const int g0 = blockIdx.y * kDotGroup;
    const int g = min(kDotGroup, nb - g0);
    const cuDoubleComplex* v[kDotGroup];
    double re[kDotGroup], im[kDotGroup];
#pragma unroll
    for (int i = 0; i < kDotGroup; ++i) {
        v[i] = i < g ? vecs[g0 + i] + off : nullptr;
        re[i] = im[i] = 0;
    }
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < n; r += (int64_t)gridDim.x * blockDim.x) {
        const cuDoubleComplex x = w[r];
#pragma unroll
        for (int i = 0; i < kDotGroup; ++i)
            if (i < g) {
                const cuDoubleComplex a = v[i][r];
                re[i] = fma(a.x, x.x, fma(a.y, x.y, re[i]));
                im[i] = fma(a.x, x.y, fma(-a.y, x.x, im[i]));
            }
    }
    __shared__ double sre[kDotGroup][kDotThreads / 32], sim[kDotGroup][kDotThreads / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < kDotGroup; ++i) {
        for (int o = 16; o > 0; o >>= 1) {
            re[i] += __shfl_xor_sync(0xffffffffu, re[i], o);
            im[i] += __shfl_xor_sync(0xffffffffu, im[i], o);
        }
        if (lane == 0) {
            sre[i][warp] = re[i];
            sim[i][warp] = im[i];
        }
    }
    __syncthreads();
    if (threadIdx.x < g) {
        double a = 0, b = 0;
        for (int k = 0; k < kDotThreads / 32; ++k) {
            a += sre[threadIdx.x][k];
            b += sim[threadIdx.x][k];
        }
        partial[(int64_t)(g0 + threadIdx.x) * gridDim.x + blockIdx.x] = make_cuDoubleComplex(a, b);
    }
}

__global__ void sumPartialKer(const cuDoubleComplex* partial, int nblk, cuDoubleComplex* h)
{
    __shared__ double sa[kDotThreads], sb[kDotThreads];
    double a = 0, b = 0;
    for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
        const cuDoubleComplex p = partial[(int64_t)blockIdx.x * nblk + k];
        a += p.x;
        b += p.y;
    }
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sa[threadIdx.x] += sa[threadIdx.x + s];
            sb[threadIdx.x] += sb[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) h[blockIdx.x] = make_cuDoubleComplex(sa[0], sb[0]);
}

__global__ void multiAxpyKer(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* __restrict__ h,
                             double alpha, cuDoubleComplex* w, int64_t n)
{
    for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < n; r += (int64_t)gridDim.x * blockDim.x) {
        double a = 0, b = 0;
        for (int i = 0; i < nb; ++i) {
            const cuDoubleComplex v = vecs[i][off + r], c = h[i];
            a = fma(v.x, c.x, fma(-v.y, c.y, a));
            b = fma(v.x, c.y, fma(v.y, c.x, b));
        }
        cuDoubleComplex x = w[r];
        x.x = fma(alpha, a, x.x);
        x.y = fma(alpha, b, x.y);
        w[r] = x;
    }
}
}  // namespace

size_t multiDotScratch(int nb) { return (size_t)((nb + kDotGroup - 1) / kDotGroup) * kDotGroup * kDotBlocks; }

void multiDotc(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* w, int64_t n,
               cuDoubleComplex* h, cuDoubleComplex* partial)
{
    if (nb <= 0) return;
    multiDotKer<<<dim3(kDotBlocks, (nb + kDotGroup - 1) / kDotGroup), kDotThreads>>>(vecs, nb, off, w, n, partial);
    KCHECK(cudaGetLastError());
    sumPartialKer<<<nb, kDotThreads>>>(partial, kDotBlocks, h);
    KCHECK(cudaGetLastError());
}

void multiAxpy(const cuDoubleComplex* const* vecs, int nb, int64_t off, const cuDoubleComplex* h, double alpha,
               cuDoubleComplex* w, int64_t n)
{
    if (nb <= 0) return;
    multiAxpyKer<<<2048, 256>>>(vecs, nb, off, h, alpha, w, n);
    KCHECK(cudaGetLastError());
}
