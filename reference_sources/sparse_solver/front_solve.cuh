// Supernodal triangular solve with the multifrontal L D L^T factors of ref_solve (complex single
// precision): x = P^T L^-T D^-1 L^-1 P b for nr right-hand sides (column-major, n x nr, ld n).
//
// The vectors and every sum are in the precision of V: cuComplex, or cuDoubleComplex with the panels
// widened as they are read (FrontSolve64). As FGMRES's preconditioner the single-precision solve was the
// weak part, not the factor: on a hard test system the harmonic Ritz values of A M^-1 apart from two outliers
// lay 0.03-0.17 from 1 with it, and within 0.025 (mostly 4e-3) with the same factor solved in double.
//
// Every supernode has its L panel (m x k, column-major, unit lower L below the diagonal, D on it) whose
// rows are its own k pivot columns followed by its m - k rows below them.
//   forward   children before parents. A supernode starts from its own entries of P b (zeros below), adds
//             its children's update vectors through relind (one child at a time, so the result does not
//             depend on timing), solves with its unit lower diagonal block and subtracts L21 z from the
//             rows below, which become its update vector; z / D replaces its entries of P b
//   backward  parents before children. A supernode gathers x at its rows below (urow: their pivot
//             numbers, final because they belong to ancestors), subtracts L21^T x from z / D and solves
//             with L11^T
// Both sweeps read every panel once. Maximal subtrees of fronts with m <= 32 run one warp each (one row
// per lane). The other supernodes below the large-front path run one CTA each, level by level (height in
// the tree), with the front vector in shared memory and the diagonal blocks solved 16 columns at a time
// by one warp. The large fronts (all their ancestors are large too) run on the
// factorization's stream pool, ordered by events, one kernel per front and sweep whose blocks of 64 rows
// or columns pass the solution on through flags.
#pragma once
#include "front_kernels.cuh"

#include <map>
#include <string>
#include <type_traits>
#include <vector>

struct SolveNode
{
    int m, k, col0;           // rows, pivot columns, first pivot column
    int childBeg, childEnd;   // into the children list
    int64_t panelOff;         // L panel
    int64_t relOff;           // relind / urow / update vector offset; -1 without a parent
};

template <class V>
struct SolveArgs
{
    const SolveNode* node;
    const int* children;
    const int* relind;
    const int* urow;
    const cuComplex* panels;
    V* u;     // update vectors, nr columns of relLen
    V* y;     // P b, then z / D (pivot order), nr columns of n
    V* x;     // solution in pivot order
    int64_t relLen, n;
};

constexpr int kSolveThreads = 128;

// More right-hand sides than NR run as groups of NR, one per blockIdx.y, each independent of the others
// (its own vectors, flags and tickets): group g takes columns NR g .. NR g + NR - 1 of every vector.
template <int NR, class V>
__device__ __forceinline__ SolveArgs<V> groupArgs(SolveArgs<V> a)
{
    const int64_t g = blockIdx.y;
    a.y += g * NR * a.n;
    a.x += g * NR * a.n;
    a.u += g * NR * a.relLen;
    return a;
}

__device__ __forceinline__ cuComplex cadd(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ cuComplex csub(cuComplex a, cuComplex b) { return make_cuComplex(a.x - b.x, a.y - b.y); }

// the same in double precision
__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ cuDoubleComplex csubmul(cuDoubleComplex acc, cuDoubleComplex a, cuDoubleComplex b)  // acc - a * b
{
    acc.x = fma(-a.x, b.x, fma(a.y, b.y, acc.x));
    acc.y = fma(-a.x, b.y, fma(-a.y, b.x, acc.y));
    return acc;
}
__device__ __forceinline__ cuDoubleComplex caddmul(cuDoubleComplex acc, cuDoubleComplex a, cuDoubleComplex b)  // acc + a * b
{
    acc.x = fma(a.x, b.x, fma(-a.y, b.y, acc.x));
    acc.y = fma(a.x, b.y, fma(a.y, b.x, acc.y));
    return acc;
}
__device__ __forceinline__ cuDoubleComplex cdiv(cuDoubleComplex a, cuDoubleComplex b)
{
    const double den = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / den, (a.y * b.x - a.x * b.y) / den);
}
__device__ __forceinline__ cuDoubleComplex shflC(cuDoubleComplex v, int src)
{
    return make_cuDoubleComplex(__shfl_sync(0xffffffffu, v.x, src), __shfl_sync(0xffffffffu, v.y, src));
}

// Complex double-float: each part is hi + lo, two floats (about 48 bits of mantissa), so the sums are
// nearly as accurate as in double while every operation is single precision (this GPU runs FP64 at
// 1/64 of the FP32 rate).
struct __align__(16) dfComplex
{
    float2 h, l;
};

__device__ __forceinline__ void twoSum(float a, float b, float& s, float& e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void fastTwoSum(float a, float b, float& s, float& e)  // |a| >= |b|
{
    s = a + b;
    e = b - (s - a);
}
__device__ __forceinline__ void dfAdd(float& ch, float& cl, float bh, float bl)  // (ch, cl) += (bh, bl)
{
    float s, e;
    twoSum(ch, bh, s, e);
    e += cl + bl;
    fastTwoSum(s, e, ch, cl);
}
__device__ __forceinline__ void dfMac(float& ch, float& cl, float a, float bh, float bl)  // (ch, cl) += a * (bh, bl)
{
    const float p = a * bh;
    float pe = fmaf(a, bh, -p);  // a * bh = p + pe exactly
    pe = fmaf(a, bl, pe);
    float s, e;
    twoSum(ch, p, s, e);
    e += cl + pe;
    fastTwoSum(s, e, ch, cl);
}
__device__ __forceinline__ dfComplex cadd(dfComplex a, dfComplex b)
{
    dfAdd(a.h.x, a.l.x, b.h.x, b.l.x);
    dfAdd(a.h.y, a.l.y, b.h.y, b.l.y);
    return a;
}
__device__ __forceinline__ dfComplex csub(dfComplex a, dfComplex b)
{
    dfAdd(a.h.x, a.l.x, -b.h.x, -b.l.x);
    dfAdd(a.h.y, a.l.y, -b.h.y, -b.l.y);
    return a;
}
// acc + s * a * b for a panel entry a (its low part is zero, or tiny: it multiplies b's high part)
__device__ __forceinline__ dfComplex cmacSigned(dfComplex acc, dfComplex a, dfComplex b, float s)
{
    const float ax = s * a.h.x, ay = s * a.h.y;
    dfMac(acc.h.x, acc.l.x, ax, b.h.x, b.l.x);
    dfMac(acc.h.x, acc.l.x, -ay, b.h.y, b.l.y);
    dfMac(acc.h.y, acc.l.y, ax, b.h.y, b.l.y);
    dfMac(acc.h.y, acc.l.y, ay, b.h.x, b.l.x);
    if (a.l.x != 0 || a.l.y != 0) {
        acc.l.x += s * (a.l.x * b.h.x - a.l.y * b.h.y);
        acc.l.y += s * (a.l.x * b.h.y + a.l.y * b.h.x);
    }
    return acc;
}
__device__ __forceinline__ dfComplex csubmul(dfComplex acc, dfComplex a, dfComplex b) { return cmacSigned(acc, a, b, -1.0f); }
__device__ __forceinline__ dfComplex caddmul(dfComplex acc, dfComplex a, dfComplex b) { return cmacSigned(acc, a, b, 1.0f); }
__device__ __forceinline__ cuDoubleComplex dfToDouble(dfComplex a)
{
    return make_cuDoubleComplex((double)a.h.x + a.l.x, (double)a.h.y + a.l.y);
}
__device__ __forceinline__ dfComplex dfFromDouble(cuDoubleComplex v)
{
    dfComplex r;
    r.h = make_float2((float)v.x, (float)v.y);
    r.l = make_float2((float)(v.x - r.h.x), (float)(v.y - r.h.y));
    return r;
}
__device__ __forceinline__ dfComplex cdiv(dfComplex a, dfComplex b)  // n of these per solve: in double
{
    return dfFromDouble(cdiv(dfToDouble(a), dfToDouble(b)));
}
__device__ __forceinline__ dfComplex shflC(dfComplex v, int src)
{
    dfComplex r;
    r.h = make_float2(__shfl_sync(0xffffffffu, v.h.x, src), __shfl_sync(0xffffffffu, v.h.y, src));
    r.l = make_float2(__shfl_sync(0xffffffffu, v.l.x, src), __shfl_sync(0xffffffffu, v.l.y, src));
    return r;
}

// the value of lane (lane ^ o)
__device__ __forceinline__ cuComplex shflXor(cuComplex v, int o)
{
    return make_cuComplex(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o));
}
__device__ __forceinline__ cuDoubleComplex shflXor(cuDoubleComplex v, int o)
{
    return make_cuDoubleComplex(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o));
}
__device__ __forceinline__ dfComplex shflXor(dfComplex v, int o)
{
    dfComplex r;
    r.h = make_float2(__shfl_xor_sync(0xffffffffu, v.h.x, o), __shfl_xor_sync(0xffffffffu, v.h.y, o));
    r.l = make_float2(__shfl_xor_sync(0xffffffffu, v.l.x, o), __shfl_xor_sync(0xffffffffu, v.l.y, o));
    return r;
}

// v + the value of lane (lane ^ o), for warp sums
__device__ __forceinline__ cuComplex addXor(cuComplex v, int o)
{
    return make_cuComplex(v.x + __shfl_xor_sync(0xffffffffu, v.x, o), v.y + __shfl_xor_sync(0xffffffffu, v.y, o));
}
__device__ __forceinline__ cuDoubleComplex addXor(cuDoubleComplex v, int o)
{
    return make_cuDoubleComplex(v.x + __shfl_xor_sync(0xffffffffu, v.x, o), v.y + __shfl_xor_sync(0xffffffffu, v.y, o));
}
__device__ __forceinline__ dfComplex addXor(dfComplex v, int o)
{
    dfComplex w;
    w.h = make_float2(__shfl_xor_sync(0xffffffffu, v.h.x, o), __shfl_xor_sync(0xffffffffu, v.h.y, o));
    w.l = make_float2(__shfl_xor_sync(0xffffffffu, v.l.x, o), __shfl_xor_sync(0xffffffffu, v.l.y, o));
    return cadd(v, w);
}

// Development (REF_DF_SINGLE): the layout of dfComplex with single-precision arithmetic, so that chosen phases of a
// double-float solve run in single precision on the same buffers. Only h is used; results have l = 0.
struct __align__(16) sfComplex
{
    float2 h, l;
};
__device__ __forceinline__ sfComplex sf(cuComplex v) { return sfComplex{v, make_float2(0, 0)}; }
__device__ __forceinline__ sfComplex cadd(sfComplex a, sfComplex b) { return sf(cadd(a.h, b.h)); }
__device__ __forceinline__ sfComplex csub(sfComplex a, sfComplex b) { return sf(csub(a.h, b.h)); }
__device__ __forceinline__ sfComplex csubmul(sfComplex acc, sfComplex a, sfComplex b) { return sf(csubmul(acc.h, a.h, b.h)); }
__device__ __forceinline__ sfComplex caddmul(sfComplex acc, sfComplex a, sfComplex b) { return sf(caddmul(acc.h, a.h, b.h)); }
__device__ __forceinline__ sfComplex cdiv(sfComplex a, sfComplex b) { return sf(cdiv(a.h, b.h)); }
__device__ __forceinline__ sfComplex shflC(sfComplex v, int src) { return sf(shflC(v.h, src)); }
__device__ __forceinline__ sfComplex shflXor(sfComplex v, int o) { return sf(shflXor(v.h, o)); }
__device__ __forceinline__ sfComplex addXor(sfComplex v, int o) { return sf(addXor(v.h, o)); }

template <class V> __device__ __forceinline__ V vzero();
template <> __device__ __forceinline__ cuComplex vzero<cuComplex>() { return make_cuComplex(0, 0); }
template <> __device__ __forceinline__ cuDoubleComplex vzero<cuDoubleComplex>() { return make_cuDoubleComplex(0, 0); }
template <> __device__ __forceinline__ dfComplex vzero<dfComplex>() { return dfComplex{make_float2(0, 0), make_float2(0, 0)}; }
template <> __device__ __forceinline__ sfComplex vzero<sfComplex>() { return sf(make_cuComplex(0, 0)); }

// a panel entry in the vectors' precision
template <class V> __device__ __forceinline__ V widen(cuComplex p);
template <> __device__ __forceinline__ cuComplex widen<cuComplex>(cuComplex p) { return p; }
template <> __device__ __forceinline__ cuDoubleComplex widen<cuDoubleComplex>(cuComplex p) { return make_cuDoubleComplex(p.x, p.y); }
template <> __device__ __forceinline__ dfComplex widen<dfComplex>(cuComplex p) { return dfComplex{p, make_float2(0, 0)}; }
template <> __device__ __forceinline__ sfComplex widen<sfComplex>(cuComplex p) { return sf(p); }

// the solve's input and output vectors: single precision with single vectors, double otherwise
template <class V> using SolveIO = std::conditional_t<std::is_same_v<V, cuComplex>, cuComplex, cuDoubleComplex>;
template <class V> __device__ __forceinline__ V fromIO(SolveIO<V> v);
template <> __device__ __forceinline__ cuComplex fromIO<cuComplex>(cuComplex v) { return v; }
template <> __device__ __forceinline__ cuDoubleComplex fromIO<cuDoubleComplex>(cuDoubleComplex v) { return v; }
template <> __device__ __forceinline__ dfComplex fromIO<dfComplex>(cuDoubleComplex v) { return dfFromDouble(v); }
__device__ __forceinline__ cuComplex toIO(cuComplex v) { return v; }
__device__ __forceinline__ cuDoubleComplex toIO(cuDoubleComplex v) { return v; }
__device__ __forceinline__ cuDoubleComplex toIO(dfComplex v) { return dfToDouble(v); }

template <int NR, class V>
__global__ void __launch_bounds__(kSolveThreads) solveFwdKer(SolveArgs<V> a, const int* list, int count)
{
    a = groupArgs<NR>(a);
    extern __shared__ __align__(16) unsigned char solveSh[];
    V* w = reinterpret_cast<V*>(solveSh);  // front vector, m x NR, ld m
    const int tid = threadIdx.x, nt = blockDim.x;
    const V zero = vzero<V>();
    for (int it = blockIdx.x; it < count; it += gridDim.x) {
        const SolveNode x = a.node[list[it]];
        const int m = x.m, k = x.k;
        const cuComplex* __restrict__ P = a.panels + x.panelOff;
        for (int i = tid; i < m; i += nt)
            for (int c = 0; c < NR; ++c) w[c * m + i] = i < k ? a.y[c * a.n + x.col0 + i] : zero;
        __syncthreads();
        for (int ci = x.childBeg; ci < x.childEnd; ++ci) {  // one child at a time: positions may repeat
            const SolveNode ch = a.node[a.children[ci]];
            const int mc = ch.m - ch.k;
            for (int i = tid; i < mc; i += nt) {
                const int pos = a.relind[ch.relOff + i];
                for (int c = 0; c < NR; ++c) w[c * m + pos] = cadd(w[c * m + pos], a.u[c * a.relLen + ch.relOff + i]);
            }
            __syncthreads();
        }
        for (int jb = 0; jb < k; jb += 16) {
            const int jn = min(16, k - jb);
            if (tid < 32) {  // diagonal block, unit lower, lane = row (its row of the block in registers)
                V lr[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) lr[j] = tid < jn && j < tid ? widen<V>(P[(jb + tid) + (int64_t)m * (jb + j)]) : zero;
                V v[NR];
                for (int c = 0; c < NR; ++c) v[c] = tid < jn ? w[c * m + jb + tid] : zero;
#pragma unroll
                for (int j = 0; j < 15; ++j)
                    for (int c = 0; c < NR; ++c) {
                        const V zj = shflC(v[c], j);
                        if (tid > j) v[c] = csubmul(v[c], lr[j], zj);
                    }
                if (tid < jn)
                    for (int c = 0; c < NR; ++c) w[c * m + jb + tid] = v[c];
            }
            __syncthreads();
            for (int i = jb + jn + tid; i < m; i += nt) {  // rows below the block (L11 and L21)
                V acc[NR];
                cuComplex p[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) p[j] = j < jn ? P[i + (int64_t)m * (jb + j)] : make_cuComplex(0, 0);
                for (int c = 0; c < NR; ++c) acc[c] = w[c * m + i];
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < jn)
                        for (int c = 0; c < NR; ++c) acc[c] = csubmul(acc[c], widen<V>(p[j]), w[c * m + jb + j]);
                for (int c = 0; c < NR; ++c) w[c * m + i] = acc[c];
            }
            __syncthreads();
        }
        for (int i = tid; i < m; i += nt)
            for (int c = 0; c < NR; ++c) {
                if (i < k) a.y[c * a.n + x.col0 + i] = cdiv(w[c * m + i], widen<V>(P[i + (int64_t)m * i]));
                else if (x.relOff >= 0) a.u[c * a.relLen + x.relOff + (i - k)] = w[c * m + i];
            }
        __syncthreads();
    }
}

template <int NR, class V>
__global__ void __launch_bounds__(kSolveThreads) solveBwdKer(SolveArgs<V> a, const int* list, int count)
{
    a = groupArgs<NR>(a);
    extern __shared__ __align__(16) unsigned char solveSh[];
    V* w = reinterpret_cast<V*>(solveSh);
    const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
    const V zero = vzero<V>();
    for (int it = blockIdx.x; it < count; it += gridDim.x) {
        const SolveNode x = a.node[list[it]];
        const int m = x.m, k = x.k;
        const cuComplex* __restrict__ P = a.panels + x.panelOff;
        for (int i = tid; i < m; i += nt)
            for (int c = 0; c < NR; ++c)
                w[c * m + i] = i < k ? a.y[c * a.n + x.col0 + i] : a.x[c * a.n + a.urow[x.relOff + (i - k)]];
        __syncthreads();
        for (int jb = ((k - 1) / 16) * 16; jb >= 0; jb -= 16) {
            const int jn = min(16, k - jb);
            for (int j = warp; j < jn; j += nw) {  // column j of the block minus the solved rows below it
                V acc[NR];
                for (int c = 0; c < NR; ++c) acc[c] = zero;
                const cuComplex* col = P + (int64_t)m * (jb + j);
                for (int i = jb + jn + lane; i < m; i += 32) {
                    const V p = widen<V>(col[i]);
                    for (int c = 0; c < NR; ++c) acc[c] = caddmul(acc[c], p, w[c * m + i]);
                }
                for (int c = 0; c < NR; ++c)
                    for (int o = 16; o > 0; o >>= 1) acc[c] = addXor(acc[c], o);
                if (lane == 0)
                    for (int c = 0; c < NR; ++c) w[c * m + jb + j] = csub(w[c * m + jb + j], acc[c]);
            }
            __syncthreads();
            if (warp == 0) {  // diagonal block, transposed: lane = column (its column of the block in registers)
                V lc[16];
#pragma unroll
                for (int l = 0; l < 16; ++l)
                    lc[l] = lane < jn && l > lane && l < jn ? widen<V>(P[(jb + l) + (int64_t)m * (jb + lane)]) : zero;
                V v[NR];
                for (int c = 0; c < NR; ++c) v[c] = lane < jn ? w[c * m + jb + lane] : zero;
#pragma unroll
                for (int l = 15; l > 0; --l)
                    for (int c = 0; c < NR; ++c) {
                        const V xl = shflC(v[c], l);
                        if (lane < l) v[c] = csubmul(v[c], lc[l], xl);
                    }
                if (lane < jn)
                    for (int c = 0; c < NR; ++c) w[c * m + jb + lane] = v[c];
            }
            __syncthreads();
        }
        for (int i = tid; i < k; i += nt)
            for (int c = 0; c < NR; ++c) a.x[c * a.n + x.col0 + i] = w[c * m + i];
        __syncthreads();
    }
}

// ---------------- small subtrees and mid-size fronts: one warp per front
// A front with m <= 32 R and k <= 32 is solved by one warp, its rows spread over the lanes (lane + 32 s, s < R),
// so the warp needs no block barriers. Forward: k steps, each a shuffle of z_j from its row and one
// multiply-add per row below, the next panel column prefetched. Backward: the rows below the pivots first,
// four columns at a time (lanes over rows, then a transposed warp sum), then the k x k triangle with one
// column per lane, each lane reading its column one row ahead. The panel is read straight from global
// memory (staging it in shared memory cost occupancy: 8 warps per SM).
//   small subtrees  maximal subtrees whose fronts all have m <= 32 (R = 1): one warp each, in postorder
//                   (forward) or reverse postorder (backward); the update vectors inside the subtree, and x
//                   of its fronts, are written and read by the same warp. Before the levels (forward), after
//                   them (backward).
//   mid-size fronts the level fronts with m <= 128, k <= 32 (R = 4): one warp each, level by level
constexpr int kSmallSolveWarps = 4;
constexpr int kMidRows = 4;  // rows per lane of the mid-size fronts: m <= 128
constexpr int kMidK = 32;    // pivot columns: one row of the triangle per lane

template <int E, class V>
__device__ __forceinline__ V transposeReduce(V (&a)[E], int lane, int& e);

// forward of front r by one warp; w: the warp's shared front vector, NR x 32 R
template <int NR, int R, class V>
__device__ __forceinline__ void warpFrontFwd(const SolveArgs<V>& a, int r, V (*w)[32 * R], int lane)
{
    const V zero = vzero<V>();
    const SolveNode x = a.node[r];
    const int m = x.m, k = x.k;
    const cuComplex* __restrict__ P = a.panels + x.panelOff;
    for (int i = lane; i < m; i += 32)
        for (int c = 0; c < NR; ++c) w[c][i] = i < k ? a.y[c * a.n + x.col0 + i] : zero;
    __syncwarp();
    for (int ci = x.childBeg; ci < x.childEnd; ++ci) {  // one child at a time: positions may repeat
        const SolveNode ch = a.node[a.children[ci]];
        for (int i = lane; i < ch.m - ch.k; i += 32) {
            const int pos = a.relind[ch.relOff + i];
            for (int c = 0; c < NR; ++c) w[c][pos] = cadd(w[c][pos], a.u[c * a.relLen + ch.relOff + i]);
        }
        __syncwarp();
    }
    V v[R][NR];
    cuComplex l[R];  // column j at this lane's rows
#pragma unroll
    for (int s = 0; s < R; ++s) {
        const int i = lane + 32 * s;
        for (int c = 0; c < NR; ++c) v[s][c] = i < m ? w[c][i] : zero;
        l[s] = i < m ? P[i] : make_cuComplex(0, 0);
    }
    for (int j = 0; j < k; ++j) {  // z_j (row j: lane j, k <= 32) is final; the rows below subtract L(i, j) z_j
        cuComplex ln[R];
#pragma unroll
        for (int s = 0; s < R; ++s) {
            const int i = lane + 32 * s;
            ln[s] = j + 1 < k && i < m ? P[i + (int64_t)m * (j + 1)] : make_cuComplex(0, 0);
        }
        for (int c = 0; c < NR; ++c) {
            const V zj = shflC(v[0][c], j);
#pragma unroll
            for (int s = 0; s < R; ++s) {
                const int i = lane + 32 * s;
                if (i > j && i < m) v[s][c] = csubmul(v[s][c], widen<V>(l[s]), zj);
            }
        }
#pragma unroll
        for (int s = 0; s < R; ++s) l[s] = ln[s];
    }
#pragma unroll
    for (int s = 0; s < R; ++s) {
        const int i = lane + 32 * s;
        if (i < k) {
            const V d = widen<V>(P[i + (int64_t)m * i]);
            for (int c = 0; c < NR; ++c) a.y[c * a.n + x.col0 + i] = cdiv(v[s][c], d);
        } else if (i < m && x.relOff >= 0)
            for (int c = 0; c < NR; ++c) a.u[c * a.relLen + x.relOff + (i - k)] = v[s][c];
    }
    __syncwarp();
}

// backward of front r by one warp; sum: the warp's shared L21^T x per pivot column, 32 x NR
template <int NR, int R, class V>
__device__ __forceinline__ void warpFrontBwd(const SolveArgs<V>& a, int r, V (*sum)[NR], int lane)
{
    const V zero = vzero<V>();
    constexpr int C = 4, E = C * NR;                // columns per chunk, their entries
    const bool owner = (lane & (32 / E - 1)) == 0;  // one lane per entry after the transposed sum
    const SolveNode x = a.node[r];
    const int m = x.m, k = x.k;
    const cuComplex* __restrict__ P = a.panels + x.panelOff;
    V v[R][NR];
#pragma unroll
    for (int s = 0; s < R; ++s) {
        const int i = lane + 32 * s;
        for (int c = 0; c < NR; ++c)
            v[s][c] = i < k ? a.y[c * a.n + x.col0 + i] : i < m ? a.x[c * a.n + a.urow[x.relOff + (i - k)]] : zero;
    }
    for (int jb = 0; jb < k; jb += C) {  // the rows below: L21^T x for C columns at a time
        V acc[E];
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] = zero;
#pragma unroll
        for (int s = 0; s < R; ++s) {
            const int i = lane + 32 * s;
            if (i >= k && i < m)
#pragma unroll
                for (int jj = 0; jj < C; ++jj)
                    if (jb + jj < k) {
                        const V p = widen<V>(P[i + (int64_t)m * (jb + jj)]);
                        for (int c = 0; c < NR; ++c) acc[jj * NR + c] = caddmul(acc[jj * NR + c], p, v[s][c]);
                    }
        }
        int e;
        const V rs = transposeReduce<E>(acc, lane, e);
        if (owner && jb + e / NR < k) sum[jb + e / NR][e % NR] = rs;
    }
    __syncwarp();
    V rr[NR];  // lane j < k: its pivot's right-hand side, then x_j
    for (int c = 0; c < NR; ++c) rr[c] = lane < k ? csub(v[0][c], sum[lane][c]) : zero;
    // x_i is final: every column j < i subtracts L(i, j) x_i; lane j reads its column, one row ahead
    cuComplex l = k >= 2 && lane < k - 1 ? P[(k - 1) + (int64_t)m * lane] : make_cuComplex(0, 0);
    for (int i = k - 1; i >= 1; --i) {
        const cuComplex ln = i >= 2 && lane < i - 1 ? P[(i - 1) + (int64_t)m * lane] : make_cuComplex(0, 0);
        for (int c = 0; c < NR; ++c) {
            const V xi = shflC(rr[c], i);
            if (lane < i) rr[c] = csubmul(rr[c], widen<V>(l), xi);
        }
        l = ln;
    }
    if (lane < k)
        for (int c = 0; c < NR; ++c) a.x[c * a.n + x.col0 + lane] = rr[c];
    __syncwarp();
}

template <int NR, class V>
__global__ void __launch_bounds__(32 * kSmallSolveWarps) smallFwdKer(SolveArgs<V> a, const int2* sub, int count, int* next)
{
    a = groupArgs<NR>(a);
    next += blockIdx.y;  // the group's work counter
    __shared__ V ws[kSmallSolveWarps][NR][32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (;;) {
        int t = 0;
        if (lane == 0) t = atomicAdd(next, 1);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= count) return;
        const int2 range = sub[t];
        for (int r = range.x; r <= range.y; ++r) warpFrontFwd<NR, 1, V>(a, r, ws[wid], lane);
    }
}

template <int NR, class V>
__global__ void __launch_bounds__(32 * kSmallSolveWarps) smallBwdKer(SolveArgs<V> a, const int2* sub, int count, int* next)
{
    a = groupArgs<NR>(a);
    next += blockIdx.y;
    __shared__ V ss[kSmallSolveWarps][kMidK][NR];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (;;) {
        int t = 0;
        if (lane == 0) t = atomicAdd(next, 1);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= count) return;
        const int2 range = sub[t];
        for (int r = range.y; r >= range.x; --r) warpFrontBwd<NR, 1, V>(a, r, ss[wid], lane);  // parents first
    }
}

template <int NR, class V>
__global__ void __launch_bounds__(32 * kSmallSolveWarps) midFwdKer(SolveArgs<V> a, const int* list, int count)
{
    a = groupArgs<NR>(a);
    __shared__ V ws[kSmallSolveWarps][NR][32 * kMidRows];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int t = blockIdx.x * kSmallSolveWarps + wid; t < count; t += gridDim.x * kSmallSolveWarps)
        warpFrontFwd<NR, kMidRows, V>(a, list[t], ws[wid], lane);
}

template <int NR, class V>
__global__ void __launch_bounds__(32 * kSmallSolveWarps) midBwdKer(SolveArgs<V> a, const int* list, int count)
{
    a = groupArgs<NR>(a);
    __shared__ V ss[kSmallSolveWarps][kMidK][NR];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int t = blockIdx.x * kSmallSolveWarps + wid; t < count; t += gridDim.x * kSmallSolveWarps)
        warpFrontBwd<NR, kMidRows, V>(a, list[t], ss[wid], lane);
}

// Sums of E values over the 32 lanes of a warp, every lane holding all E (E a power of two, at most 32):
// at each shuffle level a lane keeps half of its values and adds its partner's other half, so the levels
// move E/2 + E/4 + ... values instead of E each. Returns the sum of entry e (set here) of this lane; when
// E < 32, the lanes that differ only in the low bits hold the same entry. The order of the sums is fixed.
template <int E, class V>
__device__ __forceinline__ V transposeReduce(V (&a)[E], int lane, int& e)
{
    static_assert(E >= 1 && E <= 32 && (E & (E - 1)) == 0, "E: a power of two up to 32");
    e = 0;
#pragma unroll
    for (int o = 16, cur = E; o > 0; o >>= 1) {
        const bool bit = (lane & o) != 0;
        if (cur > 1) {
            const int half = cur / 2;
#pragma unroll
            for (int q = 0; q < 16; ++q)
                if (q < half) {
                    const V send = bit ? a[q] : a[q + half], keep = bit ? a[q + half] : a[q];
                    a[q] = cadd(keep, shflXor(send, o));
                }
            if (bit) e += half;
            cur = half;
        } else {
            a[0] = cadd(a[0], shflXor(a[0], o));
        }
    }
    return a[0];
}

// The large fronts' 64 x 64 unit lower diagonal blocks, inverted once per factorization so that the solves
// apply them as products (every thread busy) instead of 63 dependent steps of one warp.
struct InvBlock
{
    int64_t panelOff, off;  // the front's panel; this block's inverse in the inverse storage
    int m, c0, cn, pad;     // panel rows, first column of the block, its columns
};

// inverse of block blockIdx.x into linv (column-major 64 x 64, zeros above the diagonal and outside cn),
// in V: thread j computes column j by forward substitution
template <class V>
__global__ void __launch_bounds__(64) bigInvKer(const cuComplex* __restrict__ panels, const InvBlock* blocks, V* linv)
{
    extern __shared__ __align__(16) unsigned char invSh[];
    V (*X)[65] = reinterpret_cast<V (*)[65]>(invSh);  // X[i][j]
    const InvBlock b = blocks[blockIdx.x];
    const int j = threadIdx.x;
    const cuComplex* L = panels + b.panelOff + b.c0 + (int64_t)b.m * b.c0;  // L(i, l) = L[i + m l]
    const V zero = vzero<V>();
    for (int i = 0; i < 64; ++i) X[i][j] = i == j && j < b.cn ? widen<V>(make_cuComplex(1, 0)) : zero;
    if (j < b.cn)
        for (int i = j + 1; i < b.cn; ++i) {
            V acc = zero;
            for (int l = j; l < i; ++l) acc = caddmul(acc, widen<V>(L[i + (int64_t)b.m * l]), X[l][j]);
            X[i][j] = csub(zero, acc);
        }
    __syncthreads();
    for (int e = j; e < 64 * 64; e += 64) linv[b.off + e] = X[e & 63][e >> 6];
}

// ---------------- large fronts: one kernel per front and sweep, without grid synchronization
// The front vector w (m x nr, ld m) lives in global memory. Blocks of 64 rows (forward) or 64 columns
// (backward) are handed out in ticket order (an atomic counter), so a block only ever waits for blocks
// that are already running; a finished diagonal block publishes its part of the solution and sets its
// flag to 1. Every sweep starts from zeroed flags (a memset in the sweep), so a captured sweep can be replayed.
constexpr int kBigThreads = 256;

__device__ __forceinline__ void waitValue(const int* p, int v)  // polls with a short backoff
{
    while (*(volatile const int*)p != v) __nanosleep(64);
    __threadfence();
}

__device__ __forceinline__ void waitFlag(const int* flag) { waitValue(flag, 1); }

__device__ __forceinline__ cuComplex ldcg(const cuComplex* p)  // around L1: published by another SM
{
    const float2 v = __ldcg(reinterpret_cast<const float2*>(p));
    return make_cuComplex(v.x, v.y);
}
__device__ __forceinline__ cuDoubleComplex ldcg(const cuDoubleComplex* p)
{
    const double2 v = __ldcg(reinterpret_cast<const double2*>(p));
    return make_cuDoubleComplex(v.x, v.y);
}
__device__ __forceinline__ dfComplex ldcg(const dfComplex* p)
{
    const float4 v = __ldcg(reinterpret_cast<const float4*>(p));
    return dfComplex{make_float2(v.x, v.y), make_float2(v.z, v.w)};
}
__device__ __forceinline__ sfComplex ldcg(const sfComplex* p)
{
    const float2 v = __ldcg(reinterpret_cast<const float2*>(p));
    return sf(make_cuComplex(v.x, v.y));
}

// forward assembly: own entries of P b, zeros below, then the children's update vectors one child at a
// time (one CTA: a fixed order of the sums); resets the ticket
template <class V>
__global__ void bigFwdAssembleKer(V* w, int m, int k, const V* y, int64_t n, int nr, int col0, const SolveNode* node,
                                  const int* children, int cb, int ce, const int64_t* wOff, const V* wBase, const V* u,
                                  int64_t relLen, const int* relind, int* ticket, int tStride)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int64_t g = blockIdx.y;  // right-hand-side group: columns nr g..
    w += g * nr * m;
    y += g * nr * n;
    u += g * nr * relLen;
    ticket += g * tStride;
    for (int i = tid; i < m; i += nt)
        for (int c = 0; c < nr; ++c) w[(int64_t)c * m + i] = i < k ? y[c * n + col0 + i] : vzero<V>();
    __syncthreads();
    for (int ci = cb; ci < ce; ++ci) {
        const int ch = children[ci];
        const SolveNode y2 = node[ch];
        const int mc = y2.m - y2.k;
        const bool big = wOff[ch] >= 0;  // a large child keeps its update vector below its own rows
        const int64_t lds = big ? y2.m : relLen;
        const V* src = big ? wBase + wOff[ch] + g * nr * lds + y2.k : u + y2.relOff;
        for (int i = tid; i < mc; i += nt) {
            const int pos = relind[y2.relOff + i];
            for (int c = 0; c < nr; ++c) w[(int64_t)c * m + pos] = cadd(w[(int64_t)c * m + pos], src[c * lds + i]);
        }
        __syncthreads();
    }
    if (tid == 0) *ticket = 0;
}

// Forward: row blocks of the diagonal part first (ticket t < kb: rows 64t.., which need the column blocks
// before them, then solve their diagonal block and publish z), then row blocks of L21 (need every column
// block; they leave the update vector in w). 4 threads per row, 16 columns of a column block each.
template <int NR, class V>
__global__ void __launch_bounds__(kBigThreads) bigFwdKer(V* w, const cuComplex* __restrict__ P, const V* __restrict__ Li, int m,
                                                         int k, V* y, int64_t n, int col0, int* ticket, int* flag, int tStride,
                                                         int fStride)
{
    {
        const int64_t g = blockIdx.y;  // right-hand-side group
        w += g * NR * m;
        y += g * NR * n;
        ticket += g * tStride;
        flag += g * fStride;
    }
    __shared__ int sT;
    __shared__ V sz[64][NR];  // z of a column block; then this block's right-hand side r
    __shared__ V sacc[4][64][NR];
    const int tid = threadIdx.x, row = tid & 63, quarter = tid >> 6;
    if (tid == 0) sT = atomicAdd(ticket, 1);
    __syncthreads();
    const int t = sT, kb = (k + 63) / 64;
    const bool diag = t < kb;
    const int r0 = diag ? t * 64 : k + (t - kb) * 64, r1 = min(r0 + 64, diag ? k : m), ncb = diag ? t : kb;
    const int i = r0 + row;
    V li[16];  // diagonal tickets: row `row`, columns quarter * 16.. of the block's inverse, fetched before waiting
    cuComplex d = make_cuComplex(1, 0);
    if (diag) {
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) li[jj] = Li[(int64_t)t * 4096 + row + 64 * (quarter * 16 + jj)];
        if (i < r1) d = P[i + (int64_t)m * i];
    }
    V acc[NR];
    for (int c = 0; c < NR; ++c) acc[c] = vzero<V>();
    for (int cb = 0; cb < ncb; ++cb) {
        const int c0 = cb * 64, cn = min(64, k - c0);
        cuComplex p[16];  // this thread's part of the block, loaded before waiting for its z
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int j = quarter * 16 + jj;
            p[jj] = i < r1 && j < cn ? P[i + (int64_t)m * (c0 + j)] : make_cuComplex(0, 0);
        }
        if (tid == 0) waitFlag(flag + cb);
        __syncthreads();
        for (int e = tid; e < cn * NR; e += kBigThreads) sz[e % cn][e / cn] = ldcg(w + (int64_t)(e / cn) * m + c0 + e % cn);
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < 16; ++jj)
            if (quarter * 16 + jj < cn)
                for (int c = 0; c < NR; ++c) acc[c] = caddmul(acc[c], widen<V>(p[jj]), sz[quarter * 16 + jj][c]);
        __syncthreads();
    }
    for (int c = 0; c < NR; ++c) sacc[quarter][row][c] = acc[c];
    __syncthreads();
    if (tid < 64 && i < r1)
        for (int c = 0; c < NR; ++c) {
            const V s = cadd(cadd(sacc[0][row][c], sacc[1][row][c]), cadd(sacc[2][row][c], sacc[3][row][c]));
            sz[row][c] = csub(w[(int64_t)c * m + i], s);
        }
    __syncthreads();
    if (!diag) {
        if (tid < 64 && i < r1)
            for (int c = 0; c < NR; ++c) w[(int64_t)c * m + i] = sz[row][c];
        return;
    }
    // the diagonal block: z = L11^-1 r, each thread 16 columns of its row, then the four quarters summed
    const int jn = r1 - r0;
    V z[NR];
    for (int c = 0; c < NR; ++c) z[c] = vzero<V>();
#pragma unroll
    for (int jj = 0; jj < 16; ++jj)
        if (quarter * 16 + jj < jn)
            for (int c = 0; c < NR; ++c) z[c] = caddmul(z[c], li[jj], sz[quarter * 16 + jj][c]);
    for (int c = 0; c < NR; ++c) sacc[quarter][row][c] = z[c];
    __syncthreads();
    if (tid < 64 && i < r1) {
        for (int c = 0; c < NR; ++c) {
            const V v = cadd(cadd(sacc[0][row][c], sacc[1][row][c]), cadd(sacc[2][row][c], sacc[3][row][c]));
            w[(int64_t)c * m + i] = v;
            y[c * n + col0 + i] = cdiv(v, widen<V>(d));
        }
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) atomicExch(flag + t, 1);
}

constexpr int kBigSlice = 1024;  // rows of L21 per partial block of the backward sweep

// backward gather: own entries of z / D, x at the rows below; resets the ticket and the partial counters
template <class V>
__global__ void bigBwdGatherKer(V* w, int m, int k, const V* y, const V* x, const int* urow, int64_t n, int nr, int col0,
                                int* ticket, int* partDone, int tStride, int fStride)
{
    const int64_t g = blockIdx.y;
    w += g * nr * m;
    y += g * nr * n;
    x += g * nr * n;
    ticket += g * tStride;
    partDone += g * fStride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x)
        for (int c = 0; c < nr; ++c) w[(int64_t)c * m + i] = i < k ? y[c * n + col0 + i] : x[c * n + urow[i - k]];
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < (k + 63) / 64; b += gridDim.x * blockDim.x) partDone[b] = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) *ticket = 0;
}

// Backward. The first kb * S tickets are partial blocks: L21^T x for one column block over one slice of
// kBigSlice rows of L21 (no waiting), into part[slice][column]. The next kb tickets are the column
// blocks from the last: each sums its S partials in slice order, subtracts the solved row blocks of L11
// below its diagonal block as they are published, then solves its diagonal block transposed and
// publishes x. Warp w takes 8 columns of a block, lanes the rows.
template <int NR, class V>
__global__ void __launch_bounds__(kBigThreads) bigBwdKer(V* w, const cuComplex* __restrict__ P, const V* __restrict__ Li, int m,
                                                         int k, V* x, int64_t n, int col0, int* ticket, int* flag, int* partDone,
                                                         V* part, int S, int tStride, int fStride)
{
    {
        const int64_t g = blockIdx.y;  // right-hand-side group
        w += g * NR * m;
        x += g * NR * n;
        ticket += g * tStride;
        flag += g * fStride;
        partDone += g * fStride;
        part += g * S * k * NR;
    }
    extern __shared__ __align__(16) unsigned char bigSh[];
    V (*sli)[65] = reinterpret_cast<V (*)[65]>(bigSh);  // the diagonal block's inverse, sli[i][j]
    __shared__ int sT;
    __shared__ V st[64][NR];
    __shared__ V sp[4][64][NR];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) sT = atomicAdd(ticket, 1);
    __syncthreads();
    const int kb = (k + 63) / 64, t = sT, partial = t < kb * S;
    const int cb = partial ? t / S : kb - 1 - (t - kb * S), c0 = cb * 64, cn = min(64, k - c0);
    V acc[8 * NR];  // column warp * 8 + e / NR, right-hand side e % NR
    for (int e = 0; e < 8 * NR; ++e) acc[e] = vzero<V>();
    auto rows = [&](int ib, int ie, bool published) {
        for (int i = ib + lane; i < ie; i += 32) {
            V v[NR];
            for (int c = 0; c < NR; ++c) v[c] = published ? ldcg(w + (int64_t)c * m + i) : w[(int64_t)c * m + i];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int j = warp * 8 + jj;
                if (j < cn) {
                    const V p = widen<V>(P[i + (int64_t)m * (c0 + j)]);
                    for (int c = 0; c < NR; ++c) acc[jj * NR + c] = caddmul(acc[jj * NR + c], p, v[c]);
                }
            }
        }
    };
    // the warp's 8 columns summed over its lanes: lane holds entry e (column warp * 8 + e / NR, rhs e % NR)
    auto reduce = [&](int& e) { return transposeReduce<8 * NR>(acc, lane, e); };
    const bool owner = (lane & (32 / (8 * NR) - 1)) == 0;  // one lane per entry
    if (partial) {
        const int sl = t % S, ib = k + sl * kBigSlice;
        rows(ib, min(ib + kBigSlice, m), false);
        int e;
        const V s = reduce(e);
        const int j = warp * 8 + e / NR;
        if (owner && j < cn) part[((int64_t)sl * k + c0 + j) * NR + e % NR] = s;
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(partDone + cb, 1);
        return;
    }
    for (int e = tid; e < 64 * 64; e += kBigThreads)  // this block's inverse, fetched while the blocks below are solved
        sli[e & 63][e >> 6] = Li[(int64_t)cb * 4096 + e];
    // own entries minus L21^T x from the partial blocks (done early, off the chain), in slice order
    if (tid == 0 && S > 0) waitValue(partDone + cb, S);
    __syncthreads();
    for (int e = tid; e < cn * NR; e += kBigThreads) {
        const int j = e / NR, c = e % NR;
        V v = w[(int64_t)c * m + c0 + j];
        for (int sl = 0; sl < S; ++sl) v = csub(v, ldcg(part + ((int64_t)sl * k + c0 + j) * NR + c));
        st[j][c] = v;
    }
    __syncthreads();  // st is updated below by other threads, even when no row block follows
    for (int rb = kb - 1; rb > cb; --rb) {  // the solved row blocks below: the block loaded before waiting
        const int ib = rb * 64, ie = min(ib + 64, k);
        cuComplex p[2][8];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int i = ib + lane + 32 * h, j = warp * 8 + jj;
                p[h][jj] = i < ie && j < cn ? P[i + (int64_t)m * (c0 + j)] : make_cuComplex(0, 0);
            }
        if (tid == 0) waitFlag(flag + rb);
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = ib + lane + 32 * h;
            if (i < ie) {
                V v[NR];
                for (int c = 0; c < NR; ++c) v[c] = ldcg(w + (int64_t)c * m + i);
#pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                    for (int c = 0; c < NR; ++c) acc[jj * NR + c] = caddmul(acc[jj * NR + c], widen<V>(p[h][jj]), v[c]);
            }
        }
    }
    {
        int e;
        const V s = reduce(e);
        const int j = warp * 8 + e / NR;
        if (owner && j < cn) st[j][e % NR] = csub(st[j][e % NR], s);
    }
    __syncthreads();
    // the diagonal block, transposed: x = L11^-T r, thread (column j, quarter q of the rows) then the quarters summed
    {
        const int j = tid & 63, q = tid >> 6;
        V s[NR];
        for (int c = 0; c < NR; ++c) s[c] = vzero<V>();
#pragma unroll
        for (int ii = 0; ii < 16; ++ii) {
            const int i = q * 16 + ii;
            if (i < cn)
                for (int c = 0; c < NR; ++c) s[c] = caddmul(s[c], sli[i][j], st[i][c]);
        }
        for (int c = 0; c < NR; ++c) sp[q][j][c] = s[c];
        __syncthreads();
        if (tid < cn) {
            for (int c = 0; c < NR; ++c) {
                const V v = cadd(cadd(sp[0][j][c], sp[1][j][c]), cadd(sp[2][j][c], sp[3][j][c]));
                w[(int64_t)c * m + c0 + j] = v;
                x[c * n + col0 + c0 + j] = v;
            }
            __threadfence();
        }
        __syncthreads();
        if (tid == 0) atomicExch(flag + cb, 1);
    }
}

template <class V>
__global__ void permInKer(const SolveIO<V>* b, V* y, const int* perm, int64_t n, int nr)
{
    for (int64_t q = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; q < n; q += (int64_t)gridDim.x * blockDim.x)
        for (int c = 0; c < nr; ++c) y[c * n + q] = fromIO<V>(b[c * n + perm[q]]);
}

template <class V>
__global__ void permOutKer(const V* xp, SolveIO<V>* x, const int* perm, int64_t n, int nr)
{
    for (int64_t q = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; q < n; q += (int64_t)gridDim.x * blockDim.x)
        for (int c = 0; c < nr; ++c) x[c * n + perm[q]] = toIO(xp[c * n + q]);
}

inline void solveCk(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}

// The solve of one factorization's structure; the panels are read at every solve (refactorizations
// only change their values). V: the vectors' precision.
template <class V>
class FrontSolveT
{
    static constexpr size_t kInvSmem = 64 * 65 * sizeof(V);  // a diagonal block's inverse in shared memory

public:
    // node, children, parent, big: the factorization's supernodes in postorder; dRelind/dUrow: per update
    // row its position in the parent's rows and its pivot number; dPerm: pivot -> original row; streams and
    // cuBLAS handles: the large-front pool (one handle per stream)
    void setup(int64_t n, int maxNr, const std::vector<SolveNode>& node, const std::vector<int>& children,
               const std::vector<int>& parent, const std::vector<char>& big, const int* dChildren, const int* dRelind,
               const int* dUrow, const cuComplex* dPanels, int64_t relLen, const int* dPerm,
               const std::vector<cudaStream_t>& streams)
    {
        n_ = n;
        maxNr_ = maxNr;
        maxG_ = std::max(1, maxNr / 2);  // groups of 2 right-hand sides
        node_ = node;
        children_ = children;
        parent_ = parent;
        big_ = big;
        streams_ = streams;
        perm_ = dPerm;
        const int R = (int)node.size();
        // small subtrees (one warp each): maximal subtrees whose fronts all have m <= 32, contiguous in
        // postorder, largest first for the warps that take them in turn
        std::vector<char> small(R, 0);
        std::vector<int> first(R);
        const char* smallEnv = std::getenv("REF_SOLVE_SMALL");  // development: 0 sends them through the levels
        const bool useSmall = !(smallEnv && std::atoi(smallEnv) == 0);
        for (int r = 0; r < R; ++r) {
            small[r] = useSmall && node[r].m <= 32 && !big[r];
            first[r] = r;
            for (int ci = node[r].childBeg; ci < node[r].childEnd; ++ci) {
                small[r] = small[r] && small[children[ci]];
                first[r] = std::min(first[r], first[children[ci]]);
            }
        }
        std::vector<int2> sub;
        int nSmall = 0;
        for (int r = 0; r < R; ++r)
            if (small[r] && (parent[r] < 0 || !small[parent[r]])) {
                sub.push_back(make_int2(first[r], r));
                nSmall += r - first[r] + 1;
            }
        std::stable_sort(sub.begin(), sub.end(), [](int2 x, int2 y) { return x.y - x.x > y.y - y.x; });
        nSub_ = (int)sub.size();
        // levels of the other fronts below the large ones: height above the leaves, small subtrees left out
        std::vector<int> level(R, 0);
        int nlev = 0;
        for (int r = 0; r < R; ++r) {
            if (big[r] || small[r]) continue;
            for (int ci = node[r].childBeg; ci < node[r].childEnd; ++ci)
                if (!small[children[ci]]) level[r] = std::max(level[r], level[children[ci]] + 1);
            nlev = std::max(nlev, level[r] + 1);
        }
        const char* midEnv = std::getenv("REF_SOLVE_MID");  // development: 0 sends them through the CTA kernels
        const bool useMid = !(midEnv && std::atoi(midEnv) == 0);
        auto mid = [&](int r) { return useMid && node[r].m <= 32 * kMidRows && node[r].k <= kMidK; };
        levPtr_.assign(nlev + 1, 0);
        levMaxM_.assign(nlev, 0);
        levMid_.assign(nlev, 0);
        for (int r = 0; r < R; ++r)
            if (!big[r] && !small[r]) {
                levPtr_[level[r] + 1]++;
                if (mid(r)) levMid_[level[r]]++;
                else levMaxM_[level[r]] = std::max(levMaxM_[level[r]], node[r].m);
            }
        for (int l = 0; l < nlev; ++l) levPtr_[l + 1] += levPtr_[l];
        if (std::getenv("REF_SOLVE_STATS")) {  // development: the level fronts by m (count, panel entries, columns)
            const int lim[] = {64, 128, 256, 512, 1 << 30};
            int64_t cnt[5] = {}, ent[5] = {}, col[5] = {};
            for (int r = 0; r < R; ++r) {
                if (big[r] || small[r]) continue;
                int b = 0;
                while (node[r].m > lim[b]) ++b;
                ++cnt[b];
                ent[b] += (int64_t)node[r].m * node[r].k;
                col[b] += node[r].k;
            }
            for (int b = 0; b < 5; ++b)
                std::printf("  level fronts m <= %d: %lld, panel entries %.3g, columns %lld (mean k %.1f)\n", lim[b],
                            (long long)cnt[b], (double)ent[b], (long long)col[b], cnt[b] ? (double)col[b] / cnt[b] : 0.0);
        }
        std::vector<int> list(std::max(levPtr_[nlev], 1)), next(levPtr_.begin(), levPtr_.end() - 1),
            nextCta(nlev);
        for (int l = 0; l < nlev; ++l) nextCta[l] = levPtr_[l] + levMid_[l];
        for (int r = 0; r < R; ++r)  // each level: its mid-size fronts first, then the others
            if (!big[r] && !small[r]) list[mid(r) ? next[level[r]]++ : nextCta[level[r]]++] = r;
        for (int l = 0; l < nlev; ++l)
            if ((size_t)levMaxM_[l] * 2 * sizeof(V) > 96 * 1024)
                die("front_solve: a front below the large-front path is too big for shared memory");
        for (int r = 0; r < R; ++r)
            if (big[r]) bigOrder_.push_back(r);  // postorder
        int64_t wTotal = 0, flagTotal = 0, partTotal = 0;
        wOff_.assign(R, -1);
        flagOff_.assign(R, -1);
        partOff_.assign(R, -1);
        for (int r : bigOrder_) {
            wOff_[r] = wTotal;
            wTotal += (int64_t)node[r].m * maxNr;
            flagOff_[r] = flagTotal;
            flagTotal += (node[r].k + 63) / 64;
            partOff_[r] = partTotal;
            partTotal += (int64_t)slices(node[r]) * node[r].k * maxNr;
        }
        auto up = [](const void* h, size_t bytes) {
            void* d = nullptr;
            solveCk(cudaMalloc(&d, std::max<size_t>(bytes, 1)), "cudaMalloc");
            if (bytes) solveCk(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
            return d;
        };
        dNode_ = static_cast<SolveNode*>(up(node.data(), R * sizeof(SolveNode)));
        dList_ = static_cast<int*>(up(list.data(), list.size() * sizeof(int)));
        dSub_ = static_cast<int2*>(up(sub.data(), sub.size() * sizeof(int2)));
        solveCk(cudaMalloc(&dSubNext_, 2 * maxG_ * sizeof(int)), "cudaMalloc subtree counters");
        int occ = 0;
        solveCk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, smallFwdKer<2, V>, 32 * kSmallSolveWarps, 0), "occupancy");
        int dev = 0, nsm = 0;
        solveCk(cudaGetDevice(&dev), "device");
        solveCk(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev), "SM count");
        subGrid_ = std::max(1, std::min(nsm * std::max(occ, 1), (nSub_ + kSmallSolveWarps - 1) / kSmallSolveWarps));
        solveCk(cudaMalloc(&dU_, std::max<int64_t>(relLen, 1) * maxNr * sizeof(V)), "cudaMalloc u");
        solveCk(cudaMalloc(&dY_, n * maxNr * sizeof(V)), "cudaMalloc y");
        solveCk(cudaMalloc(&dX_, n * maxNr * sizeof(V)), "cudaMalloc x");
        solveCk(cudaMalloc(&dW_, std::max<int64_t>(wTotal, 1) * sizeof(V)), "cudaMalloc w");
        dWOff_ = static_cast<int64_t*>(up(wOff_.data(), R * sizeof(int64_t)));
        flagTotal_ = std::max<int64_t>(flagTotal, 1);  // flags and partial counters, per group
        solveCk(cudaMalloc(&dFlag_, flagTotal_ * maxG_ * sizeof(int)), "cudaMalloc flags");
        solveCk(cudaMemset(dFlag_, 0, flagTotal_ * maxG_ * sizeof(int)), "memset flags");
        solveCk(cudaMalloc(&dTicket_, std::max<size_t>(R, 1) * maxG_ * sizeof(int)), "cudaMalloc tickets");
        solveCk(cudaMalloc(&dPartDone_, flagTotal_ * maxG_ * sizeof(int)), "cudaMalloc partial counters");
        solveCk(cudaMalloc(&dPart_, std::max<int64_t>(partTotal, 1) * sizeof(V)), "cudaMalloc partials");
        // the inverses of the large fronts' 64 x 64 diagonal blocks (newFactor() computes them)
        std::vector<InvBlock> inv;
        linvOff_.assign(R, -1);
        int64_t linvTotal = 0;
        for (int r : bigOrder_) {
            linvOff_[r] = linvTotal;
            for (int c0 = 0; c0 < node[r].k; c0 += 64) {
                inv.push_back(InvBlock{node[r].panelOff, linvTotal, node[r].m, c0, std::min(64, node[r].k - c0), 0});
                linvTotal += 4096;
            }
        }
        nInv_ = (int)inv.size();
        dInv_ = static_cast<InvBlock*>(up(inv.data(), inv.size() * sizeof(InvBlock)));
        solveCk(cudaMalloc(&dLinv_, std::max<int64_t>(linvTotal, 1) * sizeof(V)), "cudaMalloc inverses");
        linvBytes_ = linvTotal * sizeof(V);
        dChildren_ = dChildren;
        a_ = SolveArgs<V>{dNode_, dChildren, dRelind, dUrow, dPanels, dU_, dY_, dX_, relLen, n};
        done_.resize(R);
        for (int r : bigOrder_) solveCk(cudaEventCreateWithFlags(&done_[r], cudaEventDisableTiming), "event");
        end_.resize(streams.size());
        for (auto& e : end_) solveCk(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        solveCk(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming), "event");
        stream_.assign(R, -1);  // a chain of large fronts stays on one stream (the factorization's rule)
        int rr = 0;
        for (int r : bigOrder_) {
            int si = -1;
            for (int ci = node[r].childBeg; ci < node[r].childEnd; ++ci)
                if (big[children[ci]]) si = stream_[children[ci]];
            stream_[r] = si >= 0 ? si : rr++ % (int)streams.size();
        }
        for (const void* f : {(const void*)solveFwdKer<1, V>, (const void*)solveFwdKer<2, V>, (const void*)solveBwdKer<1, V>,
                              (const void*)solveBwdKer<2, V>})
            solveCk(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024), "smem attribute");
        for (const void* f : {(const void*)bigBwdKer<1, V>, (const void*)bigBwdKer<2, V>, (const void*)bigInvKer<V>})
            solveCk(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kInvSmem), "smem attribute");
        // load every kernel now: the large-front kernels spin on flags, and a lazy module load while one spins
        // could wait for a context synchronization that never comes
        auto load = [](const void* f) {
            cudaFuncAttributes attr;
            solveCk(cudaFuncGetAttributes(&attr, f), "cudaFuncGetAttributes");
        };
        for (const void* f : {(const void*)smallFwdKer<1, V>, (const void*)smallFwdKer<2, V>, (const void*)smallBwdKer<1, V>,
                              (const void*)smallBwdKer<2, V>, (const void*)midFwdKer<1, V>, (const void*)midFwdKer<2, V>,
                              (const void*)midBwdKer<1, V>, (const void*)midBwdKer<2, V>})
            load(f);
        for (const void* f : {(const void*)bigFwdKer<1, V>, (const void*)bigFwdKer<2, V>, (const void*)bigBwdKer<1, V>,
                              (const void*)bigBwdKer<2, V>, (const void*)bigFwdAssembleKer<V>, (const void*)bigBwdGatherKer<V>,
                              (const void*)permInKer<V>, (const void*)permOutKer<V>, (const void*)bigInvKer<V>})
            load(f);
        if constexpr (std::is_same_v<V, dfComplex>) {
            // development: REF_DF_SINGLE=small,mid,cta,bigfwd,bigbwd runs those phases in single precision (sfComplex)
            if (const char* e = std::getenv("REF_DF_SINGLE")) {
                const std::string v = std::string(",") + e + ",";
                const char* names[] = {"small", "mid", "cta", "bigfwd", "bigbwd"};
                for (int i = 0; i < 5; ++i)
                    if (v.find(std::string(",") + names[i] + ",") != std::string::npos) cheap_ |= 1u << i;
                std::printf("front solve: REF_DF_SINGLE=%s, phases in single precision mask %#x\n", e, cheap_);
            }
            using W = sfComplex;
            for (const void* f : {(const void*)solveFwdKer<1, W>, (const void*)solveFwdKer<2, W>, (const void*)solveBwdKer<1, W>,
                                  (const void*)solveBwdKer<2, W>})
                solveCk(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024), "smem attribute");
            for (const void* f : {(const void*)bigBwdKer<1, W>, (const void*)bigBwdKer<2, W>})
                solveCk(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kInvSmem), "smem attribute");
            for (const void* f : {(const void*)smallFwdKer<1, W>, (const void*)smallFwdKer<2, W>, (const void*)smallBwdKer<1, W>,
                                  (const void*)smallBwdKer<2, W>, (const void*)midFwdKer<1, W>, (const void*)midFwdKer<2, W>,
                                  (const void*)midBwdKer<1, W>, (const void*)midBwdKer<2, W>, (const void*)bigFwdKer<1, W>,
                                  (const void*)bigFwdKer<2, W>, (const void*)bigBwdKer<1, W>, (const void*)bigBwdKer<2, W>,
                                  (const void*)bigFwdAssembleKer<W>, (const void*)bigBwdGatherKer<W>})
                load(f);
        }
        std::printf("front solve (%s vectors): %d small subtrees (%d supernodes, one warp each), %d levels below the "
                    "large fronts (%d supernodes), %zu large fronts; update vectors %.2f GB, large-front vectors %.2f MB, inverses of "
                    "their diagonal blocks %.2f GB\n",
                    std::is_same_v<V, cuComplex> ? "single" : std::is_same_v<V, dfComplex> ? "double-float" : "double", nSub_,
                    nSmall, nlev, levPtr_[nlev], bigOrder_.size(), relLen * maxNr * (double)sizeof(V) / (1 << 30),
                    wTotal * (double)sizeof(V) / (1 << 20), linvBytes_ / (double)(1 << 30));
    }

    // After every factorization (new panel values), before the next solve: the inverses of the large fronts'
    // diagonal blocks, on the legacy stream
    void newFactor()
    {
        if (nInv_ > 0) bigInvKer<V><<<nInv_, 64, kInvSmem>>>(a_.panels, dInv_, dLinv_);
        solveCk(cudaGetLastError(), "block inverses");
        factorSeen_ = true;
    }

    // With graphs on, the sweeps between reading b and writing x (they use only this object's buffers) are
    // captured once per number of right-hand sides and replayed: the host no longer issues every kernel
    void useGraphs(bool on) { graphs_ = on; }

    // x = A^-1 b (through the factors) for nr <= maxNr right-hand sides, n x nr column-major; on the legacy
    // stream (the large fronts fork from it and join it again)
    void solve(const SolveIO<V>* b, SolveIO<V>* x, int nr)
    {
        if (nr != 1 && nr % 2 != 0) die("front_solve: 1 or an even number of right-hand sides");
        if (nr > std::max(maxNr_, 1)) die("front_solve: more right-hand sides than set up for");
        const int NR = nr == 1 ? 1 : 2, G = nr == 1 ? 1 : nr / 2;  // groups of NR
        if (!factorSeen_) die("front_solve: newFactor() was not called after the factorization");
        const int g = (int)std::min<int64_t>((n_ + 255) / 256, 1 << 20);
        permInKer<V><<<g, 256>>>(b, dY_, perm_, n_, nr);
        if (graphs_) {
            cudaGraphExec_t& ex = graph_[nr];
            if (!ex) {
                if (!cs_) solveCk(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking), "stream");
                cudaGraph_t gr;
                ls_ = cs_;
                solveCk(cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal), "begin capture");
                sweeps(NR, G);
                solveCk(cudaStreamEndCapture(cs_, &gr), "end capture");
                ls_ = 0;
                solveCk(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                solveCk(cudaGraphDestroy(gr), "graph");
            }
            solveCk(cudaGraphLaunch(ex, 0), "graph launch");
        } else sweeps(NR, G);
        permOutKer<V><<<g, 256>>>(dX_, x, perm_, n_, nr);
        solveCk(cudaGetLastError(), "front solve");
    }

private:
    void sweeps(int NR, int G)  // on ls_
    {
        if (nSub_ > 0) solveCk(cudaMemsetAsync(dSubNext_, 0, 2 * maxG_ * sizeof(int), ls_), "memset subtree counters");
        launchSmall(true, NR, G);
        const int nlev = (int)levMaxM_.size();
        for (int l = 0; l < nlev; ++l) launchLevel(true, l, NR, G);
        if (!bigOrder_.empty()) bigSweep(true, NR, G);
        if (!bigOrder_.empty()) bigSweep(false, NR, G);
        for (int l = nlev - 1; l >= 0; --l) launchLevel(false, l, NR, G);
        launchSmall(false, NR, G);
    }

    void launchSmall(bool fwd, int NR, int G)
    {
        if (nSub_ == 0) return;
        const dim3 grid(subGrid_, G);
        const int b = 32 * kSmallSolveWarps;
        int* next = dSubNext_ + (fwd ? 0 : maxG_);
        withType(kCheapSmall, [&](auto tag) {
            using W = decltype(tag);
            const SolveArgs<W> a = argsAs<W>();
            if (fwd) {
                if (NR == 1) smallFwdKer<1, W><<<grid, b, 0, ls_>>>(a, dSub_, nSub_, next);
                else smallFwdKer<2, W><<<grid, b, 0, ls_>>>(a, dSub_, nSub_, next);
            } else {
                if (NR == 1) smallBwdKer<1, W><<<grid, b, 0, ls_>>>(a, dSub_, nSub_, next);
                else smallBwdKer<2, W><<<grid, b, 0, ls_>>>(a, dSub_, nSub_, next);
            }
        });
    }

    void launchLevel(bool fwd, int l, int NR, int G)
    {
        const int nMid = levMid_[l];
        if (nMid > 0) {
            const dim3 g(std::min((nMid + kSmallSolveWarps - 1) / kSmallSolveWarps, 65535), G);
            const int b = 32 * kSmallSolveWarps;
            const int* ml = dList_ + levPtr_[l];
            withType(kCheapMid, [&](auto tag) {
                using W = decltype(tag);
                const SolveArgs<W> a = argsAs<W>();
                if (fwd) {
                    if (NR == 1) midFwdKer<1, W><<<g, b, 0, ls_>>>(a, ml, nMid);
                    else midFwdKer<2, W><<<g, b, 0, ls_>>>(a, ml, nMid);
                } else {
                    if (NR == 1) midBwdKer<1, W><<<g, b, 0, ls_>>>(a, ml, nMid);
                    else midBwdKer<2, W><<<g, b, 0, ls_>>>(a, ml, nMid);
                }
            });
        }
        const int count = levPtr_[l + 1] - levPtr_[l] - nMid;
        if (count == 0) return;
        const size_t sh = (size_t)levMaxM_[l] * NR * sizeof(V);
        const dim3 grid(std::min(count, 65535), G);
        const int* list = dList_ + levPtr_[l] + nMid;
        withType(kCheapCta, [&](auto tag) {
            using W = decltype(tag);
            const SolveArgs<W> a = argsAs<W>();
            if (fwd) {
                if (NR == 1) solveFwdKer<1, W><<<grid, kSolveThreads, sh, ls_>>>(a, list, count);
                else solveFwdKer<2, W><<<grid, kSolveThreads, sh, ls_>>>(a, list, count);
            } else {
                if (NR == 1) solveBwdKer<1, W><<<grid, kSolveThreads, sh, ls_>>>(a, list, count);
                else solveBwdKer<2, W><<<grid, kSolveThreads, sh, ls_>>>(a, list, count);
            }
        });
    }

    // the large fronts on the stream pool, forward in postorder or backward in reverse postorder
    void bigSweep(bool fwd, int NR, int G)
    {
        const int tS = (int)node_.size(), fS = (int)flagTotal_;  // per-group strides of tickets and flags
        solveCk(cudaMemsetAsync(dFlag_, 0, flagTotal_ * maxG_ * sizeof(int), ls_), "memset flags");
        solveCk(cudaEventRecord(start_, ls_), "event");
        for (cudaStream_t s : streams_) solveCk(cudaStreamWaitEvent(s, start_, 0), "wait");
        const int nb = (int)bigOrder_.size();
        for (int t = 0; t < nb; ++t) {
            const int r = fwd ? bigOrder_[t] : bigOrder_[nb - 1 - t];
            const SolveNode& x = node_[r];
            const int m = x.m, k = x.k, si = stream_[r], kb = (k + 63) / 64;
            cudaStream_t st = streams_[si];
            const cuComplex* P = a_.panels + x.panelOff;
            int* flag = dFlag_ + flagOff_[r];
            int* ticket = dTicket_ + r;
            withType(fwd ? kCheapBigFwd : kCheapBigBwd, [&](auto tag) {
            using W = decltype(tag);
            W* const dW = reinterpret_cast<W*>(dW_);
            W* const dY = reinterpret_cast<W*>(dY_);
            W* const dX = reinterpret_cast<W*>(dX_);
            W* const dU = reinterpret_cast<W*>(dU_);
            W* w = dW + wOff_[r];
            const W* Li = reinterpret_cast<const W*>(dLinv_ + linvOff_[r]);
            if (fwd) {
                for (int ci = x.childBeg; ci < x.childEnd; ++ci) {
                    const int c = children_[ci];
                    if (big_[c] && stream_[c] != si) solveCk(cudaStreamWaitEvent(st, done_[c], 0), "wait");
                }
                bigFwdAssembleKer<W><<<dim3(1, G), 256, 0, st>>>(w, m, k, dY, n_, NR, x.col0, dNode_, dChildren_, x.childBeg,
                                                                 x.childEnd, dWOff_, dW, dU, a_.relLen, a_.relind, ticket, tS);
                const dim3 grid(kb + (m - k + 63) / 64, G);
                if (NR == 1)
                    bigFwdKer<1, W><<<grid, kBigThreads, 0, st>>>(w, P, Li, m, k, dY, n_, x.col0, ticket, flag, tS, fS);
                else bigFwdKer<2, W><<<grid, kBigThreads, 0, st>>>(w, P, Li, m, k, dY, n_, x.col0, ticket, flag, tS, fS);
            } else {
                const int p = parent_[r];
                if (p >= 0 && stream_[p] != si) solveCk(cudaStreamWaitEvent(st, done_[p], 0), "wait");
                int* partDone = dPartDone_ + flagOff_[r];
                W* part = reinterpret_cast<W*>(dPart_ + partOff_[r]);
                const int S = slices(x);
                const dim3 grid(kb * (S + 1), G);
                bigBwdGatherKer<W><<<dim3(std::min((m + 255) / 256, 4096), G), 256, 0, st>>>(
                    w, m, k, dY, dX, a_.urow + (x.relOff >= 0 ? x.relOff : 0), n_, NR, x.col0, ticket, partDone, tS, fS);
                if (NR == 1)
                    bigBwdKer<1, W><<<grid, kBigThreads, kInvSmem, st>>>(w, P, Li, m, k, dX, n_, x.col0, ticket, flag, partDone,
                                                                         part, S, tS, fS);
                else
                    bigBwdKer<2, W><<<grid, kBigThreads, kInvSmem, st>>>(w, P, Li, m, k, dX, n_, x.col0, ticket, flag, partDone,
                                                                         part, S, tS, fS);
            }
            });
            solveCk(cudaEventRecord(done_[r], st), "event");
        }
        for (size_t i = 0; i < streams_.size(); ++i) {
            solveCk(cudaEventRecord(end_[i], streams_[i]), "event");
            solveCk(cudaStreamWaitEvent(ls_, end_[i], 0), "wait");
        }
    }

    // development (REF_DF_SINGLE): phases of a double-float solve run in single precision on the same buffers
    static constexpr unsigned kCheapSmall = 1, kCheapMid = 2, kCheapCta = 4, kCheapBigFwd = 8, kCheapBigBwd = 16;
    unsigned cheap_ = 0;
    template <class F> void withType(unsigned bit, F&& f)
    {
        if constexpr (std::is_same_v<V, dfComplex>)
            if (cheap_ & bit) return f(sfComplex{});
        f(V{});
    }
    template <class W> SolveArgs<W> argsAs() const
    {
        return SolveArgs<W>{a_.node, a_.children, a_.relind, a_.urow, a_.panels, reinterpret_cast<W*>(a_.u),
                            reinterpret_cast<W*>(a_.y), reinterpret_cast<W*>(a_.x), a_.relLen, a_.n};
    }

    int64_t n_ = 0, flagTotal_ = 1;
    int maxNr_ = 0, maxG_ = 1;
    std::vector<SolveNode> node_;
    std::vector<int> children_, parent_, levPtr_, levMaxM_, levMid_, bigOrder_, stream_;
    std::vector<char> big_;
    static int slices(const SolveNode& x) { return (x.m - x.k + kBigSlice - 1) / kBigSlice; }

    std::vector<int64_t> wOff_, flagOff_, partOff_, linvOff_;
    InvBlock* dInv_ = nullptr;
    V* dLinv_ = nullptr;
    int nInv_ = 0;
    size_t linvBytes_ = 0;
    bool factorSeen_ = false;
    std::vector<cudaStream_t> streams_;
    bool graphs_ = false;
    cudaStream_t ls_ = 0, cs_ = nullptr;  // the sweeps' stream (legacy, or the capture's)
    std::map<int, cudaGraphExec_t> graph_;  // the captured sweeps by number of right-hand sides
    int64_t* dWOff_ = nullptr;
    int *dFlag_ = nullptr, *dTicket_ = nullptr, *dPartDone_ = nullptr;
    V* dPart_ = nullptr;
    const int* dChildren_ = nullptr;
    std::vector<cudaEvent_t> done_, end_;
    cudaEvent_t start_ = nullptr;
    const int* perm_ = nullptr;
    SolveNode* dNode_ = nullptr;
    int* dList_ = nullptr;
    int2* dSub_ = nullptr;      // small subtrees: [first, root] in postorder
    int* dSubNext_ = nullptr;   // their work counters, forward and backward
    int nSub_ = 0, subGrid_ = 1;
    V *dU_ = nullptr, *dY_ = nullptr, *dX_ = nullptr, *dW_ = nullptr;
    SolveArgs<V> a_{};
};

using FrontSolve = FrontSolveT<cuComplex>;          // all single precision
using FrontSolve64 = FrontSolveT<cuDoubleComplex>;  // vectors and sums in double, panels single
using FrontSolveDF = FrontSolveT<dfComplex>;        // vectors and sums in double-float, panels single
