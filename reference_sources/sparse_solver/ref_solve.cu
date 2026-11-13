// Supernodal multifrontal factorization, step 1: scheduling the small supernodes of a sparse
// complex-symmetric LDL^T by dependency counters.
//
// Region: every supernode whose columns lie in a leaf of the ND partition tree of a ref_order plan. The
// region is closed under descendants, so A restricted to its columns equals L D L^T restricted to them,
// which the run checks with a random vector. The supernodal structure comes from the GPU symbolic
// analysis (gpu_symbolic.cu), which reproduces CHOLMOD's with the plan's permutation entry for entry;
// --symbolic cholmod uses CHOLMOD itself, --symbolic both runs both and compares them. The front maps
// (update-row positions in the parent, assembly positions of A's entries) are built on the GPU as well.
//
// Each supernode is a multifrontal front in complex single precision (1x1 pivots; an exactly zero pivot is
// replaced by a tiny one): assemble the entries of A (precomputed positions), pull the children's update
// matrices (extend-add by the parent, so no atomics), partial LDL^T of the k pivot columns, giving the L
// panel (m x k, D on its diagonal) and the (m-k) x (m-k) update matrix for the parent.
//   small fronts  subtrees whose fronts all have m <= --small-m (32) go to smallKer first: one warp per
//                 subtree, in postorder, the front in shared memory
//   kernel 2      (default) the others, one block each, in place: left-looking panel in blocks of 16
//                 columns, then one register-tiled pass U -= L21 D L21^T over the update matrix
//   kernel 1      the first version: rank-1 updates on a front scratch in global memory
// The L panels stay in device memory, or with --writeback go back to pinned host memory asynchronously by
// DMA: a block writes its finished panel into a device ring buffer (allocated with one atomic, waiting while
// the ring is full) and counts the written elements per ring chunk; a host thread polls the chunk counters
// and copies every run of complete chunks to the host with cudaMemcpyAsync on its own stream (copy
// engine, overlapped with the factorization), then clears their counters and publishes how far the ring
// may be reused. On the host the panels lie in the order they were written; hostOff[] locates each one.
//
// Schedules, which give bitwise identical factors:
//   level      one launch per tree level, its supernodes spread over the resident blocks
//   persistent one launch; resident blocks take ready supernodes from a queue, and the block that finishes
//              the last child of a supernode enqueues it (dependency counter)
//
// --cpu-ref runs the same multifrontal factorization on the CPU in complex double and compares the panels
// supernode by supernode (debugging; slow on big matrices).
//
// usage: ref_solve --user-perm P.perm --user-tree P.rowtree [options] <dir> [<dir> ...]
//   The analysis is done once on the first directory, and every directory is then factorized with its own
//   values. --no-verify skips the
//   factorization check (production timing). Options: [--whole] [--large-m M]
//   [--streams S] [--nb NB] [--kernel 1|2] [--small-m M] [--writeback] [--ring-mb M] [--chunk-mb C]
//   [--threads T] [--blocks-per-sm B] [--reps R] [--schedule both|level|persistent] [--symbolic gpu|cholmod|both]
//   [--solve R] [--solve-precision single|double|double-float] [--upd-plan on|off] [--graphs on|off] [--stats]
//   [--cpu-ref], and the
//   full solve with the options of testbench/mixed_solve.h (--gmres, --dd-ir N, --inner-tol T, --mp-iters N,
//   --recycle, ...).
//   --upd-plan (default on with --whole; needs the level schedule) places the update matrices by liveness
//   instead of statically (far less device memory).
//   --graphs (default on) captures the factorization with the level schedule, and the front solve, as CUDA
//   graphs once per pattern and replays them: the host no longer issues every kernel and cuBLAS call.
//   --solve R (with --whole) times the supernodal triangular solve (front_solve.cuh) on --solve-nr (2) random
//   right-hand sides, best of R; without --no-verify it compares x with the same sweeps in double from the host copy
//   of the factor, and reports ||b - A x|| / ||b||.
#include "bench_common.h"
#include "mixed_solve.h"

#include "front_kernels.cuh"
#include "front_solve.cuh"
#include "gpu_symbolic.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <suitesparse/cholmod.h>

#include <array>
#include <map>
#include <set>
#include <climits>
#include <cstring>
#include <random>

namespace {

void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}
#define CK(x) ck((x), #x)

struct SNode  // one supernode of the region
{
    int m, k;          // rows of the front (= of the L panel), pivot columns
    int parent;        // region index of the parent, -1 if the parent is outside the region
    int childBeg, childEnd;
    int64_t panelOff;  // L panel (m x k, column-major)
    int64_t updOff;    // update matrix ((m-k) x (m-k), column-major, lower triangle used)
    int64_t relOff;    // positions of this supernode's update rows in the parent's front
    int64_t aBeg, aEnd;  // entries of A assembled into this front
};

struct Args
{
    const SNode* sn;
    const int* children;
    const int* relind;
    const int* aPos;
    const cuComplex* aVal;
    cuComplex* panels;  // device panels (no write-back)
    cuComplex* upd;
    cuComplex* scratch;
    int64_t frontStride;
    unsigned* zeroPivots;
    // DMA write-back (dma != 0): ring of ringElems values in chunks of chunkElems, nChunks counters of
    // elements written per chunk, the global allocation head, how far the host has copied (and freed)
    int dma;
    cuComplex* ring;
    unsigned long long ringElems, chunkElems;
    int nChunks;
    unsigned* chunkCnt;
    unsigned long long* allocHead;
    const unsigned long long* freed;
    int64_t* hostOff;  // global allocation offset of each panel = its offset in the host buffer
    // --upd-plan: a negative updOff is -1 - offset on the small-front warp's own stack, which starts at
    // upd + arenaBase + warp * arenaPerWarp
    int64_t arenaBase, arenaPerWarp;
    int splitSchur;  // level schedule: the Schur update of fronts with m - k > kSplitMu is left to schurTileKer
    int stepPanels;  // level schedule: panels with k > kStepW go right-looking, kStepW columns per step
    // schurTileKer: for the child at position ci of children, the first row of each tile of its parent's update
    // matrix, tileBound[tileBoundOff[ci] + t], t = 0 .. (tiles of the parent)
    const int* tileBound;
    const int* tileBoundOff;
};
constexpr int kSplitMu = 64;
constexpr int kStepW = 64;  // block columns of a stepped panel
constexpr size_t kTileSmem = (64 * 65 + 2 * KC * 64) * sizeof(cuComplex);  // schurTileKer: the tile and the staging

// After the block's barrier that follows a front written to the ring at off: thread 0 makes the panel
// visible and counts its elements per ring chunk (the host copies a chunk once its count is complete).
__device__ __forceinline__ void wbCommit(const Args& a, unsigned long long off, int64_t len)
{
    for (unsigned long long p = off, e = off + len; p < e;) {
        const unsigned long long c = p / a.chunkElems, ce = min(e, (c + 1) * a.chunkElems);
        atomicAdd(&a.chunkCnt[c % a.nChunks], (unsigned)(ce - p));
        p = ce;
    }
}

// Factorize front r with the whole block. F: this block's front scratch (m x m, column-major), w: shared
// buffer of m complex values. With write-back, thread 0 returns the panel's ring offset in wbOff (to commit
// after the caller's barrier) and keeps the last freed mark it saw in freedSeen.
__device__ void factorFront(int r, const Args& a, cuComplex* F, cuComplex* w, unsigned long long& wbOff,
                            unsigned long long& freedSeen)
{
    const SNode* __restrict__ sn = a.sn;
    const int* __restrict__ children = a.children;
    const int* __restrict__ relind = a.relind;
    const int* __restrict__ aPos = a.aPos;
    const cuComplex* __restrict__ aVal = a.aVal;
    cuComplex* upd = a.upd;
    unsigned* zeroPivots = a.zeroPivots;
    const SNode me = sn[r];
    const int m = me.m, k = me.k, tid = threadIdx.x, nt = blockDim.x;
    const int64_t mm = (int64_t)m * m, len = (int64_t)m * k;  // the panel: first k columns of the front
    __shared__ unsigned long long sOff;
    unsigned long long off = 0;
    if (a.dma && tid == 0) off = atomicAdd(a.allocHead, (unsigned long long)len);  // used after the assembly
    for (int64_t idx = tid; idx < mm; idx += nt) F[idx] = make_cuComplex(0, 0);
    __syncthreads();
    for (int64_t e = me.aBeg + tid; e < me.aEnd; e += nt) {  // distinct positions
        const cuComplex v = aVal[e];
        F[aPos[e]].x += v.x;
        F[aPos[e]].y += v.y;
    }
    if (a.dma && tid == 0) {
        // [off, off + len) may be written once the host has copied everything below off + len - ringElems.
        // Waiting here is deadlock-free: every lower range belongs to a front that is already being computed.
        while (off + len > freedSeen + a.ringElems) {
            freedSeen = *(volatile const unsigned long long*)a.freed;
            if (off + len > freedSeen + a.ringElems) __nanosleep(200);
        }
        sOff = off;
        wbOff = off;
        a.hostOff[r] = (int64_t)off;
    }
    __syncthreads();
    for (int ci = me.childBeg; ci < me.childEnd; ++ci) {  // children one at a time: no write conflicts
        const SNode ch = sn[children[ci]];
        const int mc = ch.m - ch.k;
        const cuComplex* U = upd + ch.updOff;
        const int* rel = relind + ch.relOff;
        for (int64_t idx = tid; idx < (int64_t)mc * mc; idx += nt) {
            const int i = (int)(idx % mc), j = (int)(idx / mc);
            if (i < j) continue;
            const cuComplex u = __ldcg(U + idx);  // written by another block: bypass L1
            const int64_t p = rel[i] + (int64_t)m * rel[j];
            F[p].x += u.x;
            F[p].y += u.y;
        }
        __syncthreads();
    }
    const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
    for (int j = 0; j < k; ++j) {
        cuComplex d = F[j + (int64_t)m * j];
        if (d.x == 0 && d.y == 0) {
            d = make_cuComplex(1e-30f, 0);
            if (tid == 0) atomicAdd(zeroPivots, 1u);
        }
        __syncthreads();  // everyone has read the pivot before it is overwritten below
        for (int i = j + 1 + tid; i < m; i += nt) {
            const cuComplex v = F[i + (int64_t)m * j];
            w[i] = v;                         // a_ij, for the rank-1 update
            F[i + (int64_t)m * j] = cdiv(v, d);  // l_ij
        }
        if (tid == 0) F[j + (int64_t)m * j] = d;
        __syncthreads();
        for (int l = j + 1 + warp; l < m; l += nw) {  // F[i, l] -= a_ij * l_lj, i >= l > j
            const cuComplex ll = F[l + (int64_t)m * j];
            cuComplex* col = F + (int64_t)m * l;
            for (int i = l + lane; i < m; i += 32) {
                const cuComplex t = cmul(w[i], ll);
                col[i].x -= t.x;
                col[i].y -= t.y;
            }
        }
        __syncthreads();
    }
    if (!a.dma) {
        cuComplex* P = a.panels + me.panelOff;
        for (int64_t idx = tid; idx < len; idx += nt) P[idx] = F[idx];
    } else {  // into the ring, split at its end; committed by the caller
        const unsigned long long start = sOff % a.ringElems;
        const int64_t first = (int64_t)min((unsigned long long)len, a.ringElems - start);
        for (int64_t idx = tid; idx < first; idx += nt) a.ring[start + idx] = F[idx];
        for (int64_t idx = tid; idx < len - first; idx += nt) a.ring[idx] = F[first + idx];
    }
    const int mu = m - k;
    cuComplex* Uo = upd + me.updOff;
    for (int64_t idx = tid; idx < (int64_t)mu * mu; idx += nt) {
        const int i = (int)(idx % mu), j = (int)(idx / mu);
        Uo[idx] = F[(k + i) + (int64_t)m * (k + j)];
    }
}


// ---------------- front kernel v2: blocked, in place, no front scratch
// The panel P (m x k) is factorized where it is stored (device panels, or a per-block buffer that is copied
// to the write-back ring), the update matrix U ((m-k) x (m-k), lower) in place in the update buffer:
//   1. zero P and U, scatter the entries of A into P, pull the children's update matrices into P or U;
//   2. P by left-looking blocks of NB columns: register-tiled update of the block column by the finished
//      columns, LDL^T of its NB x NB diagonal block by one warp, then each row below solved in registers;
//   3. U -= L21 D L21^T in register-tiled TM x TM tiles of the lower triangle, one pass over U.
// U[i, j] -= sum_p L[k+i, p] d_p L[k+j, p] on the lower triangle of U (mu x mu), L and D from the factorized
// panel P (m x k). TM x TM tiles; the 256 threads form 16 x 16, each owning (TM/16)^2 outputs, rows
// contiguous so that the read-modify-write of U is coalesced.
// One tile of the sum acc[x][y] = sum_{pa <= p < pb} L[r0+i, p] d_p L[c0+j, p] for this thread's outputs
// i = tx R + x, j = ty R + y; rows r0+i >= rEnd and columns c0+j >= cEnd count as zero. By 256 threads, in chunks of
// KC pivots staged in shared memory; with DB each chunk's loads are issued before the previous chunk is summed
// (16 more registers: not for the one-block path, inside the level kernel).
template <int TM, bool DB>
__device__ void tileAcc(const cuComplex* P, int m, int pa, int pb, int r0, int rEnd, int c0, int cEnd, cuComplex* sh,
                        cuComplex (&acc)[TM / 16][TM / 16])
{
    constexpr int R = TM / 16, NE = KC * TM / 256;  // NE: staged entries per thread and chunk
    cuComplex (*sA)[TM] = reinterpret_cast<cuComplex (*)[TM]>(sh);            // L[r0+i, p] d_p
    cuComplex (*sB)[TM] = reinterpret_cast<cuComplex (*)[TM]>(sh + KC * TM);  // L[c0+j, p]
    const cuComplex zero = make_cuComplex(0, 0);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
#pragma unroll
    for (int x = 0; x < R; ++x)
#pragma unroll
        for (int y = 0; y < R; ++y) acc[x][y] = zero;
    cuComplex va[NE], vb[NE];  // the next chunk, loaded while the current one is summed
    auto load = [&](int p0) {
#pragma unroll
        for (int q = 0; q < NE; ++q) {
            const int e = tid + 256 * q, ii = e % TM, p = p0 + e / TM;
            va[q] = vb[q] = zero;
            if (p < pb) {
                const cuComplex* col = P + (int64_t)m * p;
                if (r0 + ii < rEnd) va[q] = cmul(col[r0 + ii], P[(int64_t)p * (m + 1)]);
                if (c0 + ii < cEnd) vb[q] = col[c0 + ii];
            }
        }
    };
    if (DB && pa < pb) load(pa);
    for (int p0 = pa; p0 < pb; p0 += KC) {
        if (!DB) load(p0);
#pragma unroll
        for (int q = 0; q < NE; ++q) {
            const int e = tid + 256 * q;
            sA[e / TM][e % TM] = va[q];
            sB[e / TM][e % TM] = vb[q];
        }
        __syncthreads();
        if (DB && p0 + KC < pb) load(p0 + KC);
#pragma unroll 4
        for (int pp = 0; pp < KC; ++pp) {
            cuComplex av[R], bv[R];
#pragma unroll
            for (int x = 0; x < R; ++x) av[x] = sA[pp][tx * R + x];
#pragma unroll
            for (int y = 0; y < R; ++y) bv[y] = sB[pp][ty * R + y];
#pragma unroll
            for (int x = 0; x < R; ++x)
#pragma unroll
                for (int y = 0; y < R; ++y) acc[x][y] = caddmul(acc[x][y], av[x], bv[y]);
        }
        __syncthreads();
    }
}

// tile (ti, tj), tj <= ti, of the Schur update: rows and columns k + (ti, tj) TM, all k pivots
template <int TM, bool DB>
__device__ void schurTileAcc(const cuComplex* P, int m, int k, int ti, int tj, cuComplex* sh,
                             cuComplex (&acc)[TM / 16][TM / 16])
{
    tileAcc<TM, DB>(P, m, 0, k, k + ti * TM, m, k + tj * TM, m, sh, acc);
}

// One tile (ti, tj), tj <= ti, of it applied to U
template <int TM>
__device__ void schurTile(const cuComplex* P, int m, int k, cuComplex* U, int mu, int ti, int tj, cuComplex* sh)
{
    constexpr int R = TM / 16;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    cuComplex acc[R][R];
    schurTileAcc<TM, false>(P, m, k, ti, tj, sh, acc);
#pragma unroll
    for (int y = 0; y < R; ++y) {
        const int j = tj * TM + ty * R + y;
#pragma unroll
        for (int x = 0; x < R; ++x) {
            const int i = ti * TM + tx * R + x;
            if (j < mu && i < mu && i >= j) {
                cuComplex* u = U + i + (int64_t)mu * j;
                u->x -= acc[x][y].x;
                u->y -= acc[x][y].y;
            }
        }
    }
}

// all of its tiles, by one block
template <int TM>
__device__ void schurUpdate(const cuComplex* P, int m, int k, cuComplex* U, int mu, cuComplex* sh)
{
    const int T = (mu + TM - 1) / TM;
    for (int t = 0; t < T * (T + 1) / 2; ++t) {
        int ti = 0;
        while ((ti + 1) * (ti + 2) / 2 <= t) ++ti;
        schurTile<TM>(P, m, k, U, mu, ti, t - ti * (ti + 1) / 2, sh);
    }
}

__device__ __forceinline__ int lowerBound(const int* v, int n, int x)  // first index with v[i] >= x (v increasing)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (v[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// The update matrices of a level's fronts with m - k > kSplitMu, one 64 x 64 tile per block at a time (tiles:
// front, tile row, tile column), after the level's kernel has factorized their panels. A tile is assembled in
// shared memory from the children's update matrices (a child's rows and columns that land in the tile are each
// one contiguous range, since its row map increases; added one child at a time), then L21 D L21^T is
// subtracted and the tile is written once: U is neither zeroed nor read-modified-written in global memory, and
// a large front's update is spread over the GPU. Same sums in the same order as the one-block path.
__global__ void __launch_bounds__(256) schurTileKer(Args a, const int4* tiles, int count)
{
    extern __shared__ __align__(16) cuComplex tileSh[];
    cuComplex (*T)[65] = reinterpret_cast<cuComplex (*)[65]>(tileSh);  // T[row][column] of the tile
    cuComplex* sh = tileSh + 64 * 65;                                  // schurTile's staging
    // a batch of children, in the staging area (free until schurTileAcc): mc, their rows in the tile [y, z),
    // their columns from w to sColEnd, update matrices and row maps
    int4* sPart = reinterpret_cast<int4*>(sh);
    int* sColEnd = reinterpret_cast<int*>(sPart + 32);
    const cuComplex** sUc = reinterpret_cast<const cuComplex**>(sColEnd + 32);
    const int** sRel = reinterpret_cast<const int**>(sUc + 32);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, G = gridDim.x;
    if (blockIdx.x >= count) return;
    int4 qn = tiles[blockIdx.x];  // the next tile and its front, loaded a tile ahead
    SNode next = a.sn[qn.x];
    for (int t = blockIdx.x; t < count; t += G) {
        const int4 q = qn;
        const SNode me = next;
        if (t + G < count) {
            qn = tiles[t + G];
            next = a.sn[qn.x];
        }
        const int m = me.m, k = me.k, mu = m - k, i0 = q.y * 64, j0 = q.z * 64;
        for (int e = tid; e < 64 * 64; e += 256) T[e & 63][e >> 6] = make_cuComplex(0, 0);
        for (int c0 = me.childBeg; c0 < me.childEnd; c0 += 32) {  // children one at a time: a fixed order of the sums
            const int nb = min(32, me.childEnd - c0);
            if (tid < nb) {  // the batch's descriptors and parts of the tile, all at once
                const SNode ch = a.sn[a.children[c0 + tid]];
                const int* tb = a.tileBound + a.tileBoundOff[c0 + tid];
                sPart[tid] = make_int4(ch.m - ch.k, tb[q.y], tb[q.y + 1], tb[q.z]);
                sColEnd[tid] = tb[q.z + 1];
                sUc[tid] = a.upd + ch.updOff;
                sRel[tid] = a.relind + ch.relOff;
            }
            __syncthreads();
            for (int x = 0; x < nb; ++x) {
                const int4 pt = sPart[x];
                const int mc = pt.x, ra = pt.y, nr = pt.z - pt.y, ca = pt.w, nc = sColEnd[x] - pt.w;
                const cuComplex* Uc = sUc[x];
                const int* rel = sRel[x];
                const unsigned magic = 0xffffffffu / max(nr, 1) + 1;  // umulhi divides by nr (nr = 1 has e < nc)
#pragma unroll 4
                for (int e = tid; e < nr * nc; e += 256) {
                    const int bb = nr > 1 ? (int)__umulhi((unsigned)e, magic) : e, i = ra + e - bb * nr, b = ca + bb;
                    if (i >= b) {  // the child's lower triangle
                        const cuComplex u = __ldcg(Uc + i + (int64_t)mc * b);
                        cuComplex& d = T[__ldg(rel + i) - k - i0][__ldg(rel + b) - k - j0];
                        d.x += u.x;
                        d.y += u.y;
                    }
                }
                __syncthreads();
            }
        }
        cuComplex acc[4][4];
        schurTileAcc<64, true>(a.panels + me.panelOff, m, k, q.y, q.z, sh, acc);
        cuComplex* U = a.upd + me.updOff;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            const int j = ty * 4 + y;
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int i = tx * 4 + x;
                if (j0 + j < mu && i0 + i < mu && i0 + i >= j0 + j)
                    U[(i0 + i) + (int64_t)mu * (j0 + j)] = make_cuComplex(T[i][j].x - acc[x][y].x, T[i][j].y - acc[x][y].y);
            }
        }
        __syncthreads();  // T is reused by the next tile
    }
}

// the first row of each child of a front with Schur tiles that lands in each tile of the front's update matrix:
// tileBound[tileBoundOff[ci] + t] = first b with rel[b] >= k + 64 t, t = 0 .. T; one warp per front, once per pattern
__global__ void tileBoundsKer(Args a, const int* fronts, int count, int* tileBound)
{
    const int lane = threadIdx.x & 31, nw = (gridDim.x * blockDim.x) >> 5;
    for (int x = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; x < count; x += nw) {
        const SNode me = a.sn[fronts[x]];
        const int T = (me.m - me.k + 63) / 64;
        for (int ci = me.childBeg; ci < me.childEnd; ++ci) {
            const SNode ch = a.sn[a.children[ci]];
            for (int t = lane; t <= T; t += 32)
                tileBound[a.tileBoundOff[ci] + t] = lowerBound(a.relind + ch.relOff, ch.m - ch.k, me.k + 64 * t);
        }
    }
}

// A stepped panel (k > kStepW, level schedule) goes right-looking in block columns of kStepW: the level kernel
// assembles the front and factorizes block 0, then each step s >= 1 is two launches over the level's stepped
// fronts. panelTileKer subtracts block s - 1 from the columns [s kStepW, k), one 64 x 64 tile per block at a time
// (tiles: front, tile row, tile column, s; rows >= columns), so that the bulk of a large panel's work is spread
// over the GPU instead of one block per front; panelStepKer then factorizes block s, one block per front.
__global__ void __launch_bounds__(256) panelTileKer(Args a, const int4* tiles, int count)
{
    __shared__ __align__(16) cuComplex sh[2 * KC * 64];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    for (int t = blockIdx.x; t < count; t += gridDim.x) {
        const int4 q = tiles[t];
        const SNode me = a.sn[q.x];
        const int m = me.m, k = me.k, b = q.w * kStepW, r0 = b + q.y * 64, c0 = b + q.z * 64;
        cuComplex* P = a.panels + me.panelOff;
        cuComplex acc[4][4];
        tileAcc<64, true>(P, m, b - kStepW, b, r0, m, c0, k, sh, acc);  // ends with a barrier: sh is free again
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            const int c = c0 + ty * 4 + y;
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int i = r0 + tx * 4 + x;
                if (c < k && i < m && i >= c) {
                    cuComplex* f = P + i + (int64_t)m * c;
                    f->x -= acc[x][y].x;
                    f->y -= acc[x][y].y;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256) panelStepKer(Args a, const int* list, int count, int s)
{
    __shared__ __align__(16) cuComplex sh[kShElems];
    const int b = s * kStepW;
    for (int t = blockIdx.x; t < count; t += gridDim.x) {
        const SNode me = a.sn[list[t]];
        panelFactor(a.panels + me.panelOff + b + (int64_t)me.m * b, me.m, me.m - b, min(kStepW, me.k - b), sh,
                    a.zeroPivots);  // ends with a barrier
    }
}

// Front r with kernel v2. scr: this block's panel buffer when the panel goes to the write-back ring.
__device__ void factorFront2(int r, const Args& a, cuComplex* scr, unsigned long long& wbOff,
                             unsigned long long& freedSeen)
{
    __shared__ __align__(16) cuComplex sh[kShElems];
    __shared__ unsigned long long sOff;
    const cuComplex zero = make_cuComplex(0, 0);
    const SNode me = a.sn[r];
    const int m = me.m, k = me.k, mu = m - k, tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
    const int64_t len = (int64_t)m * k;
    cuComplex* P = a.dma ? scr : a.panels + me.panelOff;
    cuComplex* U = a.upd + me.updOff;
    unsigned long long off = 0;
    if (a.dma && tid == 0) off = atomicAdd(a.allocHead, (unsigned long long)len);  // used after the assembly
    const bool stepped = a.stepPanels && k > kStepW;  // block 0 here, the rest by panelTileKer and panelStepKer
    const bool split = a.splitSchur && (mu > kSplitMu || (stepped && mu > 0));  // U by schurTileKer
    for (int64_t idx = tid; idx < len; idx += nt) P[idx] = zero;
    if (!split)
        for (int j = warp; j < mu; j += nw)
            for (int i = j + lane; i < mu; i += 32) U[i + (int64_t)mu * j] = zero;
    __syncthreads();
    for (int64_t e0 = me.aBeg; e0 < me.aEnd; e0 += 4 * nt) {  // distinct positions, all in the panel
        int pos[4];
        cuComplex v[4], f[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // every load before any store: the positions are distinct
            const int64_t e = e0 + tid + q * nt;
            pos[q] = e < me.aEnd ? __ldg(a.aPos + e) : -1;
            v[q] = e < me.aEnd ? __ldg(a.aVal + e) : zero;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) f[q] = pos[q] >= 0 ? P[pos[q]] : zero;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (pos[q] >= 0) P[pos[q]] = make_cuComplex(f[q].x + v[q].x, f[q].y + v[q].y);
    }
    if (a.dma && tid == 0) {  // as in factorFront
        while (off + len > freedSeen + a.ringElems) {
            freedSeen = *(volatile const unsigned long long*)a.freed;
            if (off + len > freedSeen + a.ringElems) __nanosleep(200);
        }
        sOff = off;
        wbOff = off;
        a.hostOff[r] = (int64_t)off;
    }
    __syncthreads();
    int* sRel = reinterpret_cast<int*>(sh);  // a child's row map (mc < 1024); sh is free until panelFactor
    for (int ci = me.childBeg; ci < me.childEnd; ++ci) {  // children one at a time: no write conflicts
        const SNode ch = a.sn[a.children[ci]];
        const int mc = ch.m - ch.k;
        const cuComplex* Uc = a.upd + ch.updOff;
        for (int i = tid; i < mc; i += nt) sRel[i] = __ldg(a.relind + ch.relOff + i);
        __syncthreads();
        // its columns b < nb, the ones that land in the panel when U is left to schurTileKer (rel increases),
        // entry e = i + mc b over all threads; distinct entries go to distinct places, so every load of a
        // batch is issued before any store
        const int nb = split ? lowerBound(sRel, mc, k) : mc, len = mc * nb;
        const unsigned magic = 0xffffffffu / mc + 1;  // umulhi divides by mc (mc = 1 has e = 0)
        for (int e0 = 0; e0 < len; e0 += 4 * nt) {
            cuComplex u[4], f[4];
            cuComplex* dst[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int e = e0 + tid + q * nt, b = (int)__umulhi((unsigned)e, magic), i = e - b * mc;
                dst[q] = nullptr;
                u[q] = zero;
                if (e < len && i >= b) {
                    u[q] = __ldcg(Uc + e);  // written by another block: bypass L1
                    const int ri = sRel[i], rb = sRel[b];
                    dst[q] = rb < k ? P + ri + (int64_t)m * rb : U + (ri - k) + (int64_t)mu * (rb - k);
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) f[q] = dst[q] ? *dst[q] : zero;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                if (dst[q]) *dst[q] = make_cuComplex(f[q].x + u[q].x, f[q].y + u[q].y);
        }
        __syncthreads();
    }
    panelFactor(P, m, m, stepped ? kStepW : k, sh, a.zeroPivots);
    if (split || stepped) {
    } else if (mu > 48) schurUpdate<64>(P, m, k, U, mu, sh);
    else if (mu > 16) schurUpdate<32>(P, m, k, U, mu, sh);
    else if (mu > 0) schurUpdate<16>(P, m, k, U, mu, sh);
    if (a.dma) {  // into the ring, split at its end; committed by the caller
        const unsigned long long start = sOff % a.ringElems;
        const int64_t first = (int64_t)min((unsigned long long)len, a.ringElems - start);
        for (int64_t idx = tid; idx < first; idx += nt) a.ring[start + idx] = P[idx];
        for (int64_t idx = tid; idx < len - first; idx += nt) a.ring[idx] = P[first + idx];
    }
}

// ---------------- block fronts with m <= 64 (level schedule): one warp each, the front in shared memory
// Its lower triangle packed by rows (row i at i (i + 1) / 2). Synchronized only within the warp, so the many small
// fronts of the lowest levels no longer pay a block's barriers and global round trips at every phase. Every phase
// is a latency chain on one warp, so each is laid out for independent work on all lanes:
// - the next front's node is loaded while this one is factorized;
// - a child's update matrix is added lane-parallel over its lower triangle, folded into column pairs (q, n-1-q) of
//   n+1 entries each, with all loads independent; rel comes from the registers of the lanes that loaded it;
// - the pivots go in blocks of kWfNb: the block's columns are factorized in registers (lane owns rows lane and
//   lane + 32), then the trailing triangle, folded the same way, takes one rank-kWfNb update per entry.
// MAXM is 48 or 64: the smaller front doubles the warps per SM.
constexpr int kWarpFrontM = 64, kWfNb = 4;
__device__ __forceinline__ int triIdx(int i, int c) { return i * (i + 1) / 2 + c; }

// entry e of a lower triangle of order n folded into column pairs: row i >= column c; false for the duplicate
// half of the middle column when n is odd. magic = 0xffffffff / (n + 1) + 1 makes umulhi a division by n + 1.
__device__ __forceinline__ bool foldedEntry(int e, int n, unsigned magic, int& i, int& c)
{
    const int q = (int)__umulhi((unsigned)e, magic), r = e - q * (n + 1);
    const bool first = r < n - q;
    c = first ? q : n - 1 - q;
    i = first ? q + r : c + r - (n - q);
    return first || 2 * q != n - 1;
}

template <int MAXM>
__global__ void __launch_bounds__(32) warpFrontKer(Args a, const int* list, int count)
{
    __shared__ cuComplex F[MAXM * (MAXM + 1) / 2];
    __shared__ __align__(16) cuComplex Ub[MAXM][kWfNb];  // the block's columns before the division, by row
    const cuComplex zero = make_cuComplex(0, 0);
    const int lane = threadIdx.x, i0 = lane, i1 = lane + 32;
    SNode next;
    if (blockIdx.x < count) next = a.sn[__ldg(list + blockIdx.x)];
    for (int t = blockIdx.x; t < count; t += gridDim.x) {
        const SNode me = next;
        if (t + gridDim.x < count) next = a.sn[__ldg(list + t + gridDim.x)];
        const int m = me.m, k = me.k, mu = m - k;
        for (int e = lane; e < m * (m + 1) / 2; e += 32) F[e] = zero;
        __syncwarp();
#pragma unroll 4
        for (int64_t e = me.aBeg + lane; e < me.aEnd; e += 32) {  // distinct positions in the panel's lower part
            const int pos = __ldg(a.aPos + e), i = pos % m, j = pos / m;
            const cuComplex v = __ldg(a.aVal + e);
            if (i >= j) {
                F[triIdx(i, j)].x += v.x;
                F[triIdx(i, j)].y += v.y;
            }
        }
        for (int c0 = me.childBeg; c0 < me.childEnd; c0 += 32) {  // children one at a time, their nodes in parallel
            int cm = 0, crel = 0;
            int64_t coff = 0;
            if (c0 + lane < me.childEnd) {
                const SNode ch = a.sn[__ldg(a.children + c0 + lane)];
                cm = ch.m - ch.k;
                crel = ch.relOff;
                coff = ch.updOff;
            }
            for (int x = 0; x < min(32, me.childEnd - c0); ++x) {
                const int mc = __shfl_sync(0xffffffffu, cm, x), ro = __shfl_sync(0xffffffffu, crel, x);
                const cuComplex* Uc = a.upd + __shfl_sync(0xffffffffu, coff, x);
                if (mc == 0) continue;
                const int r0 = lane < mc ? __ldg(a.relind + ro + lane) : 0;
                const int r1 = MAXM > 32 && lane + 32 < mc ? __ldg(a.relind + ro + lane + 32) : 0;
                const int len = (mc + 1) / 2 * (mc + 1);
                const unsigned magic = 0xffffffffu / (mc + 1) + 1;
                __syncwarp();  // the previous child is added
#pragma unroll 4
                for (int e0 = 0; e0 < len; e0 += 32) {  // all lanes take part in the shuffles
                    int i = 0, b = 0;
                    const bool ok = foldedEntry(e0 + lane, mc, magic, i, b) && e0 + lane < len;
                    cuComplex u = zero;
                    if (ok) u = __ldcg(Uc + i + (int64_t)mc * b);
                    int ri = __shfl_sync(0xffffffffu, r0, i & 31), rb = __shfl_sync(0xffffffffu, r0, b & 31);
                    if (MAXM > 32) {
                        const int si = __shfl_sync(0xffffffffu, r1, i & 31), sb = __shfl_sync(0xffffffffu, r1, b & 31);
                        ri = i < 32 ? ri : si;
                        rb = b < 32 ? rb : sb;
                    }
                    if (ok) {  // distinct entries go to distinct parent entries, as rel increases
                        F[triIdx(ri, rb)].x += u.x;
                        F[triIdx(ri, rb)].y += u.y;
                    }
                }
            }
        }
        __syncwarp();
        const bool in0 = i0 < m, in1 = MAXM > 32 && i1 < m;
        for (int j0 = 0; j0 < k; j0 += kWfNb) {
            const int nb = min(kWfNb, k - j0);
            cuComplex u0[kWfNb], u1[kWfNb], l0[kWfNb], l1[kWfNb];  // the block's columns of my rows: as is, and / d
#pragma unroll
            for (int p = 0; p < kWfNb; ++p) {
                const int c = j0 + p;
                u0[p] = p < nb && in0 && i0 >= c ? F[triIdx(i0, c)] : zero;
                u1[p] = p < nb && in1 && i1 >= c ? F[triIdx(i1, c)] : zero;
                l0[p] = l1[p] = zero;
            }
#pragma unroll
            for (int p = 0; p < kWfNb; ++p) {
                if (p >= nb) break;
                const int j = j0 + p;
                cuComplex d = j < 32 ? shflC(u0[p], j) : shflC(u1[p], j - 32);
                if (d.x == 0 && d.y == 0) {
                    d = make_cuComplex(1e-30f, 0);
                    if (lane == 0) atomicAdd(a.zeroPivots, 1u);
                }
                const cuComplex dinv = crcp(d);
                if (i0 > j) l0[p] = cmul(u0[p], dinv);
                if (i1 > j) l1[p] = cmul(u1[p], dinv);
#pragma unroll
                for (int q = p + 1; q < kWfNb; ++q) {
                    if (q >= nb) break;
                    const int c = j0 + q;
                    const cuComplex ac = c < 32 ? shflC(u0[p], c) : shflC(u1[p], c - 32);
                    if (i0 >= c) u0[q] = csubmul(u0[q], l0[p], ac);
                    if (i1 >= c) u1[q] = csubmul(u1[q], l1[p], ac);
                }
                if (i0 == j) u0[p] = d;
                if (i1 == j) u1[p] = d;
            }
            const int base = j0 + nb;
#pragma unroll
            for (int p = 0; p < kWfNb; ++p) {
                const int j = j0 + p;
                if (p < nb && in0 && i0 >= j) F[triIdx(i0, j)] = i0 == j ? u0[p] : l0[p];
                if (p < nb && in1 && i1 >= j) F[triIdx(i1, j)] = i1 == j ? u1[p] : l1[p];
                if (in0 && i0 >= base) Ub[i0][p] = u0[p];  // zero for p >= nb
                if (in1 && i1 >= base) Ub[i1][p] = u1[p];
            }
            __syncwarp();
            // the trailing triangle: F(i, c) -= sum_p l(i, p) u(c, p), each entry by one lane
            const int n = m - base, len = (n + 1) / 2 * (n + 1);
            const unsigned magic = 0xffffffffu / (n + 1) + 1;
#pragma unroll 2
            for (int e = lane; e < len; e += 32) {
                int ii, cc;
                if (!foldedEntry(e, n, magic, ii, cc)) continue;
                const int i = base + ii, c = base + cc;
                const cuComplex* li = F + triIdx(i, j0);
                const float4 ua = reinterpret_cast<const float4*>(Ub[c])[0], ub = reinterpret_cast<const float4*>(Ub[c])[1];
                const cuComplex uc[4] = {make_cuComplex(ua.x, ua.y), make_cuComplex(ua.z, ua.w), make_cuComplex(ub.x, ub.y),
                                         make_cuComplex(ub.z, ub.w)};
                cuComplex f = F[triIdx(i, c)];
#pragma unroll
                for (int p = 0; p < kWfNb; ++p)
                    if (p < nb) f = csubmul(f, li[p], uc[p]);
                F[triIdx(i, c)] = f;
            }
            __syncwarp();
        }
        cuComplex* P = a.panels + me.panelOff;  // the panel (zero above the diagonal) and the update matrix
        for (int j = 0; j < k; ++j) {
            if (in0) P[i0 + (int64_t)m * j] = i0 >= j ? F[triIdx(i0, j)] : zero;
            if (in1) P[i1 + (int64_t)m * j] = i1 >= j ? F[triIdx(i1, j)] : zero;
        }
        cuComplex* U = a.upd + me.updOff;
        for (int c = k; c < m; ++c) {
            if (in0 && i0 >= c) U[(i0 - k) + (int64_t)mu * (c - k)] = F[triIdx(i0, c)];
            if (in1 && i1 >= c) U[(i1 - k) + (int64_t)mu * (c - k)] = F[triIdx(i1, c)];
        }
        __syncwarp();  // F is reused
    }
}
static_assert(kWfNb == 4, "the trailing update reads a row of Ub as two float4");

// ---------------- small fronts: one warp per subtree
// A subtree whose fronts all have m <= 32 is factorized by one warp in postorder, front after front, with
// the front in shared memory (lane i owns row i) and no synchronization beyond the warp: the subtree's
// update matrices are written and read by the same warp. These subtrees run before the block kernels.
constexpr int kSmallWarps = 4;  // warps per block of smallKer

__device__ void smallFront(int r, const Args& a, cuComplex (*F)[33], unsigned long long& freedSeen, int64_t arena)
{
    const cuComplex zero = make_cuComplex(0, 0);
    const int lane = threadIdx.x & 31;
    const SNode me = a.sn[r];
    const int m = me.m, k = me.k, mu = m - k;
    const int64_t len = (int64_t)m * k;
    unsigned long long off = 0;
    if (a.dma && lane == 0) off = atomicAdd(a.allocHead, (unsigned long long)len);
#pragma unroll
    for (int c = 0; c < 32; ++c) F[lane][c] = zero;
    __syncwarp();
    for (int64_t e = me.aBeg + lane; e < me.aEnd; e += 32) {  // distinct positions, all in the panel
        const int pos = a.aPos[e], i = pos % m, j = pos / m;
        const cuComplex v = a.aVal[e];
        F[i][j].x += v.x;
        F[i][j].y += v.y;
    }
    __syncwarp();
    for (int ci = me.childBeg; ci < me.childEnd; ++ci) {  // children one at a time
        const SNode ch = a.sn[a.children[ci]];
        const int mc = ch.m - ch.k;
        const cuComplex* Uc = a.upd + (ch.updOff >= 0 ? ch.updOff : arena - 1 - ch.updOff);
        const int ri = lane < mc ? a.relind[ch.relOff + lane] : 0;
        for (int b = 0; b < mc; ++b) {
            const int rb = __shfl_sync(0xffffffffu, ri, b);
            if (lane >= b && lane < mc) {
                const cuComplex u = __ldcg(Uc + lane + (int64_t)mc * b);
                F[ri][rb].x += u.x;
                F[ri][rb].y += u.y;
            }
        }
        __syncwarp();
    }
    for (int j = 0; j < k; ++j) {
        cuComplex d = F[j][j];
        if (d.x == 0 && d.y == 0) {
            d = make_cuComplex(1e-30f, 0);
            if (lane == 0) atomicAdd(a.zeroPivots, 1u);
        }
        const bool mine = lane > j && lane < m;
        const cuComplex aij = mine ? F[lane][j] : zero;  // before the division
        const cuComplex l = cmul(aij, crcp(d));
        __syncwarp();
        for (int c = j + 1; c < m; ++c) {
            const cuComplex ac = shflC(aij, c);
            if (mine && c <= lane) F[lane][c] = csubmul(F[lane][c], l, ac);
        }
        if (mine) F[lane][j] = l;
        if (lane == j) F[j][j] = d;
        __syncwarp();
    }
    // panel (the upper part of its diagonal block is zero) and update matrix
    if (!a.dma) {
        cuComplex* P = a.panels + me.panelOff;
        if (lane < m)
            for (int j = 0; j < k; ++j) P[lane + (int64_t)m * j] = F[lane][j];
    } else {
        if (lane == 0) {  // as in factorFront
            while (off + len > freedSeen + a.ringElems) {
                freedSeen = *(volatile const unsigned long long*)a.freed;
                if (off + len > freedSeen + a.ringElems) __nanosleep(200);
            }
            a.hostOff[r] = (int64_t)off;
        }
        off = __shfl_sync(0xffffffffu, off, 0);
        if (lane < m)
            for (int j = 0; j < k; ++j) a.ring[(off + lane + (unsigned long long)m * j) % a.ringElems] = F[lane][j];
    }
    cuComplex* U = a.upd + (me.updOff >= 0 ? me.updOff : arena - 1 - me.updOff);
    for (int c = k; c < m; ++c)
        if (lane >= c && lane < m) U[(lane - k) + (int64_t)mu * (c - k)] = F[lane][c];
    __syncwarp();  // F is reused, U read by the parent in this warp
    if (a.dma && lane == 0) {
        __threadfence();
        wbCommit(a, off, len);
    }
}

// subtrees: [first, root] ranges of region indices (postorder), largest first; warps take them in turn
__global__ void smallKer(Args a, const int2* subtrees, int count, int* next)
{
    __shared__ cuComplex Fs[kSmallWarps][32][33];
    const int lane = threadIdx.x & 31;
    cuComplex (*F)[33] = Fs[threadIdx.x >> 5];
    unsigned long long freedSeen = 0;
    const int64_t arena = a.arenaBase + (int64_t)(blockIdx.x * kSmallWarps + (threadIdx.x >> 5)) * a.arenaPerWarp;
    for (;;) {
        int t = 0;
        if (lane == 0) t = atomicAdd(next, 1);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= count) return;
        const int2 range = subtrees[t];
        for (int r = range.x; r <= range.y; ++r) smallFront(r, a, F, freedSeen, arena);
    }
}

template <int V>
__device__ __forceinline__ void front(int r, const Args& a, cuComplex* F, cuComplex* w, unsigned long long& wbOff,
                                      unsigned long long& freedSeen)
{
    if constexpr (V == 1) factorFront(r, a, F, w, wbOff, freedSeen);
    else factorFront2(r, a, F, wbOff, freedSeen);
}

template <int V>
__global__ void levelKer(Args a, const int* list, int count)
{
    extern __shared__ cuComplex w[];
    cuComplex* F = a.scratch + (int64_t)blockIdx.x * a.frontStride;
    unsigned long long wbOff = 0, freedSeen = 0;
    for (int t = blockIdx.x; t < count; t += gridDim.x) {
        front<V>(list[t], a, F, w, wbOff, freedSeen);
        __syncthreads();
        if (a.dma && threadIdx.x == 0) {
            __threadfence();  // the panel is in memory before the copy engine may read it
            wbCommit(a, wbOff, (int64_t)a.sn[list[t]].m * a.sn[list[t]].k);
        }
    }
}

// queue[0..R): region indices in the order they became ready (-1 = not yet); head: next slot to take;
// tail: next slot to fill; cnt[r]: children of r not finished yet
template <int V>
__global__ void persistentKer(Args a, int R, int* queue, int* head, int* tail, int* cnt)
{
    extern __shared__ cuComplex w[];
    __shared__ int cur;
    cuComplex* F = a.scratch + (int64_t)blockIdx.x * a.frontStride;
    unsigned long long wbOff = 0, freedSeen = 0;
    for (;;) {
        if (threadIdx.x == 0) {
            const int slot = atomicAdd(head, 1);
            int v = -1;
            if (slot < R)  // the slot is filled once its supernode's last child finishes
                while ((v = atomicAdd(&queue[slot], 0)) < 0) __nanosleep(100);
            cur = v;
        }
        __syncthreads();
        const int r = cur;
        if (r < 0) return;
        front<V>(r, a, F, w, wbOff, freedSeen);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();  // panel and update matrix visible before the parent (and the copy engine) can read them
            if (a.dma) wbCommit(a, wbOff, (int64_t)a.sn[r].m * a.sn[r].k);
            const int p = a.sn[r].parent;
            if (p >= 0 && atomicSub(&cnt[p], 1) == 1) atomicExch(&queue[atomicAdd(tail, 1)], p);
        }
        __syncthreads();
    }
}

inline int gridOf(int64_t n) { return (int)std::max<int64_t>(1, std::min<int64_t>((n + 255) / 256, 1 << 20)); }

__global__ void slotOfKer(int64_t len, const int32_t* aSrc, int32_t* slot)
{
    for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < len; e += (int64_t)gridDim.x * blockDim.x)
        slot[aSrc[e]] = (int32_t)e;
}

// values of a chunk of CSR slots into their assembly slots, in complex single precision
__global__ void placeValuesKer(const cuDoubleComplex* v, const int32_t* slot, int64_t len, cuComplex* aVal)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < len; i += (int64_t)gridDim.x * blockDim.x) {
        const int e = slot[i];
        if (e >= 0) aVal[e] = make_cuComplex((float)v[i].x, (float)v[i].y);
    }
}

template <class T>
std::vector<T> readFile(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die(("cannot open " + path).c_str());
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<T> v(bytes / sizeof(T));
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) die(("short read " + path).c_str());
    std::fclose(f);
    return v;
}

}  // namespace

int main(int argc, char** argv)
{
    const auto tMain = Clock::now();
    std::string userPerm, userTree;
    std::vector<std::string> dirs;
    bool writeback = false, cpuRef = false, stats = false, whole = false, doCheck = true;
    std::string schedule = "both", symbolic = "gpu";
    int threads = 256, blocksPerSm = 0, reps = 3, kernel = 2, smallM = 32, largeM = 1024, nStreams = 8, nbOpt = 0;
    int solveReps = 0, solveNrOpt = 2;
    std::string solvePrecision = "double-float";
    std::string updPlanOpt = "on", graphsOpt = "on";  // of FGMRES's preconditioner: the vectors and sums of the front solve
    double ringMb = 1024, chunkMb = 4;
    MixedOptions mopt;  // the full solve: FGMRES, double-double refinement, recycling (testbench/mixed_solve.h)
    for (int a = 1; a < argc; ++a) {
        if (parseMixedOption(a, argc, argv, mopt)) continue;
        if (!std::strcmp(argv[a], "--writeback")) writeback = true;
        else if (!std::strcmp(argv[a], "--stats")) stats = true;
        else if (!std::strcmp(argv[a], "--whole")) whole = true;
        else if (!std::strcmp(argv[a], "--no-verify")) doCheck = false;
        else if (!std::strcmp(argv[a], "--user-perm") && a + 1 < argc) userPerm = argv[++a];
        else if (!std::strcmp(argv[a], "--user-tree") && a + 1 < argc) userTree = argv[++a];
        else if (!std::strcmp(argv[a], "--schedule") && a + 1 < argc) schedule = argv[++a];
        else if (!std::strcmp(argv[a], "--symbolic") && a + 1 < argc) symbolic = argv[++a];
        else if (!std::strcmp(argv[a], "--solve") && a + 1 < argc) solveReps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--solve-nr") && a + 1 < argc) solveNrOpt = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--solve-precision") && a + 1 < argc) solvePrecision = argv[++a];
        else if (!std::strcmp(argv[a], "--upd-plan") && a + 1 < argc) updPlanOpt = argv[++a];
        else if (!std::strcmp(argv[a], "--graphs") && a + 1 < argc) graphsOpt = argv[++a];
        else if (!std::strcmp(argv[a], "--large-m") && a + 1 < argc) largeM = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--streams") && a + 1 < argc) nStreams = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--nb") && a + 1 < argc) nbOpt = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--kernel") && a + 1 < argc) kernel = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--small-m") && a + 1 < argc) smallM = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--ring-mb") && a + 1 < argc) ringMb = std::atof(argv[++a]);
        else if (!std::strcmp(argv[a], "--chunk-mb") && a + 1 < argc) chunkMb = std::atof(argv[++a]);
        else if (!std::strcmp(argv[a], "--cpu-ref")) cpuRef = true;
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc) threads = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--blocks-per-sm") && a + 1 < argc) blocksPerSm = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--reps") && a + 1 < argc) reps = std::atoi(argv[++a]);
        else if (argv[a][0] == '-') die((std::string("unknown option ") + argv[a]).c_str());
        else dirs.push_back(argv[a]);
    }
    if (dirs.empty() || userPerm.empty() || userTree.empty()) {
        std::fprintf(stderr, "usage: %s --user-perm P.perm --user-tree P.rowtree [options] <dir> [<dir> ...]\n"
                     "  options: [--whole] [--large-m M] [--streams S] [--nb NB] [--kernel 1|2] [--small-m M] [--writeback]\n"
                     "           [--ring-mb M] [--chunk-mb C] [--threads T] [--blocks-per-sm B] [--reps R]\n"
                     "           [--schedule both|level|persistent] [--symbolic gpu|cholmod|both] [--solve R] [--no-verify]\n"
                     "           [--stats]\n"
                     "           [--cpu-ref]\n", argv[0]);
        return 1;
    }
    std::string opts;  // the run's options, as the report's description
    for (int a = 1; a < argc; ++a)
        if (std::find(dirs.begin(), dirs.end(), argv[a]) == dirs.end()) opts += (opts.empty() ? "" : " ") + std::string(argv[a]);
    RunReport report("ref", opts);
    report.start = tMain;
    CK(cudaFree(nullptr));  // CUDA context
    report.init = since(tMain);

    // ---------------- matrix and plan
    auto t = Clock::now();
    int sym = 0;
    Csr A;
    double conv = 0;
    if (!readMatrixACsr(dirs[0] + "/A.bin", A, sym, &conv)) die("read A failed");
    report.read += since(t) - conv;
    report.convert += conv;
    const int64_t n = A.n;
    report.n = n;
    report.nnz = A.nnz;

    // ---------------- right-hand sides and the FP64 refinement around our factors (--gmres: the full solve)
    int nports = 0;
    std::vector<int64_t> portRows;  // rows where B has an entry: their solution values are the port values
    std::vector<Complex> bHost;
    int32_t *dRow = nullptr, *dCol = nullptr;
    Complex *dVal = nullptr, *dB = nullptr, *dX = nullptr;
    std::unique_ptr<MixedSolver> mixed;
    if (mopt.gmres) {
        if (!whole || writeback) die("--gmres needs --whole and the panels in device memory");
        auto tb = Clock::now();
        bHost = loadDenseB(dirs[0], n, nports, &portRows);
        report.read += since(tb);
        if (nports != 1 && nports != 2 && nports != 4) die("--gmres: 1, 2 or 4 right-hand sides");
        tb = Clock::now();
        CK(cudaMalloc(&dRow, (n + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&dCol, A.nnz * sizeof(int32_t)));
        CK(cudaMalloc(&dVal, A.nnz * sizeof(Complex)));
        CK(cudaMalloc(&dB, n * nports * sizeof(Complex)));
        CK(cudaMalloc(&dX, n * nports * sizeof(Complex)));
        CK(cudaMemcpy(dRow, A.rowPtr.data(), (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dCol, A.col.data(), A.nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, bHost.data(), n * nports * sizeof(Complex), cudaMemcpyHostToDevice));
        report.upload += since(tb);
        // its dd-IR setup (lower-part index) runs in the background during the analysis and the first factorization
        mixed = std::make_unique<MixedSolver>(A, bHost, nports, dRow, dCol, dVal, dB, dX, mopt);
        std::printf("full solve: %d right-hand sides, FP64 FGMRES preconditioned by the front solve%s\n", nports,
                    mopt.ddIr > 0 ? ", inside double-double refinement" : "");
    }
    const auto tAnalysis = Clock::now();
    std::vector<int32_t> perm = readFile<int32_t>(userPerm), tree = readFile<int32_t>(userTree);
    if ((int64_t)perm.size() != n) die("plan does not match the matrix");
    const size_t nleaves = (tree.size() + 1) / 2;
    int64_t leafRows = 0;
    for (size_t b = 0; b < nleaves; ++b) leafRows += tree[b];
    std::vector<int64_t> planPos(n);
    for (int64_t q = 0; q < n; ++q) planPos[perm[q]] = q;  // leaves come first in the plan
    std::printf("n = %lld, nnz(upper) = %lld; plan: %zu leaves holding %lld rows (%.1f%%); read %.1f s\n",
                (long long)n, (long long)A.nnz, nleaves, (long long)leafRows, 100.0 * leafRows / n, since(t));

    // ---------------- symbolic analysis: on the GPU, or CHOLMOD's (the reference), or both and compared
    if (symbolic != "gpu" && symbolic != "cholmod" && symbolic != "both") die("--symbolic: gpu, cholmod or both");
    t = Clock::now();
    SymbolicResult gs;
    SymbolicDevice sdev;  // the analysis on the device, for the front maps
    const bool hostLs = doCheck || cpuRef || symbolic == "both";  // rows on the host: check, CPU reference, comparison
    if (symbolic != "cholmod") gpuSymbolic(n, A.rowPtr.data(), A.col.data(), perm.data(), tree, gs, &sdev, hostLs, true);
    cholmod_common cc;
    cholmod_l_start(&cc);
    cholmod_sparse* S = nullptr;
    cholmod_factor* Lf = nullptr;
    if (symbolic != "gpu") {
        const auto tch = Clock::now();
        S = cholmod_l_allocate_sparse(n, n, A.nnz, 1, 1, -1, CHOLMOD_PATTERN, &cc);  // upper CSR = lower CSC
        std::copy(A.rowPtr.begin(), A.rowPtr.end(), static_cast<int64_t*>(S->p));
        std::copy(A.col.begin(), A.col.end(), static_cast<int64_t*>(S->i));
        cc.nmethods = 1;
        cc.method[0].ordering = CHOLMOD_GIVEN;
        cc.postorder = 1;
        cc.supernodal = CHOLMOD_SUPERNODAL;
        std::vector<int64_t> p64(perm.begin(), perm.end());
        Lf = cholmod_l_analyze_p(S, p64.data(), nullptr, 0, &cc);
        if (!Lf || cc.status < CHOLMOD_OK) die("cholmod_analyze_p failed");
        std::printf("CHOLMOD symbolic: %lld supernodes, nnz(L) %.4g, %.1f s\n", (long long)Lf->nsuper, cc.lnz, since(tch));
    }
    if (symbolic == "both") {  // the GPU result must be CHOLMOD's, entry for entry
        auto same = [](const char* what, const int64_t* a, const std::vector<int64_t>& b, int64_t len) {
            int64_t i = 0;
            while (i < len && a[i] == b[i]) ++i;
            if (i < len) std::printf("  %-9s DIFFERS at %lld of %lld: CHOLMOD %lld, GPU %lld\n", what, (long long)i,
                                     (long long)len, (long long)a[i], (long long)b[i]);
            return i == len;
        };
        bool ok = (int64_t)Lf->nsuper == gs.nsuper;
        if (!ok) std::printf("  supernodes: CHOLMOD %lld, GPU %lld\n", (long long)Lf->nsuper, (long long)gs.nsuper);
        ok = same("Perm", static_cast<const int64_t*>(Lf->Perm), gs.perm, n) && ok;
        ok = same("ColCount", static_cast<const int64_t*>(Lf->ColCount), gs.colCount, n) && ok;
        if (ok) {
            ok = same("super", static_cast<const int64_t*>(Lf->super), gs.super, gs.nsuper + 1) && ok;
            ok = same("pi", static_cast<const int64_t*>(Lf->pi), gs.pi, gs.nsuper + 1) && ok;
            if (ok) ok = same("Ls", static_cast<const int64_t*>(Lf->s), gs.Ls, gs.pi[gs.nsuper]) && ok;
        }
        std::printf("symbolic: the GPU result %s CHOLMOD's\n", ok ? "is identical to" : "DIFFERS from");
        if (!ok) die("GPU symbolic analysis differs from CHOLMOD");
    }
    const bool fromGpu = symbolic != "cholmod";
    const int64_t ns = fromGpu ? gs.nsuper : (int64_t)Lf->nsuper;
    const int64_t* super = fromGpu ? gs.super.data() : static_cast<const int64_t*>(Lf->super);
    const int64_t* pi = fromGpu ? gs.pi.data() : static_cast<const int64_t*>(Lf->pi);
    const int64_t* Ls = fromGpu ? gs.Ls.data() : static_cast<const int64_t*>(Lf->s);
    const int64_t* Perm = fromGpu ? gs.perm.data() : static_cast<const int64_t*>(Lf->Perm);
    const double lnz = fromGpu ? gs.lnz : cc.lnz;
    std::vector<int64_t> invP(n);
    for (int64_t j = 0; j < n; ++j) invP[Perm[j]] = j;
    std::vector<int32_t> sparent;  // parent supernode: the one holding the first row below the columns
    if (fromGpu) sparent = std::move(gs.sparent);
    else {
        std::vector<int32_t> superOf(n);
        for (int64_t s = 0; s < ns; ++s)
            for (int64_t j = super[s]; j < super[s + 1]; ++j) superOf[j] = (int32_t)s;
        sparent.assign(ns, -1);
        for (int64_t s = 0; s < ns; ++s)
            if (pi[s + 1] - pi[s] > super[s + 1] - super[s]) sparent[s] = superOf[Ls[pi[s] + super[s + 1] - super[s]]];
        uploadSymbolic(n, A.rowPtr.data(), A.col.data(), Perm, ns, super, pi, Ls, sdev);
    }
    std::printf("symbolic analysis (%s): %lld supernodes, nnz(L) %.4g, %.1f s\n", symbolic.c_str(), (long long)ns, lnz,
                since(t));

    // ---------------- region: supernodes whose columns all lie in ND leaves
    t = Clock::now();
    auto leafCol = [&](int64_t j) { return whole || planPos[Perm[j]] < leafRows; };
    std::vector<int32_t> reg(ns, -1);
    std::vector<int64_t> regSuper;
    int64_t mixedLS = 0;  // supernodes with both leaf and separator columns
    for (int64_t s = 0; s < ns; ++s) {
        bool all = true, any = false;
        for (int64_t j = super[s]; j < super[s + 1]; ++j) {
            const bool l = leafCol(j);
            all &= l;
            any |= l;
        }
        if (all) { reg[s] = (int32_t)regSuper.size(); regSuper.push_back(s); }
        else if (any) ++mixedLS;
    }
    const int R = (int)regSuper.size();
    std::vector<SNode> sn(R);
    std::vector<std::vector<int>> kids(R);
    int64_t panelTotal = 0, updTotal = 0, regionCols = 0;
    int maxM = 0, roots = 0;
    double flops = 0;
    for (int r = 0; r < R; ++r) {
        const int64_t s = regSuper[r];
        SNode& x = sn[r];
        x.k = (int)(super[s + 1] - super[s]);
        x.m = (int)(pi[s + 1] - pi[s]);
        x.panelOff = panelTotal;
        panelTotal += (int64_t)x.m * x.k;
        x.updOff = updTotal;
        updTotal += (int64_t)(x.m - x.k) * (x.m - x.k);
        x.parent = -1;
        if (sparent[s] >= 0 && reg[sparent[s]] >= 0) {
            x.parent = reg[sparent[s]];
            kids[x.parent].push_back(r);
        }
        if (x.parent < 0) ++roots;
        maxM = std::max(maxM, x.m);
        regionCols += x.k;
        for (int j = 0; j < x.k; ++j) flops += 4.0 * (double)(x.m - j) * (x.m - j);  // complex rank-1 updates
    }
    std::vector<int> children;
    for (int r = 0; r < R; ++r) {
        sn[r].childBeg = (int)children.size();
        children.insert(children.end(), kids[r].begin(), kids[r].end());
        sn[r].childEnd = (int)children.size();
    }
    // on the GPU: positions of each supernode's update rows in its parent's row list (relind), and the
    // assembly map: every entry of P A P^T whose column lies in the region, at its position in that front;
    // its inverse (CSR slot -> assembly slot) places every frequency's values on the device
    const auto tMaps = Clock::now();
    std::vector<int32_t> parentReg(R);
    int64_t relLen = 0;
    for (int r = 0; r < R; ++r) {
        sn[r].relOff = relLen;
        parentReg[r] = sn[r].parent;
        if (sn[r].parent >= 0) relLen += sn[r].m - sn[r].k;
    }
    std::vector<int64_t> relOffs(R), aCount;
    for (int r = 0; r < R; ++r) relOffs[r] = sn[r].relOff;
    int32_t *dRelind = nullptr, *dUrow = nullptr, *dAPos = nullptr, *dASrc = nullptr;
    gpuFrontMaps(sdev, reg, regSuper, parentReg, relOffs, relLen, &dRelind, &dUrow, &dAPos, &dASrc, aCount);
    sdev.release();
    const int64_t aLen = aCount[R];
    int32_t* dSlot = nullptr;  // CSR slot -> assembly slot, -1 for an entry outside the region
    CK(cudaMalloc(&dSlot, std::max<int64_t>(A.nnz, 1) * sizeof(int32_t)));
    CK(cudaMemset(dSlot, 0xff, A.nnz * sizeof(int32_t)));
    if (aLen > 0) slotOfKer<<<gridOf(aLen), 256>>>(aLen, dASrc, dSlot);
    CK(cudaFree(dASrc));
    std::vector<cuComplex> aVal(cpuRef ? aLen : 0);  // host copy for the CPU reference
    std::vector<int> relind, aPos;  // host copies for the CPU reference only
    if (cpuRef) {
        relind.resize(relLen);
        aPos.resize(aLen);
        CK(cudaMemcpy(relind.data(), dRelind, relLen * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(aPos.data(), dAPos, aLen * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    std::printf("front maps on the GPU: %lld update rows, %lld entries of A, %.2f s\n", (long long)relLen, (long long)aLen,
                since(tMaps));
    for (int r = 0; r < R; ++r) { sn[r].aBeg = aCount[r]; sn[r].aEnd = aCount[r + 1]; }
    // --whole: fronts that are large (m > largeM) or lie above one go to the large-front path (cuBLAS, several
    // streams, after the block kernels); they never belong to a small-front subtree
    std::vector<char> big(R, 0);
    std::vector<int> bigOrder;  // postorder
    if (whole)
        for (int r = 0; r < R; ++r) {
            big[r] = sn[r].m > largeM;
            for (int c : kids[r]) big[r] = big[r] || big[c];
            if (big[r]) bigOrder.push_back(r);
        }
    // small fronts: subtrees whose fronts all have m <= smallM go to smallKer, one warp each
    if (smallM > 32) die("--small-m is at most 32");
    std::vector<char> isS(R, 0);
    std::vector<int> fd(R), subSize(R, 1);
    std::vector<int2> subtrees;
    int smallFronts = 0;
    for (int r = 0; r < R; ++r) {  // postorder: children before parents
        isS[r] = sn[r].m <= smallM && !big[r];
        fd[r] = r;
        for (int c : kids[r]) {
            isS[r] = isS[r] && isS[c];
            fd[r] = std::min(fd[r], fd[c]);
            subSize[r] += subSize[c];
        }
    }
    for (int r = 0; r < R; ++r) {
        if (!isS[r] || (sn[r].parent >= 0 && isS[sn[r].parent])) continue;
        if (r - fd[r] + 1 != subSize[r]) die("region supernodes are not in postorder");
        subtrees.push_back(make_int2(fd[r], r));
        smallFronts += subSize[r];
    }
    std::stable_sort(subtrees.begin(), subtrees.end(), [](int2 x, int2 y) { return x.y - x.x > y.y - y.x; });
    const int RL = R - smallFronts - (int)bigOrder.size();  // fronts of the block kernels
    // levels (leaves 0) of the block-kernel fronts for the level-by-level schedule
    std::vector<int> level(R, 0);
    int nlev = 0;
    for (int r = 0; r < R; ++r) {
        if (isS[r] || big[r]) continue;
        for (int c : kids[r])
            if (!isS[c]) level[r] = std::max(level[r], level[c] + 1);
        nlev = std::max(nlev, level[r] + 1);
    }
    std::vector<int> levelPtr(nlev + 1, 0), levelList(std::max(RL, 1));
    for (int r = 0; r < R; ++r)
        if (!isS[r] && !big[r]) levelPtr[level[r] + 1]++;
    for (int l = 0; l < nlev; ++l) levelPtr[l + 1] += levelPtr[l];
    // each level: its fronts with m <= 48, then 48 < m <= kWarpFrontM (warpFrontKer with the level schedule), then
    // the others
    std::vector<int> levelW48(nlev, 0), levelW64(nlev, 0);
    const bool warpFronts = !writeback;
    {
        auto cls = [&](int r) { return !warpFronts || sn[r].m > kWarpFrontM ? 2 : sn[r].m > 48 ? 1 : 0; };
        for (int r = 0; r < R; ++r)
            if (!isS[r] && !big[r] && cls(r) < 2) (cls(r) ? levelW64 : levelW48)[level[r]]++;
        std::vector<int> next[3];
        for (int l = 0; l < nlev; ++l) {
            next[0].push_back(levelPtr[l]);
            next[1].push_back(levelPtr[l] + levelW48[l]);
            next[2].push_back(levelPtr[l] + levelW48[l] + levelW64[l]);
        }
        for (int r = 0; r < R; ++r)
            if (!isS[r] && !big[r]) levelList[next[cls(r)][level[r]]++] = r;
    }
    const double GB = 1024.0 * 1024 * 1024;
    std::printf("region: %d supernodes (%d mixed leaf/separator ones excluded), %lld columns (%.1f%% of n), %d roots, "
                "%d levels, largest front %d\n", R, (int)mixedLS, (long long)regionCols, 100.0 * regionCols / n, roots,
                nlev, maxM);
    std::printf("region work: %.3g real flops in the partial factorizations; L panels %.2f GB, update matrices "
                "%.2f GB, %lld entries of A; setup %.1f s\n",
                flops, panelTotal * 8.0 / GB, updTotal * 8.0 / GB, (long long)aLen, since(t));
    if (stats) {  // where the work is, by front size m: count, share of flops, mean pivots k, Schur-update share
        constexpr int NBK = 12;
        const int edges[NBK] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 1 << 30};
        double cnt[NBK] = {}, fl[NBK] = {}, kk[NBK] = {}, schur[NBK] = {}, mem[NBK] = {};
        for (int r = 0; r < R; ++r) {
            int b = 0;
            while (sn[r].m > edges[b]) ++b;
            double f = 0;
            for (int j = 0; j < sn[r].k; ++j) f += 4.0 * (double)(sn[r].m - j) * (sn[r].m - j);
            const double mu = sn[r].m - sn[r].k;
            cnt[b] += 1, fl[b] += f, kk[b] += sn[r].k, schur[b] += 4.0 * mu * mu * sn[r].k;
            mem[b] += 8.0 * ((double)sn[r].m * sn[r].k + mu * mu);  // panel + update matrix, bytes
        }
        std::printf("  front m      fronts   flops%%  mean k  Schur%%  flop/byte(panel+update)\n");
        for (int b = 0, lo = 1; b < NBK; lo = edges[b] + 1, ++b)
            if (cnt[b] > 0)
                std::printf("  %4d-%-6s %9.0f  %6.1f  %6.1f  %6.1f  %6.1f\n", lo,
                            b == NBK - 1 ? "" : std::to_string(edges[b]).c_str(), cnt[b], 100 * fl[b] / flops,
                            kk[b] / cnt[b], 100 * schur[b] / std::max(fl[b], 1.0), fl[b] / mem[b]);
        // update-matrix memory: all static, freed once pulled (sequential postorder = multifrontal stack), and
        // freed once pulled under the level-by-level schedule of all fronts
        std::vector<double> ub(R);
        for (int r = 0; r < R; ++r) ub[r] = 8.0 * (double)(sn[r].m - sn[r].k) * (sn[r].m - sn[r].k);
        double stack = 0, peakSeq = 0;
        for (int r = 0; r < R; ++r) {  // postorder
            peakSeq = std::max(peakSeq, stack + ub[r]);
            for (int c : kids[r]) stack -= ub[c];
            stack += ub[r];
        }
        std::vector<int> lv(R, 0);
        int nl = 0;
        for (int r = 0; r < R; ++r) {
            for (int c : kids[r]) lv[r] = std::max(lv[r], lv[c] + 1);
            nl = std::max(nl, lv[r] + 1);
        }
        std::vector<double> delta(nl + 1, 0);  // live during levels [lv[r], lv[parent]]
        for (int r = 0; r < R; ++r) {
            const int until = sn[r].parent >= 0 ? lv[sn[r].parent] : lv[r];
            delta[lv[r]] += ub[r];
            delta[until + 1] -= ub[r];
        }
        double live = 0, peakLev = 0;
        for (int l = 0; l < nl; ++l) peakLev = std::max(peakLev, live += delta[l]);
        std::printf("  update matrices: %.2f GB static, peak %.2f GB if freed once pulled in postorder, %.2f GB "
                    "level by level (%d levels); L panels %.2f GB\n", updTotal * 8.0 / GB, peakSeq / GB, peakLev / GB, nl,
                    panelTotal * 8.0 / GB);
    }
    if (whole) {  // only fronts below the large-front path keep a static update matrix
        updTotal = 0;
        for (int r = 0; r < R; ++r)
            if (!big[r]) { sn[r].updOff = updTotal; updTotal += (int64_t)(sn[r].m - sn[r].k) * (sn[r].m - sn[r].k); }
        if (writeback) die("--whole does not support --writeback");
        if (kernel != 2) die("--whole needs kernel 2");
    }
    // --upd-plan (with --whole): update matrices by liveness instead of statically. With the level schedule a block
    // front's update matrix lives from its level to its parent's (to the large-front path when the parent is large),
    // so offsets come from a best-fit pass over the levels. A small-front subtree runs on one warp in postorder:
    // the update matrices that stay inside it live on that warp's own stack, only its root's in the level plan.
    // On by default: it needs the level schedule (a few percent of the factorization time),
    // and memory is what limits the problem size.
    int64_t arenaPerWarp = 0;  // complex values of one small-front warp's stack
    if (updPlanOpt != "on" && updPlanOpt != "off") die("--upd-plan: on or off");
    const bool updPlan = whole && updPlanOpt == "on";
    if (updPlan) {
        if (schedule == "persistent") die("--schedule persistent needs --upd-plan off (lifetimes are level intervals)");
        schedule = "level";
        const auto tPlan = Clock::now();
        auto usize = [&](int r) { return (int64_t)(sn[r].m - sn[r].k) * (sn[r].m - sn[r].k); };
        // small-front subtrees: stack offsets (encoded as -1 - offset) of their inner update matrices
        for (const int2& st : subtrees) {
            int64_t top = 0;
            for (int r = st.x; r <= st.y; ++r) {
                for (int c : kids[r]) top -= usize(c);  // the children's matrices are on top, consumed into r's front
                if (r == st.y) break;                   // the root's goes to the level plan
                sn[r].updOff = -1 - top;
                top += usize(r);
                arenaPerWarp = std::max(arenaPerWarp, top);
            }
        }
        // level plan: block fronts at their level, small-subtree roots at level -1; freed after the parent's level
        // (never, before the large-front path, when the parent is large). Matrices with the same lifetime form a
        // group, so the allocator sees at most (nlev + 1) x (nlev + 2) groups instead of millions of matrices.
        // A group goes back to back into the best-fitting free range; if none is large enough, it is split at
        // matrix boundaries over the free ranges (largest first, each filled with a run of the group's matrices
        // in decreasing size), and the rest goes to the end.
        auto lev = [&](int r) { return isS[r] ? -1 : level[r]; };
        const int nEnd = nlev + 2;
        std::vector<int64_t> groupLen((size_t)(nlev + 1) * nEnd, 0);
        std::vector<int> groupOf(R, -1);
        for (int r = 0; r < R; ++r) {
            if (big[r] || usize(r) == 0) continue;
            if (isS[r] && sn[r].parent >= 0 && isS[sn[r].parent]) continue;  // inside a small-front subtree
            const int p = sn[r].parent;
            const int g = (lev(r) + 1) * nEnd + (p < 0 || big[p] ? nlev + 1 : lev(p) + 1);
            groupOf[r] = g;
            groupLen[g] += usize(r);
        }
        const size_t G = groupLen.size();
        std::vector<int> grpBeg(G + 1, 0), grpItems;
        for (int r = 0; r < R; ++r)
            if (groupOf[r] >= 0) ++grpBeg[groupOf[r] + 1];
        for (size_t g = 0; g < G; ++g) grpBeg[g + 1] += grpBeg[g];
        grpItems.resize(grpBeg[G]);
        {
            std::vector<int> fill(grpBeg.begin(), grpBeg.end() - 1);
            for (int r = 0; r < R; ++r)
                if (groupOf[r] >= 0) grpItems[fill[groupOf[r]]++] = r;
        }
        std::set<std::pair<int64_t, int64_t>> bySize;  // free ranges as (length, offset)
        std::map<int64_t, int64_t> byOff;              // offset -> length
        int64_t peak = 0;
        auto take = [&](int64_t len) {  // best fit, else at the end (growing a free range that reaches it)
            auto it = bySize.lower_bound({len, INT64_MIN});
            if (it != bySize.end()) {
                const int64_t l = it->first, o = it->second;
                bySize.erase(it);
                byOff.erase(o);
                if (l > len) {
                    bySize.insert({l - len, o + len});
                    byOff.emplace(o + len, l - len);
                }
                return o;
            }
            if (!byOff.empty()) {
                auto last = std::prev(byOff.end());
                if (last->first + last->second == peak) {
                    const int64_t o = last->first;
                    bySize.erase({last->second, o});
                    byOff.erase(last);
                    peak = o + len;
                    return o;
                }
            }
            const int64_t o = peak;
            peak += len;
            return o;
        };
        auto give = [&](int64_t o, int64_t l) {  // free and coalesce with the neighbours
            auto next = byOff.lower_bound(o);
            if (next != byOff.end() && next->first == o + l) {
                bySize.erase({next->second, next->first});
                l += next->second;
                next = byOff.erase(next);
            }
            if (next != byOff.begin()) {
                auto prev = std::prev(next);
                if (prev->first + prev->second == o) {
                    bySize.erase({prev->second, prev->first});
                    o = prev->first;
                    l += prev->second;
                    byOff.erase(prev);
                }
            }
            byOff.emplace(o, l);
            bySize.insert({l, o});
        };
        int64_t bound = 0;  // no placement needs less: the most that is live at once (at a level's allocations)
        for (int l = 0; l <= nlev; ++l) {
            int64_t live = 0;
            for (int b = 0; b <= l; ++b)
                for (int e = l; e < nEnd; ++e) live += groupLen[(size_t)b * nEnd + e];
            bound = std::max(bound, live);
        }
        std::vector<std::vector<std::pair<int64_t, int64_t>>> groupRanges(G);  // (offset, length) pieces
        auto placeRun = [&](const int* it, const int* end, int64_t o) {  // back to back from o
            for (; it != end; ++it) {
                sn[*it].updOff = o;
                o += usize(*it);
            }
        };
        auto placeGroup = [&](int g) {
            const int64_t len = groupLen[g];
            if (bySize.lower_bound({len, INT64_MIN}) != bySize.end()) {  // one free range holds it all
                const int64_t o = take(len);
                placeRun(&grpItems[grpBeg[g]], &grpItems[grpBeg[g + 1]], o);
                groupRanges[g].push_back({o, len});
                return;
            }
            std::vector<int> rest(grpItems.begin() + grpBeg[g], grpItems.begin() + grpBeg[g + 1]);
            std::sort(rest.begin(), rest.end(), [&](int x, int y) { return usize(x) > usize(y); });
            const std::vector<std::pair<int64_t, int64_t>> holes(bySize.rbegin(), bySize.rend());  // largest first
            for (const auto& h : holes) {
                const int64_t hl = h.first, ho = h.second;
                if (rest.empty() || usize(rest.back()) > hl) continue;
                // the first matrix that fits, then the run after it while the range has room
                const auto first = std::partition_point(rest.begin(), rest.end(), [&](int r) { return usize(r) > hl; });
                auto last = first;
                int64_t used = 0;
                for (; last != rest.end() && used + usize(*last) <= hl; ++last) used += usize(*last);
                placeRun(&*first, &*first + (last - first), ho);
                rest.erase(first, last);
                bySize.erase({hl, ho});
                byOff.erase(ho);
                if (hl > used) {
                    bySize.insert({hl - used, ho + used});
                    byOff.emplace(ho + used, hl - used);
                }
                groupRanges[g].push_back({ho, used});
            }
            if (!rest.empty()) {
                int64_t sum = 0;
                for (int r : rest) sum += usize(r);
                const int64_t o = take(sum);
                placeRun(rest.data(), rest.data() + rest.size(), o);
                groupRanges[g].push_back({o, sum});
            }
        };
        for (int l = 0; l <= nlev; ++l) {  // l = level + 1: the groups that start here, largest first, then frees
            std::vector<int> in;
            for (int e = 0; e < nEnd; ++e)
                if (groupLen[(size_t)l * nEnd + e] > 0) in.push_back(l * nEnd + e);
            std::sort(in.begin(), in.end(), [&](int x, int y) { return groupLen[x] > groupLen[y]; });
            for (int g : in) placeGroup(g);
            for (int b = 0; b <= nlev; ++b)
                for (const auto& [o, len] : groupRanges[(size_t)b * nEnd + l]) give(o, len);
        }
        std::printf("update matrices by liveness (level schedule): %.2f GB in the level plan (static: %.2f GB), small-front "
                    "stacks %.3f MB per warp; at least %.2f GB live at once; planned in %.2f s\n", peak * 8.0 / GB, updTotal * 8.0 / GB,
                    arenaPerWarp * 8.0 / 1048576, bound * 8.0 / GB, since(tPlan));
        updTotal = peak;  // the small-front stacks follow, once the number of warps is known
    }

    // ---------------- device data
    int dev = 0, nsm = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    if (kernel != 1 && kernel != 2) die("--kernel must be 1 or 2");
    if (kernel == 2 && threads != 256) die("kernel 2 needs 256 threads");
    auto levelK = kernel == 1 ? levelKer<1> : levelKer<2>;
    auto persistentK = kernel == 1 ? persistentKer<1> : persistentKer<2>;
    int64_t maxPanel = 0;
    for (int r = 0; r < R; ++r) maxPanel = std::max<int64_t>(maxPanel, (int64_t)sn[r].m * sn[r].k);
    const size_t shm = kernel == 1 ? (size_t)maxM * sizeof(cuComplex) : 0;  // v2 uses static shared memory
    if (shm > 48 * 1024) {
        CK(cudaFuncSetAttribute(levelK, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(persistentK, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    }
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, persistentK, threads, shm));
    // v1: each resident block owns a maxM^2 front scratch; v2 needs a panel buffer only for the write-back
    if (blocksPerSm <= 0) blocksPerSm = kernel == 1 ? std::min(occ, 2) : occ;
    blocksPerSm = std::min(blocksPerSm, occ);
    const int nblocks = nsm * blocksPerSm;  // all resident: required by the persistent kernel's waiting
    const int64_t frontStride = kernel == 1 ? (int64_t)maxM * maxM : writeback ? maxPanel : 0;
    std::printf("GPU: kernel v%d, %d SMs x %d blocks of %d threads (occupancy allows %d); front scratch %.2f GB; "
                "L panels %s\n", kernel, nsm, blocksPerSm, threads, occ, nblocks * frontStride * 8.0 / GB,
                writeback ? "written back to pinned host memory by DMA through a device ring" : "in device memory");

    auto up = [](const void* h, size_t bytes) {
        void* d = nullptr;
        CK(cudaMalloc(&d, std::max<size_t>(bytes, 1)));
        if (bytes) CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
        return d;
    };
    Args a{};
    a.sn = static_cast<const SNode*>(up(sn.data(), R * sizeof(SNode)));
    a.children = static_cast<const int*>(up(children.data(), children.size() * sizeof(int)));
    a.relind = dRelind;
    a.aPos = dAPos;
    cuComplex* dAVal = nullptr;  // filled per frequency
    CK(cudaMalloc(&dAVal, std::max<int64_t>(aLen, 1) * sizeof(cuComplex)));
    a.aVal = dAVal;
    // write-back: device ring, chunk counters, allocation head, freed mark; pinned host buffer for the panels
    cuComplex* hostAll = nullptr;
    unsigned* hCnt = nullptr;
    unsigned long long* hFreed = nullptr;
    unsigned* hZeros = nullptr;
    cudaStream_t dataStream = nullptr, pollStream = nullptr;
    if (writeback) {
        a.dma = 1;
        a.chunkElems = (unsigned long long)(chunkMb * 1024 * 1024 / sizeof(cuComplex));
        a.nChunks = (int)std::max<double>(2, std::ceil(ringMb / chunkMb));
        a.ringElems = a.chunkElems * a.nChunks;
        if ((int64_t)a.ringElems < maxPanel + 2 * (int64_t)a.chunkElems) die("ring too small for the largest panel");
        CK(cudaMalloc(&a.ring, a.ringElems * sizeof(cuComplex)));
        CK(cudaMalloc(&a.chunkCnt, a.nChunks * sizeof(unsigned)));
        CK(cudaMalloc(&a.allocHead, sizeof(unsigned long long)));
        unsigned long long* dFreed = nullptr;
        CK(cudaMalloc(&dFreed, sizeof(unsigned long long)));
        a.freed = dFreed;
        CK(cudaMalloc(&a.hostOff, R * sizeof(int64_t)));
        CK(cudaHostAlloc(&hostAll, panelTotal * sizeof(cuComplex), cudaHostAllocDefault));
        CK(cudaHostAlloc(&hCnt, a.nChunks * sizeof(unsigned), cudaHostAllocDefault));
        CK(cudaHostAlloc(&hZeros, a.nChunks * sizeof(unsigned), cudaHostAllocDefault));
        std::memset(hZeros, 0, a.nChunks * sizeof(unsigned));
        CK(cudaHostAlloc(&hFreed, 2 * sizeof(unsigned long long), cudaHostAllocDefault));
        CK(cudaStreamCreateWithFlags(&dataStream, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&pollStream, cudaStreamNonBlocking));
        std::printf("write-back: ring %.2f MB in %d chunks of %.3f MB, host buffer %.2f GB (pinned)\n",
                    a.ringElems * 8.0 / 1048576, a.nChunks, a.chunkElems * 8.0 / 1048576, panelTotal * 8.0 / GB);
    } else {
        CK(cudaMalloc(&a.panels, panelTotal * sizeof(cuComplex)));
    }
    // Host side of the write-back while the kernels of one run execute: copy runs of complete chunks to
    // hostAll, clear their counters, publish the freed mark; at the end copy the rest. Returns the copied volume.
    double tailSec = 0;  // write-back time left after the kernels finished (last drain)
    auto drain = [&](cudaEvent_t kernelsDone) {
        unsigned long long copied = 0, head = 0;
        Clock::time_point tDone;
        int slot = 0;
        auto copyRange = [&](unsigned long long lo, unsigned long long hi) {  // [lo, hi), split at the ring end
            while (lo < hi) {
                const unsigned long long start = lo % a.ringElems, n = std::min(hi - lo, a.ringElems - start);
                CK(cudaMemcpyAsync(hostAll + lo, a.ring + start, n * sizeof(cuComplex), cudaMemcpyDeviceToHost,
                                   dataStream));
                lo += n;
            }
        };
        auto lastProgress = Clock::now();
        for (;;) {
            const bool done = cudaEventQuery(kernelsDone) == cudaSuccess;
            unsigned long long target = copied;
            if (!done && since(lastProgress) > 3.0) {  // the kernels wait for ring space that never comes
                unsigned long long st[2];
                CK(cudaMemcpyAsync(st, a.allocHead, sizeof(st[0]), cudaMemcpyDeviceToHost, pollStream));
                CK(cudaMemcpyAsync(st + 1, a.freed, sizeof(st[1]), cudaMemcpyDeviceToHost, pollStream));
                CK(cudaStreamSynchronize(pollStream));
                std::fprintf(stderr, "write-back stalled: copied %llu, allocated %llu, freed mark %llu, ring %llu, chunk %llu;"
                             " counters from the next chunk:", copied, st[0], st[1], a.ringElems, a.chunkElems);
                for (int c = 0; c < a.nChunks; ++c) std::fprintf(stderr, " %u", hCnt[(copied / a.chunkElems + c) % a.nChunks]);
                std::fprintf(stderr, "\n");
                std::exit(1);
            }
            if (done) {
                tDone = Clock::now();
                CK(cudaMemcpy(&head, a.allocHead, sizeof(head), cudaMemcpyDeviceToHost));
                target = head;
            } else {
                CK(cudaMemcpyAsync(hCnt, a.chunkCnt, a.nChunks * sizeof(unsigned), cudaMemcpyDeviceToHost, pollStream));
                CK(cudaStreamSynchronize(pollStream));
                for (int c = 0; c < a.nChunks; ++c) {  // copied stays chunk-aligned while kernels run
                    if (hCnt[(target / a.chunkElems) % a.nChunks] != a.chunkElems) break;
                    target += a.chunkElems;
                }
            }
            if (target > copied) {
                copyRange(copied, target);
                if (!done) {
                    // clear their counters by copies from pinned zeros (a memset could need an SM that the
                    // waiting kernels never give up)
                    for (unsigned long long c = copied / a.chunkElems, ce = target / a.chunkElems; c < ce;) {
                        const unsigned long long n = std::min(ce - c, (unsigned long long)a.nChunks - c % a.nChunks);
                        CK(cudaMemcpyAsync(a.chunkCnt + c % a.nChunks, hZeros, n * sizeof(unsigned),
                                           cudaMemcpyHostToDevice, dataStream));
                        c += n;
                    }
                    hFreed[slot] = target;  // the stream reads it after the copies above; 2 slots alternate
                    CK(cudaMemcpyAsync(const_cast<unsigned long long*>(a.freed), hFreed + slot, sizeof(unsigned long long),
                                       cudaMemcpyHostToDevice, dataStream));
                    slot ^= 1;
                    CK(cudaStreamSynchronize(dataStream));  // keeps hFreed slots and counters consistent
                }
                copied = target;
                lastProgress = Clock::now();
            }
            if (done) break;
        }
        CK(cudaStreamSynchronize(dataStream));
        tailSec = since(tDone);
        return copied;
    };
    int occS = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occS, smallKer, 32 * kSmallWarps, 0));
    const int smallBlocks = std::max(1, std::min(nsm * occS, (int)(subtrees.size() + kSmallWarps - 1) / kSmallWarps));
    a.arenaBase = updTotal;  // the small-front warps' stacks (--upd-plan) follow the level plan
    a.arenaPerWarp = arenaPerWarp;
    updTotal += (int64_t)smallBlocks * kSmallWarps * arenaPerWarp;
    // The update matrices are needed only during the factorization and FGMRES's Krylov storage only during the
    // solve, so with --gmres the storage starts in this buffer (and grows beyond it as the iterations need).
    const size_t updBytes = std::max<int64_t>(updTotal, 1) * sizeof(cuComplex);
    std::printf("update matrices: %.2f GB%s\n", updBytes / GB, mixed ? ", lent to FGMRES's Krylov storage during the solves" : "");
    CK(cudaMalloc(&a.upd, updBytes));
    if (mixed) mixed->setWorkspace(a.upd, updBytes);
    CK(cudaMalloc(&a.scratch, std::max<int64_t>(nblocks * frontStride, 1) * sizeof(cuComplex)));
    a.frontStride = frontStride;
    CK(cudaMalloc(&a.zeroPivots, sizeof(unsigned)));
    int* dLevelList = static_cast<int*>(up(levelList.data(), levelList.size() * sizeof(int)));
    // the level schedule's Schur update tiles (fronts with m - k > kSplitMu), per level, the costliest (largest
    // k) first; not with the write-back, whose panels are only in a per-block buffer after the level kernel
    const bool splitSchur = !writeback;
    std::vector<int> tilePtr(nlev + 1, 0);
    std::vector<int4> tiles;
    if (splitSchur)
        for (int l = 0; l < nlev; ++l) {
            const size_t t0 = tiles.size();
            for (int q = levelPtr[l]; q < levelPtr[l + 1]; ++q) {
                const int r = levelList[q], mu = sn[r].m - sn[r].k;
                if (mu <= kSplitMu && !(mu > 0 && sn[r].k > kStepW)) continue;  // as factorFront2's split
                const int T = (mu + 63) / 64;
                for (int ti = 0; ti < T; ++ti)
                    for (int tj = 0; tj <= ti; ++tj) tiles.push_back(make_int4(r, ti, tj, sn[r].k));
            }
            std::stable_sort(tiles.begin() + t0, tiles.end(), [](int4 x, int4 y) { return x.w > y.w; });
            tilePtr[l + 1] = (int)tiles.size();
        }
    const int4* dTiles = static_cast<const int4*>(up(tiles.data(), tiles.size() * sizeof(int4)));
    {  // each child's first row in every tile of its parent's update matrix, for the fronts with tiles
        std::vector<int> fronts, tbOff(std::max<size_t>(children.size(), 1), 0);
        int64_t tbLen = 0;
        for (size_t x = 0; x < tiles.size(); ++x)
            if (x == 0 || tiles[x].x != tiles[x - 1].x) fronts.push_back(tiles[x].x);
        std::sort(fronts.begin(), fronts.end());
        fronts.erase(std::unique(fronts.begin(), fronts.end()), fronts.end());
        for (int r : fronts)
            for (int ci = sn[r].childBeg; ci < sn[r].childEnd; ++ci) {
                tbOff[ci] = (int)tbLen;
                tbLen += (sn[r].m - sn[r].k + 63) / 64 + 1;
            }
        if (tbLen > INT32_MAX) die("tile bounds exceed int32");
        int* dTb = nullptr;
        CK(cudaMalloc(&dTb, std::max<int64_t>(tbLen, 1) * sizeof(int)));
        a.tileBound = dTb;
        a.tileBoundOff = static_cast<const int*>(up(tbOff.data(), tbOff.size() * sizeof(int)));
        const int* dFronts = static_cast<const int*>(up(fronts.data(), fronts.size() * sizeof(int)));
        if (!fronts.empty()) tileBoundsKer<<<std::min<int>(((int)fronts.size() + 7) / 8, 4096), 256>>>(a, dFronts, (int)fronts.size(), dTb);
        CK(cudaDeviceSynchronize());
    }
    // the stepped panels' steps s >= 1 per level: the fronts with k > s kStepW, and the tiles of their columns
    // [s kStepW, k) (rows >= columns)
    std::vector<int> levelStep(nlev + 1, 0), stepPtr(1, 0), stepFronts, stepTilePtr(1, 0);
    std::vector<int4> stepTiles;
    for (int l = 0; l < nlev; ++l) {
        levelStep[l] = (int)stepPtr.size() - 1;
        int kmax = 0;
        if (splitSchur)
            for (int q = levelPtr[l]; q < levelPtr[l + 1]; ++q) kmax = std::max(kmax, sn[levelList[q]].k);
        for (int st = 1; st * kStepW < kmax; ++st) {
            const int b = st * kStepW;
            for (int q = levelPtr[l]; q < levelPtr[l + 1]; ++q) {
                const int r = levelList[q], m = sn[r].m, k = sn[r].k;
                if (k <= b) continue;
                stepFronts.push_back(r);
                const int RT = (m - b + 63) / 64, CT = (k - b + 63) / 64;
                for (int tj = 0; tj < CT; ++tj)
                    for (int ti = tj; ti < RT; ++ti) stepTiles.push_back(make_int4(r, ti, tj, st));
            }
            stepPtr.push_back((int)stepFronts.size());
            stepTilePtr.push_back((int)stepTiles.size());
        }
    }
    levelStep[nlev] = (int)stepPtr.size() - 1;
    const int* dStepFronts = static_cast<const int*>(up(stepFronts.data(), stepFronts.size() * sizeof(int)));
    const int4* dStepTiles =
        static_cast<const int4*>(up(stepTiles.data(), stepTiles.size() * sizeof(int4)));
    int occPT = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occPT, panelTileKer, 256, 0));
    const int panelTileBlocks = std::max(1, nsm * occPT);
    if (!stepFronts.empty())
        std::printf("panels with k > %d: right-looking in blocks of %d, %zu block steps after the first, %zu tiles of "
                    "64 x 64 for their trailing columns\n", kStepW, kStepW, stepFronts.size(), stepTiles.size());
    int occW48 = 0, occW64 = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occW48, warpFrontKer<48>, 32, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occW64, warpFrontKer<64>, 32, 0));
    const int warp48Blocks = std::max(1, nsm * occW48), warp64Blocks = std::max(1, nsm * occW64);
    {
        int n48 = 0, n64 = 0;
        for (int l = 0; l < nlev; ++l) n48 += levelW48[l], n64 += levelW64[l];
        if (n48 + n64)
            std::printf("block fronts with m <= 48 / <= %d: %d / %d, one warp each in shared memory (%d / %d warps per SM)\n",
                        kWarpFrontM, n48, n64, occW48, occW64);
    }
    int occT = 0;
    CK(cudaFuncSetAttribute(schurTileKer, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kTileSmem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occT, schurTileKer, 256, kTileSmem));
    const int tileBlocks = std::max(1, nsm * occT);
    if (splitSchur)
        std::printf("update matrices of the block fronts with m - k > %d: %zu tiles of 64 x 64 over the GPU after each level, "
                    "assembled from the children and updated in shared memory\n", kSplitMu, tiles.size());
    std::vector<int> cnt0(R, 0), queue0(R, -1);
    int nready = 0;
    for (int r = 0; r < R; ++r) {  // block-kernel fronts; children done by smallKer do not count
        if (big[r]) cnt0[r] = INT_MAX / 2;  // never enqueued: the large-front path does it
        if (isS[r] || big[r]) continue;
        for (int c : kids[r]) cnt0[r] += !isS[c];
        if (cnt0[r] == 0) queue0[nready++] = r;  // in postorder
    }
    const int2* dSub = static_cast<const int2*>(up(subtrees.data(), subtrees.size() * sizeof(int2)));
    int* dNext = nullptr;
    CK(cudaMalloc(&dNext, sizeof(int)));
    // Load every kernel now: with lazy module loading, the first launch of a kernel while a waiting kernel
    // runs (smallKer or a block kernel spinning on ring space) can need a context synchronization that never
    // completes, and the write-back copies queued behind it deadlock.
    for (const void* f : {(const void*)smallKer, (const void*)levelK, (const void*)persistentK, (const void*)schurTileKer,
                          (const void*)panelTileKer, (const void*)panelStepKer,
                          (const void*)warpFrontKer<48>, (const void*)warpFrontKer<64>}) {
        cudaFuncAttributes attr;
        CK(cudaFuncGetAttributes(&attr, f));
    }
    std::printf("small fronts: %zu subtrees with %d fronts (m <= %d), one warp each, %d blocks of %d warps; "
                "block kernels: %d fronts\n", subtrees.size(), smallFronts, smallM, smallBlocks, kSmallWarps, RL);
    int *dQueue, *dCnt, *dHead, *dTail;
    CK(cudaMalloc(&dQueue, R * sizeof(int)));
    CK(cudaMalloc(&dCnt, R * sizeof(int)));
    CK(cudaMalloc(&dHead, sizeof(int)));
    CK(cudaMalloc(&dTail, sizeof(int)));
    if (graphsOpt != "on" && graphsOpt != "off") die("--graphs: on or off");
    cudaEvent_t e0, e1, eA, eFork;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&eA));
    CK(cudaEventCreateWithFlags(&eFork, cudaEventDisableTiming));
    cudaStream_t fst;  // the factorization's stream: unlike the legacy stream it can be captured into a graph
    CK(cudaStreamCreateWithFlags(&fst, cudaStreamNonBlocking));
    // large-front path: a pool of streams with a cuBLAS handle and a workspace each; front buffers come from
    // the stream-ordered allocator and are freed once the parent has pulled the update matrix
    std::vector<cudaStream_t> bst(nStreams);
    std::vector<cublasHandle_t> bh(nStreams);
    std::vector<FrontLdltStreams> bq(nStreams);  // each pool stream with its lookahead stream (the panels)
    std::vector<cudaEvent_t> bdone(R), bend(nStreams);
    std::vector<cuComplex*> bF(R, nullptr);
    std::vector<int> bstream(R, -1);
    double bigFlops = 0;
    if (!bigOrder.empty()) {
        int maxBig = 0;
        for (int r : bigOrder) {
            maxBig = std::max(maxBig, sn[r].m);
            for (int j = 0; j < sn[r].k; ++j) bigFlops += 4.0 * (double)(sn[r].m - j) * (sn[r].m - j);
        }
        cudaMemPool_t pool;
        CK(cudaDeviceGetDefaultMemPool(&pool, dev));
        uint64_t keep = UINT64_MAX;
        CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep));
        int prLeast = 0, prGreatest = 0;
        CK(cudaDeviceGetStreamPriorityRange(&prLeast, &prGreatest));
        CK(cudaFuncSetAttribute(diagShKer, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)((kShElems + kDiagShMax * kDiagShMax) * sizeof(cuComplex))));
        for (int i = 0; i < nStreams; ++i) {
            CK(cudaStreamCreateWithFlags(&bst[i], cudaStreamNonBlocking));
            frontCublas(cublasCreate(&bh[i]), "cublasCreate");
            frontCublas(cublasSetStream(bh[i], bst[i]), "cublasSetStream");
            CK(cudaEventCreateWithFlags(&bend[i], cudaEventDisableTiming));
            FrontLdltStreams& q = bq[i];
            q.h0 = bh[i];
            q.s0 = bst[i];
            CK(cudaStreamCreateWithPriority(&q.s1, cudaStreamNonBlocking, prGreatest));
            frontCublas(cublasCreate(&q.h1), "cublasCreate");
            frontCublas(cublasSetStream(q.h1, q.s1), "cublasSetStream");
            for (cudaEvent_t& e : q.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            for (cuComplex*& w : q.W) CK(cudaMalloc(&w, (size_t)maxBig * 256 * sizeof(cuComplex)));
        }
        for (int r : bigOrder) CK(cudaEventCreateWithFlags(&bdone[r], cudaEventDisableTiming));
        std::printf("large-front path: %zu fronts (m > %d or above one), %.3g real flops (%.1f%%), %d streams, largest %d\n",
                    bigOrder.size(), largeM, bigFlops, 100 * bigFlops / flops, nStreams, maxBig);
    }
    // queue the large-front path after the kernels on stream s, and make s wait for it; eA (the end of the
    // kernels below) is an event node when this is captured
    auto issueBig = [&](cudaStream_t s) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CK(cudaStreamIsCapturing(s, &cap));
        CK(cudaEventRecordWithFlags(eA, s, cap == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
        CK(cudaEventRecord(eFork, s));
        for (int i = 0; i < nStreams; ++i) CK(cudaStreamWaitEvent(bst[i], eFork, 0));
        int rr = 0;
        for (int r : bigOrder) {
            const SNode& x = sn[r];
            int si = -1;  // the stream of the last large child keeps a chain on one stream
            for (int c : kids[r])
                if (big[c]) si = bstream[c];
            if (si < 0) si = rr++ % nStreams;
            bstream[r] = si;
            cudaStream_t st = bst[si];
            for (int c : kids[r])
                if (big[c] && bstream[c] != si) CK(cudaStreamWaitEvent(st, bdone[c], 0));
            cuComplex* F = nullptr;
            CK(cudaMallocAsync(reinterpret_cast<void**>(&F), (size_t)x.m * x.m * sizeof(cuComplex), st));
            bF[r] = F;
            frontZeroKer<<<std::min(x.m, 4096), 256, 0, st>>>(F, x.m);
            if (x.aEnd > x.aBeg)
                frontAKer<<<(int)std::min<int64_t>((x.aEnd - x.aBeg + 255) / 256, 4096), 256, 0, st>>>(F, a.aPos + x.aBeg,
                                                                                                   a.aVal + x.aBeg, x.aEnd - x.aBeg);
            for (int c : kids[r]) {  // one child at a time: no write conflicts
                const SNode& y = sn[c];
                const int mc = y.m - y.k;
                if (mc == 0) continue;
                const cuComplex* U = big[c] ? bF[c] + y.k + (int64_t)y.m * y.k : a.upd + y.updOff;
                extendAddKer<<<std::min(mc, 4096), 256, 0, st>>>(F, x.m, U, big[c] ? y.m : mc, mc, a.relind + y.relOff);
                if (big[c]) CK(cudaFreeAsync(bF[c], st));
            }
            const int nbr = nbOpt > 0 ? nbOpt : x.m > 6000 ? 256 : x.m > 2000 ? 128 : 64;
            largeFrontLdlt(bq[si], F, x.m, x.m, x.k, nbr, 512, a.zeroPivots);
            panelCopyKer<<<(int)std::min<int64_t>(((int64_t)x.m * x.k + 255) / 256, 8192), 256, 0, st>>>(a.panels + x.panelOff,
                                                                                                       F, x.m, x.k);
            if (x.parent < 0) CK(cudaFreeAsync(F, st));
            CK(cudaEventRecord(bdone[r], st));
        }
        for (int i = 0; i < nStreams; ++i) {
            CK(cudaEventRecord(bend[i], bst[i]));
            CK(cudaStreamWaitEvent(s, bend[i], 0));
        }
    };
    // the whole factorization with the level schedule on stream s: small subtrees, the levels, the large fronts
    auto issueLevels = [&](cudaStream_t s) {
        CK(cudaMemsetAsync(a.zeroPivots, 0, sizeof(unsigned), s));
        CK(cudaMemsetAsync(dNext, 0, sizeof(int), s));
        if (!subtrees.empty()) smallKer<<<smallBlocks, 32 * kSmallWarps, 0, s>>>(a, dSub, (int)subtrees.size(), dNext);
        Args aL = a;  // the level schedule leaves the larger Schur updates to the tile kernel
        aL.splitSchur = splitSchur;
        aL.stepPanels = splitSchur;
        for (int l = 0; l < nlev; ++l) {
            const int n48 = levelW48[l], n64 = levelW64[l], nw = n48 + n64;
            const int count = levelPtr[l + 1] - levelPtr[l] - nw;
            if (n48 > 0) warpFrontKer<48><<<std::min(n48, warp48Blocks), 32, 0, s>>>(aL, dLevelList + levelPtr[l], n48);
            if (n64 > 0)
                warpFrontKer<64><<<std::min(n64, warp64Blocks), 32, 0, s>>>(aL, dLevelList + levelPtr[l] + n48, n64);
            if (count > 0) levelK<<<std::min(count, nblocks), threads, shm, s>>>(aL, dLevelList + levelPtr[l] + nw, count);
            for (int x = levelStep[l]; x < levelStep[l + 1]; ++x) {
                const int npt = stepTilePtr[x + 1] - stepTilePtr[x], nf = stepPtr[x + 1] - stepPtr[x];
                panelTileKer<<<std::min(npt, panelTileBlocks), 256, 0, s>>>(aL, dStepTiles + stepTilePtr[x], npt);
                panelStepKer<<<std::min(nf, nblocks), 256, 0, s>>>(aL, dStepFronts + stepPtr[x], nf, x - levelStep[l] + 1);
            }
            const int nt = tilePtr[l + 1] - tilePtr[l];
            if (nt > 0) schurTileKer<<<std::min(nt, tileBlocks), 256, kTileSmem, s>>>(aL, dTiles + tilePtr[l], nt);
        }
        if (!bigOrder.empty()) issueBig(s);
    };
    // With --graphs the level schedule is captured once (it depends only on the pattern; every frequency's
    // values go into the same buffers) and replayed for every factorization. The large fronts' buffers become
    // the graph's allocation nodes.
    cudaGraphExec_t factorGraph = nullptr;
    if (graphsOpt == "on" && schedule != "persistent" && !writeback) {
        const auto tg = Clock::now();
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(fst, cudaStreamCaptureModeThreadLocal));
        issueLevels(fst);
        CK(cudaStreamEndCapture(fst, &g));
        size_t nodes = 0;
        CK(cudaGraphGetNodes(g, nullptr, &nodes));
        CK(cudaGraphInstantiate(&factorGraph, g, 0));
        CK(cudaGraphDestroy(g));
        CK(cudaGraphUpload(factorGraph, fst));
        CK(cudaStreamSynchronize(fst));
        std::printf("factorization graph: %zu nodes, captured and instantiated in %.3f s\n", nodes, since(tg));
    }

    // ---------------- the triangular solve (--solve): the preconditioner of the FGMRES path
    FrontSolve fsolve;      // --solve's test, and the preconditioner with --solve-precision single
    FrontSolve64 fsolve64;  // the preconditioner with --solve-precision double
    FrontSolveDF fsolveDF;  // the preconditioner with --solve-precision double-float
    if (solvePrecision != "single" && solvePrecision != "double" && solvePrecision != "double-float")
        die("--solve-precision: single, double or double-float");
    const bool precond64 = mixed && solvePrecision == "double", precondDF = mixed && solvePrecision == "double-float";
    const int kSolveNr = solveNrOpt;  // right-hand sides of the test (--solve-nr; the test systems have 2 ports)
    cuComplex *dSb = nullptr, *dSx = nullptr, *dPb = nullptr, *dPx = nullptr;  // test / preconditioner vectors
    std::vector<Complex> sb;  // the test right-hand sides
    if (solveReps > 0 || mixed) {
        if (!whole || writeback) die("--solve needs --whole and the panels in device memory");
        std::vector<SolveNode> nodes(R);
        std::vector<int> parentOf(R);
        for (int r = 0; r < R; ++r) {
            nodes[r] = SolveNode{sn[r].m, sn[r].k, (int)super[regSuper[r]], sn[r].childBeg, sn[r].childEnd, sn[r].panelOff,
                                 sn[r].parent >= 0 ? sn[r].relOff : -1};
            parentOf[r] = sn[r].parent;
        }
        std::vector<int32_t> perm32(Perm, Perm + n);
        int* dPerm = static_cast<int*>(up(perm32.data(), n * sizeof(int32_t)));
        if (solveReps > 0 || (mixed && !precond64 && !precondDF))
            fsolve.setup(n, std::max(kSolveNr, nports), nodes, children, parentOf, big, a.children, a.relind, dUrow, a.panels,
                         relLen, dPerm, bst);
        if (precond64 || (solveReps > 0 && solvePrecision == "double"))
            fsolve64.setup(n, std::max(nports, solveReps > 0 ? kSolveNr : 0), nodes, children, parentOf, big, a.children, a.relind, dUrow, a.panels, relLen, dPerm, bst);
        if (precondDF || (solveReps > 0 && solvePrecision == "double-float"))
            fsolveDF.setup(n, std::max(nports, solveReps > 0 ? kSolveNr : 0), nodes, children, parentOf, big, a.children, a.relind, dUrow, a.panels, relLen, dPerm, bst);
        fsolve.useGraphs(graphsOpt == "on");
        fsolve64.useGraphs(graphsOpt == "on");
        fsolveDF.useGraphs(graphsOpt == "on");
        if (mixed && !precond64 && !precondDF) {
            CK(cudaMalloc(&dPb, n * nports * sizeof(cuComplex)));
            CK(cudaMalloc(&dPx, n * nports * sizeof(cuComplex)));
        }
    }
    if (solveReps > 0) {
        CK(cudaMalloc(&dSb, n * kSolveNr * sizeof(cuComplex)));
        CK(cudaMalloc(&dSx, n * kSolveNr * sizeof(cuComplex)));
        std::mt19937_64 gen(11);
        std::normal_distribution<double> nd;
        sb.resize(n * kSolveNr);
        std::vector<cuComplex> sb32(n * kSolveNr);
        for (int64_t i = 0; i < n * kSolveNr; ++i) {
            sb[i] = Complex(nd(gen), nd(gen));
            sb32[i] = make_cuComplex((float)sb[i].real(), (float)sb[i].imag());
        }
        CK(cudaMemcpy(dSb, sb32.data(), n * kSolveNr * sizeof(cuComplex), cudaMemcpyHostToDevice));
    }
    report.analysis = since(tAnalysis);
    // host copy of the panels: for the check, the CPU reference and the comparison of the two schedules
    const bool needHost = doCheck || cpuRef || schedule == "both";
    // each frequency's values in CSR order go to the device in chunks, and placeValuesKer puts them into
    // the fronts' assembly slots
    const int64_t chunk = std::min<int64_t>(std::max<int64_t>(A.nnz, 1), int64_t(1) << 25);
    cuDoubleComplex* dStage = nullptr;
    CK(cudaMalloc(&dStage, chunk * sizeof(cuDoubleComplex)));
    Complex* hVal = nullptr;  // pinned: later frequencies' values in production runs
    auto memUsed = []() {  // device memory in use (this process, if the GPU is not shared), GB
        size_t freeB = 0, totalB = 0;
        CK(cudaMemGetInfo(&freeB, &totalB));
        return (totalB - freeB) / 1073741824.0;
    };
    std::printf("device memory in use before the first factorization: %.2f GB\n", memUsed());
    for (size_t fi = 0; fi < dirs.size(); ++fi) {
        const std::string& dir = dirs[fi];
        const Complex* vals = A.val.data();
        if (fi > 0) {
            auto tl = Clock::now();
            if (doCheck) {  // verification run: every check, and the values on the host for the check
                readSamePattern(dir + "/A.bin", A, A.val);
                vals = A.val.data();
                if (mixed) bHost = loadDenseB(dir, n, nports, &portRows);
                report.read += since(tl);
            } else {  // production: values only (a dataset shares one pattern)
                if (!hVal) {
                    CK(cudaMallocHost(&hVal, A.nnz * sizeof(Complex)));
                    tl = Clock::now();
                }
                readMatrixAValues(dir + "/A.bin", A.nnz, hVal);
                if (mixed) bHost = loadDenseB(dir, n, nports, &portRows);
                report.read += since(tl);
                vals = hVal;
            }
        }
        auto tl = Clock::now();
        if (mixed) {  // all values in FP64 for FGMRES and the refinement, placed into the fronts from there
            CK(cudaMemcpy(dVal, vals, A.nnz * sizeof(Complex), cudaMemcpyHostToDevice));
            placeValuesKer<<<gridOf(A.nnz), 256>>>(reinterpret_cast<const cuDoubleComplex*>(dVal), dSlot, A.nnz, dAVal);
            if (fi > 0) CK(cudaMemcpy(dB, bHost.data(), n * nports * sizeof(Complex), cudaMemcpyHostToDevice));
        } else
            for (int64_t k0 = 0; k0 < A.nnz; k0 += chunk) {
                const int64_t len = std::min(chunk, A.nnz - k0);
                CK(cudaMemcpy(dStage, vals + k0, len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
                placeValuesKer<<<gridOf(len), 256>>>(dStage, dSlot + k0, len, dAVal);
            }
        CK(cudaDeviceSynchronize());
        report.upload += since(tl);
        if (cpuRef) CK(cudaMemcpy(aVal.data(), dAVal, aLen * sizeof(cuComplex), cudaMemcpyDeviceToHost));
        const double freq = dirFrequency(dir);
        std::printf("\n[%s]  %.4g Hz\n", dir.c_str(), freq);
        RunReport::Freq& rf = report.add(freq);
        std::vector<cuComplex> ref;  // factor of the level schedule, for the bitwise comparison
        if (schedule != "both" && schedule != "level" && schedule != "persistent") die("--schedule: both, level or persistent");
        for (int variant = 0; variant < 2; ++variant) {
            if ((variant == 0 && schedule == "persistent") || (variant == 1 && schedule == "level")) continue;
            double best = 1e30, bestKer = 0, bestTail = 0, bestA = 0, bestIssue = 0;
            unsigned zp = 0;
            for (int rep = 0; rep < reps; ++rep) {
                CK(cudaMemset(a.zeroPivots, 0, sizeof(unsigned)));
                CK(cudaMemcpy(dQueue, queue0.data(), R * sizeof(int), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dCnt, cnt0.data(), R * sizeof(int), cudaMemcpyHostToDevice));
                CK(cudaMemset(dHead, 0, sizeof(int)));
                CK(cudaMemcpy(dTail, &nready, sizeof(int), cudaMemcpyHostToDevice));
                CK(cudaMemset(dNext, 0, sizeof(int)));
                if (writeback) {
                    CK(cudaMemset(a.chunkCnt, 0, a.nChunks * sizeof(unsigned)));
                    CK(cudaMemset(a.allocHead, 0, sizeof(unsigned long long)));
                    CK(cudaMemset(const_cast<unsigned long long*>(a.freed), 0, sizeof(unsigned long long)));
                }
                CK(cudaDeviceSynchronize());
                const auto th = Clock::now();
                CK(cudaEventRecord(e0, fst));
                const auto ti = Clock::now();
                if (variant == 0 && factorGraph) CK(cudaGraphLaunch(factorGraph, fst));
                else if (variant == 0) issueLevels(fst);
                else {
                    if (!subtrees.empty()) smallKer<<<smallBlocks, 32 * kSmallWarps, 0, fst>>>(a, dSub, (int)subtrees.size(), dNext);
                    if (RL > 0) persistentK<<<nblocks, threads, shm, fst>>>(a, RL, dQueue, dHead, dTail, dCnt);
                    if (!bigOrder.empty()) issueBig(fst);
                }
                const double issueSec = since(ti);
                CK(cudaEventRecord(e1, fst));
                if (writeback) {
                    const unsigned long long copied = drain(e1);
                    if ((int64_t)copied != panelTotal) die("write-back copied the wrong volume");
                }
                CK(cudaEventSynchronize(e1));
                CK(cudaGetLastError());
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double t = writeback ? since(th) : ms / 1e3;  // with write-back: until the last panel is on the host
                float msA = 0;
                if (!bigOrder.empty()) CK(cudaEventElapsedTime(&msA, e0, eA));
                if (t < best) best = t, bestKer = ms / 1e3, bestTail = tailSec, bestA = msA / 1e3, bestIssue = issueSec;
                CK(cudaMemcpy(&zp, a.zeroPivots, sizeof(unsigned), cudaMemcpyDeviceToHost));
            }
            std::printf("%-10s schedule: %8.4f s (best of %d%s), %.2f TFLOP/s, %u zero pivots replaced\n",
                        variant == 0 ? "level" : "persistent", best, reps, writeback ? ", until all panels are on the host" : "",
                        flops / best / 1e12, zp);
            if (!bigOrder.empty())
                std::printf("           small and block kernels %.4f s, then the large-front path %.4f s (%.1f TFLOP/s); "
                            "issuing it took the host %.4f s\n", bestA, best - bestA, bigFlops / (best - bestA) / 1e12, bestIssue);
            if (writeback)
                std::printf("           kernels %.4f s, write-back after them %.4f s; %.1f GB/s of panels over the whole run\n",
                            bestKer, bestTail, panelTotal * 8.0 / 1e9 / best);
            rf.factor = best;  // with both schedules: the persistent one, which runs last
            if (!needHost) continue;
            const auto tv = Clock::now();
            std::vector<cuComplex> got(panelTotal);
            if (writeback) {  // panels lie on the host in the order they were written: put them in panelOff order
                std::vector<int64_t> hostOff(R);
                CK(cudaMemcpy(hostOff.data(), a.hostOff, R * sizeof(int64_t), cudaMemcpyDeviceToHost));
                for (int r = 0; r < R; ++r)
                    std::memcpy(got.data() + sn[r].panelOff, hostAll + hostOff[r], (size_t)sn[r].m * sn[r].k * sizeof(cuComplex));
            } else {
                CK(cudaMemcpy(got.data(), a.panels, panelTotal * sizeof(cuComplex), cudaMemcpyDeviceToHost));
            }
            if (ref.empty()) {  // FNV-1a over the panel bytes in panelOff order, to compare runs with and without write-back
                unsigned long long h = 1469598103934665603ull;
                const unsigned char* b = reinterpret_cast<const unsigned char*>(got.data());
                for (size_t i = 0; i < (size_t)panelTotal * sizeof(cuComplex); ++i) h = (h ^ b[i]) * 1099511628211ull;
                std::printf("panel hash %016llx\n", h);
                ref = std::move(got);
            } else std::printf("persistent factor %s the level schedule's\n",
                             std::memcmp(got.data(), ref.data(), panelTotal * sizeof(cuComplex)) == 0 ? "is bitwise identical to"
                                                                                                      : "DIFFERS from");
            report.verify += since(tv);
        }
        {  // the solves' inverses of the large fronts' diagonal blocks, from this factorization: part of it
            const auto ti = Clock::now();
            if (solveReps > 0 || (mixed && !precond64 && !precondDF)) fsolve.newFactor();
            if (precond64 || (solveReps > 0 && solvePrecision == "double")) fsolve64.newFactor();
            if (precondDF || (solveReps > 0 && solvePrecision == "double-float")) fsolveDF.newFactor();
            CK(cudaDeviceSynchronize());
            const double tInv = since(ti);
            rf.factor += tInv;
            if (fi == 0) std::printf("block inverses for the solve: %.4f s (counted in the factorization)\n", tInv);
        }

        if (const char* dp = std::getenv("REF_DUMP_DIAG")) {  // D by pivot, and the pivot order (debugging)
            std::vector<cuComplex> dg(n), pan(panelTotal);
            CK(cudaMemcpy(pan.data(), a.panels, panelTotal * sizeof(cuComplex), cudaMemcpyDeviceToHost));
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < sn[r].k; ++j) dg[super[regSuper[r]] + j] = pan[sn[r].panelOff + (int64_t)sn[r].m * j + j];
            std::vector<int32_t> pr(Perm, Perm + n);
            std::vector<float> lmax(n, 0);  // largest |L_ij| below the diagonal, per pivot
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < sn[r].k; ++j) {
                    const cuComplex* c = pan.data() + sn[r].panelOff + (int64_t)sn[r].m * j;
                    float mx = 0;
                    for (int i = j + 1; i < sn[r].m; ++i) mx = std::max(mx, std::hypot(c[i].x, c[i].y));
                    lmax[super[regSuper[r]] + j] = mx;
                }
            FILE* f = std::fopen(dp, "wb");
            std::fwrite(dg.data(), sizeof(cuComplex), n, f);
            std::fwrite(pr.data(), sizeof(int32_t), n, f);
            std::fwrite(lmax.data(), sizeof(float), n, f);
            std::fclose(f);
            std::printf("dumped D and the pivot order to %s\n", dp);
        }
        // L D L^T x on the region columns C (pivot order; rows outside C ignored), from the host copy
        std::vector<char> inC;
        std::vector<int64_t> colPanel;  // factor column -> offset of its column in the panel storage
        std::vector<int> colM;          // factor column -> region index
        auto ldltSetup = [&]() {
            if (inC.empty()) {
                inC.assign(n, 0);
                colPanel.assign(n, -1);
                colM.assign(n, 0);
                for (int r = 0; r < R; ++r) {
                    const int64_t s = regSuper[r];
                    for (int j = 0; j < sn[r].k; ++j) {
                        inC[super[s] + j] = 1;
                        colPanel[super[s] + j] = sn[r].panelOff + (int64_t)sn[r].m * j;
                        colM[super[s] + j] = (int)r;
                    }
                }
            }
        };
        auto ldltApply = [&](const std::vector<Complex>& x) {
            ldltSetup();
            // z = L^T x, z = D z, y = L z (unit lower L; D on the panel diagonal)
            std::vector<Complex> z(n, 0), y(n, 0);
            for (int64_t j = 0; j < n; ++j) {
                if (!inC[j]) continue;
                const int r = colM[j];
                const int64_t s = regSuper[r];
                const int jj = (int)(j - super[s]);
                Complex acc = x[j];
                for (int i = jj + 1; i < sn[r].m; ++i) {
                    const int64_t row = Ls[pi[s] + i];
                    if (!inC[row]) continue;
                    const cuComplex l = ref[colPanel[j] + i];
                    acc += Complex(l.x, l.y) * x[row];
                }
                const cuComplex d = ref[colPanel[j] + jj];
                z[j] = acc * Complex(d.x, d.y);
            }
            for (int64_t j = 0; j < n; ++j) {
                if (!inC[j]) continue;
                const int r = colM[j];
                const int64_t s = regSuper[r];
                const int jj = (int)(j - super[s]);
                y[j] += z[j];
                for (int i = jj + 1; i < sn[r].m; ++i) {
                    const int64_t row = Ls[pi[s] + i];
                    if (!inC[row]) continue;
                    const cuComplex l = ref[colPanel[j] + i];
                    y[row] += Complex(l.x, l.y) * z[j];
                }
            }
            return y;
        };

        if (mixed) {  // the full solve: FP64 FGMRES (in double-double refinement) preconditioned by the front solve
            const int64_t len = n * nports;
            char ftag[32];
            std::snprintf(ftag, sizeof(ftag), "%.3g", freq);
            const MixedSolver::Precond precond = [&](const Complex* v, Complex* z) {
                if (precond64) {  // the vectors stay in double
                    fsolve64.solve(reinterpret_cast<const cuDoubleComplex*>(v), reinterpret_cast<cuDoubleComplex*>(z), nports);
                    return;
                }
                if (precondDF) {
                    fsolveDF.solve(reinterpret_cast<const cuDoubleComplex*>(v), reinterpret_cast<cuDoubleComplex*>(z), nports);
                    return;
                }
                convertToC32(reinterpret_cast<const cuDoubleComplex*>(v), dPb, len, 1.0);
                fsolve.solve(dPb, dPx, nports);
                convertToC64(dPx, reinterpret_cast<cuDoubleComplex*>(z), len);
            };
            CK(cudaDeviceSynchronize());
            if (fi == 0) {  // the large fronts come from the stream-ordered pool, freed after each factorization
                cudaMemPool_t pool;
                size_t high = 0;
                CK(cudaDeviceGetDefaultMemPool(&pool, 0));
                CK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &high));
                std::printf("device memory in use before the first solve: %.2f GB (large fronts at most %.2f GB at once)\n",
                            memUsed(), high / 1073741824.0);
            }
            const auto tS = Clock::now();
            const MixedSolver::Result res = mixed->solve((int)fi, ftag, precond, portRows);
            const double tSolve = since(tS);
            if (fi == 0) std::printf("device memory in use after the first solve: %.2f GB\n", memUsed());
            std::printf("  solve (%d rhs)            : %8.2f s (%d preconditioner solves + fp64 FGMRES)\n", nports, tSolve,
                        res.solves);
            rf.solve = tSolve - res.ddWait;  // the dd-IR setup wait counts as analysis (pattern only, once)
            report.analysis += res.ddWait;
            rf.check("solves", res.solves);
            double delivered = res.delivered;
            if (doCheck) {
                const auto tx = Clock::now();
                std::vector<Complex> x(len);
                CK(cudaMemcpy(x.data(), dX, len * sizeof(Complex), cudaMemcpyDeviceToHost));
                for (double r : printResidual(A, A.val, x, bHost, nports)) delivered = std::max(delivered, r);
                printPortSolution(bHost, x, n, nports, "ref");
                report.verify += since(tx);
            }
            if (delivered >= 0) rf.check("residual", delivered);
        }

        const auto tv = Clock::now();
        if (cpuRef) {  // same algorithm on the CPU in complex double, compared supernode by supernode
            // update matrices at offsets of their own: with --whole the large fronts have none in updOff
            std::vector<int64_t> uOff(R + 1, 0);
            for (int r = 0; r < R; ++r) uOff[r + 1] = uOff[r] + (int64_t)(sn[r].m - sn[r].k) * (sn[r].m - sn[r].k);
            std::vector<Complex> P(panelTotal), U(std::max<int64_t>(uOff[R], 1)), F;
            int zeros = 0, firstBad = -1, nbad = 0;
            double worst = 0;
            // relative panel difference by decade, for small-path / block-kernel / large-front supernodes
            std::vector<std::array<int64_t, 3>> hist(20, {0, 0, 0});
            std::vector<std::array<double, 3>> hm(20, {0, 0, 0});  // mean front size m per bin
            for (int r = 0; r < R; ++r) {  // postorder: children first
                const int m = sn[r].m, k = sn[r].k;
                F.assign((size_t)m * m, Complex(0));
                for (int64_t e = sn[r].aBeg; e < sn[r].aEnd; ++e) F[aPos[e]] += Complex(aVal[e].x, aVal[e].y);
                for (int ci = sn[r].childBeg; ci < sn[r].childEnd; ++ci) {
                    const SNode& ch = sn[children[ci]];
                    const int mc = ch.m - ch.k;
                    for (int j = 0; j < mc; ++j)
                        for (int i = j; i < mc; ++i)
                            F[relind[ch.relOff + i] + (size_t)m * relind[ch.relOff + j]] += U[uOff[children[ci]] + i + (size_t)mc * j];
                }
                for (int j = 0; j < k; ++j) {
                    Complex d = F[j + (size_t)m * j];
                    if (d == Complex(0)) { d = 1e-30; ++zeros; }
                    F[j + (size_t)m * j] = d;
                    for (int i = j + 1; i < m; ++i) {
                        const Complex wv = F[i + (size_t)m * j];
                        F[i + (size_t)m * j] = wv / d;
                    }
                    for (int l = j + 1; l < m; ++l)
                        for (int i = l; i < m; ++i) F[i + (size_t)m * l] -= F[i + (size_t)m * j] * d * F[l + (size_t)m * j];
                }
                for (int64_t q = 0; q < (int64_t)m * k; ++q) P[sn[r].panelOff + q] = F[q];
                const int mu = m - k;
                for (int j = 0; j < mu; ++j)
                    for (int i = 0; i < mu; ++i) U[uOff[r] + i + (size_t)mu * j] = F[(k + i) + (size_t)m * (k + j)];
                double dn = 0, pn = 0;  // lower part of the panel only
                for (int j = 0; j < k; ++j)
                    for (int i = j; i < m; ++i) {
                        const Complex g(ref[sn[r].panelOff + i + (size_t)m * j].x, ref[sn[r].panelOff + i + (size_t)m * j].y);
                        dn += std::norm(g - P[sn[r].panelOff + i + (size_t)m * j]);
                        pn += std::norm(P[sn[r].panelOff + i + (size_t)m * j]);
                    }
                const double rel = std::sqrt(dn / std::max(pn, 1e-300));
            {
                const int path = isS[r] ? 0 : big[r] ? 2 : 1;
                const int bin = rel > 0 ? std::min(19, std::max(0, (int)std::floor(-std::log10(rel)))) : 19;
                hist[bin][path]++;
                hm[bin][path] += m;
            }
                if (!(rel < 1e-3)) {
                    ++nbad;
                    if (firstBad < 0) firstBad = r;
                }
                if (std::isfinite(rel)) worst = std::max(worst, rel);
            }
            std::printf("cpu-ref: supernodes by relative panel difference (FP32 vs double), small path / block kernels / "
                        "large fronts (mean m):\n");
            for (int b = 0; b < 20; ++b)
                if (hist[b][0] + hist[b][1] + hist[b][2])
                    std::printf("  1e-%02d .. 1e-%02d: %9lld (%5.0f) %9lld (%5.0f) %9lld (%5.0f)\n", b + 1, b,
                                (long long)hist[b][0], hist[b][0] ? hm[b][0] / hist[b][0] : 0.0, (long long)hist[b][1],
                                hist[b][1] ? hm[b][1] / hist[b][1] : 0.0, (long long)hist[b][2],
                                hist[b][2] ? hm[b][2] / hist[b][2] : 0.0);
            std::printf("cpu-ref: %d zero pivots in complex double; %d of %d supernodes differ by more than 1e-3 "
                        "(first: region index %d, %s), largest finite difference %.3e\n",
                        zeros, nbad, R, firstBad,
                        firstBad < 0 ? "-" : (sn[firstBad].childBeg == sn[firstBad].childEnd ? "a leaf" : "has children"),
                        worst);
            if (firstBad >= 0)
                std::printf("  first bad supernode: m %d, k %d, %d children, %lld A entries\n", sn[firstBad].m,
                            sn[firstBad].k, sn[firstBad].childEnd - sn[firstBad].childBeg,
                            (long long)(sn[firstBad].aEnd - sn[firstBad].aBeg));
        }

        // ---------------- check: A_CC x = L_CC D L_CC^T x on the region columns C
        if (doCheck) {
            std::mt19937_64 gen(7);
            std::normal_distribution<double> nd;
            std::vector<Complex> x(n, 0), y1(n, 0);
            ldltSetup();
            for (int64_t j = 0; j < n; ++j)
                if (inC[j]) x[j] = Complex(nd(gen), nd(gen));
            for (int64_t i = 0; i < n; ++i)  // y1 = A_CC x (symmetric, upper storage)
                for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q) {
                    const int64_t fi = invP[i], fj = invP[A.col[q]];
                    if (!inC[fi] || !inC[fj]) continue;
                    y1[fi] += A.val[q] * x[fj];
                    if (fi != fj) y1[fj] += A.val[q] * x[fi];
                }
            const std::vector<Complex> y2 = ldltApply(x);
            double num = 0, den = 0;
            for (int64_t i = 0; i < n; ++i)
                if (inC[i]) { num += std::norm(y1[i] - y2[i]); den += std::norm(y1[i]); }
            std::printf("check: ||A_CC x - L D L^T x|| / ||A_CC x|| = %.3e over %lld region columns\n",
                        std::sqrt(num / den), (long long)regionCols);
            rf.check("factor-check", std::sqrt(num / den));
        }
        report.verify += since(tv);

        // ---------------- the triangular solve: time, and with verification its accuracy
        if (solveReps > 0) {
            fsolve.solve(dSb, dSx, kSolveNr);  // warm-up
            CK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (int rep = 0; rep < solveReps; ++rep) {
                CK(cudaEventRecord(e0));
                fsolve.solve(dSb, dSx, kSolveNr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms);
            }
            CK(cudaGetLastError());
            if (const char* xp = std::getenv("REF_SOLVE_DUMP")) {  // x of both precisions for the same b (original order)
                std::vector<cuComplex> x32(n * kSolveNr);
                CK(cudaMemcpy(x32.data(), dSx, n * kSolveNr * sizeof(cuComplex), cudaMemcpyDeviceToHost));
                std::vector<Complex> xs(n * kSolveNr);
                for (int64_t i = 0; i < n * kSolveNr; ++i) xs[i] = Complex(x32[i].x, x32[i].y);
                FILE* f = std::fopen((std::string(xp) + "_single.xbin").c_str(), "wb");
                std::fwrite(xs.data(), sizeof(Complex), xs.size(), f);
                std::fclose(f);
                if (precond64) {
                    Complex *db = nullptr, *dx = nullptr;
                    std::vector<Complex> b32(n * kSolveNr);  // the same b as the single solve: its FP32 values
                    for (int64_t i = 0; i < n * kSolveNr; ++i) b32[i] = Complex((float)sb[i].real(), (float)sb[i].imag());
                    CK(cudaMalloc(&db, n * kSolveNr * sizeof(Complex)));
                    CK(cudaMalloc(&dx, n * kSolveNr * sizeof(Complex)));
                    CK(cudaMemcpy(db, b32.data(), n * kSolveNr * sizeof(Complex), cudaMemcpyHostToDevice));
                    fsolve64.solve(reinterpret_cast<const cuDoubleComplex*>(db), reinterpret_cast<cuDoubleComplex*>(dx), kSolveNr);
                    CK(cudaMemcpy(xs.data(), dx, n * kSolveNr * sizeof(Complex), cudaMemcpyDeviceToHost));
                    f = std::fopen((std::string(xp) + "_double.xbin").c_str(), "wb");
                    std::fwrite(xs.data(), sizeof(Complex), xs.size(), f);
                    std::fclose(f);
                    cudaFree(db);
                    cudaFree(dx);
                }
            }
            std::printf("solve: %.2f ms for %d right-hand sides (best of %d); factor panels %.2f GB, %.0f GB/s for two "
                        "reads\n", best, kSolveNr, solveReps, panelTotal * 8.0 / GB, 2 * panelTotal * 8.0 / 1e9 / (best / 1e3));
            rf.check("psolve-ms", best);
            std::vector<Complex> xw;  // the wider solve's x (--solve-precision double or double-float)
            if (solvePrecision != "single") {  // the same right-hand sides through the wider vectors
                Complex *db = nullptr, *dx = nullptr;
                std::vector<Complex> b32(n * kSolveNr);  // the FP32 values the single solve saw
                for (int64_t i = 0; i < n * kSolveNr; ++i) b32[i] = Complex((float)sb[i].real(), (float)sb[i].imag());
                CK(cudaMalloc(&db, n * kSolveNr * sizeof(Complex)));
                CK(cudaMalloc(&dx, n * kSolveNr * sizeof(Complex)));
                CK(cudaMemcpy(db, b32.data(), n * kSolveNr * sizeof(Complex), cudaMemcpyHostToDevice));
                auto run = [&]() {
                    const auto* b = reinterpret_cast<const cuDoubleComplex*>(db);
                    auto* x = reinterpret_cast<cuDoubleComplex*>(dx);
                    if (solvePrecision == "double") fsolve64.solve(b, x, kSolveNr);
                    else fsolveDF.solve(b, x, kSolveNr);
                };
                run();
                CK(cudaDeviceSynchronize());
                float bestW = 1e30f;
                for (int rep = 0; rep < solveReps; ++rep) {
                    CK(cudaEventRecord(e0));
                    run();
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    bestW = std::min(bestW, ms);
                }
                CK(cudaGetLastError());
                std::printf("solve (%s vectors): %.2f ms for %d right-hand sides (best of %d)\n", solvePrecision.c_str(), bestW,
                            kSolveNr, solveReps);
                rf.check("psolve-wide-ms", bestW);
                xw.resize(n * kSolveNr);
                CK(cudaMemcpy(xw.data(), dx, n * kSolveNr * sizeof(Complex), cudaMemcpyDeviceToHost));
                cudaFree(db);
                cudaFree(dx);
            }
            if (doCheck) {  // against the same sweeps in double from the host factor, and against A
                const auto ts = Clock::now();
                ldltSetup();
                std::vector<cuComplex> x32(n * kSolveNr);
                CK(cudaMemcpy(x32.data(), dSx, n * kSolveNr * sizeof(cuComplex), cudaMemcpyDeviceToHost));
                std::vector<Complex> xs(n * kSolveNr);
                for (int64_t i = 0; i < n * kSolveNr; ++i) xs[i] = Complex(x32[i].x, x32[i].y);
                const std::vector<double> res = residual(A, A.val, xs, sb, kSolveNr);
                double worstRes = 0, worstDiff = 0, worstRef = 0, worstDiffW = 0;
                std::vector<Complex> xref(n * kSolveNr);  // the double-precision solve, original order
                for (int c = 0; c < kSolveNr; ++c) {
                    worstRes = std::max(worstRes, res[c]);
                    std::vector<Complex> bp(n), xp(n);  // pivot order
                    for (int64_t q = 0; q < n; ++q) bp[q] = sb[c * n + Perm[q]];
                    xp = bp;
                    for (int64_t j = 0; j < n; ++j) {  // L z = b, column by column
                        const int r = colM[j];
                        const int64_t s = regSuper[r];
                        for (int i = (int)(j - super[s]) + 1; i < sn[r].m; ++i) {
                            const cuComplex l = ref[colPanel[j] + i];
                            xp[Ls[pi[s] + i]] -= Complex(l.x, l.y) * xp[j];
                        }
                    }
                    for (int64_t j = 0; j < n; ++j) {  // z / D
                        const cuComplex d = ref[colPanel[j] + (j - super[regSuper[colM[j]]])];
                        xp[j] /= Complex(d.x, d.y);
                    }
                    for (int64_t j = n - 1; j >= 0; --j) {  // L^T x = z / D
                        const int r = colM[j];
                        const int64_t s = regSuper[r];
                        for (int i = (int)(j - super[s]) + 1; i < sn[r].m; ++i) {
                            const cuComplex l = ref[colPanel[j] + i];
                            xp[j] -= Complex(l.x, l.y) * xp[Ls[pi[s] + i]];
                        }
                    }
                    const std::vector<Complex> y = ldltApply(xp);  // the reference solves L D L^T x = b
                    double num = 0, den = 0, dn = 0, xn = 0, dw = 0;
                    for (int64_t q = 0; q < n; ++q) {
                        num += std::norm(y[q] - bp[q]);
                        den += std::norm(bp[q]);
                        dn += std::norm(xs[c * n + Perm[q]] - xp[q]);
                        xn += std::norm(xp[q]);
                        if (!xw.empty()) dw += std::norm(xw[c * n + Perm[q]] - xp[q]);
                    }
                    worstDiffW = std::max(worstDiffW, std::sqrt(dw / xn));
                    worstRef = std::max(worstRef, std::sqrt(num / den));
                    worstDiff = std::max(worstDiff, std::sqrt(dn / xn));
                    for (int64_t q = 0; q < n; ++q) xref[c * n + Perm[q]] = xp[q];
                }
                double worstRefA = 0;  // the factor alone: ||b - A x|| of the exact solve with the FP32 factor
                for (double r : residual(A, A.val, xref, sb, kSolveNr)) worstRefA = std::max(worstRefA, r);
                std::printf("solve check: with the same factor solved in double, ||b - A x|| / ||b|| = %.3e\n", worstRefA);
                rf.check("presid-exact", worstRefA);
                std::printf("solve check: x differs from the double-precision solve with the same factor by %.3e (which "
                            "solves L D L^T x = b to %.1e); ||b - A x|| / ||b|| = %.3e (the FP32 preconditioner)\n",
                            worstDiff, worstRef, worstRes);
                rf.check("solve-check", worstDiff);
                if (!xw.empty()) {  // its b is the FP32-rounded b, the reference's is not: differences down to ~1e-8
                    std::printf("solve check: the %s solve differs from the double-precision solve by %.3e\n",
                                solvePrecision.c_str(), worstDiffW);
                    rf.check("solve-check-wide", worstDiffW);
                }
                rf.check("presid", worstRes);
                report.verify += since(ts);
            }
        }
    }
    report.print();
    if (Lf) cholmod_l_free_factor(&Lf, &cc);
    if (S) cholmod_l_free_sparse(&S, &cc);
    cholmod_l_finish(&cc);
    return 0;
}
