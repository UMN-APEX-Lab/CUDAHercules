// Supernodal symbolic analysis on the GPU; see gpu_symbolic.h for the steps.
//
// Numberings: "plan" = position in the plan's permutation (row i of P A P^T), "P1" = an unweighted
// postorder of the elimination tree (only for finding fundamental supernodes and column counts), "final"
// = CHOLMOD's weighted postorder, the pivot order of the factorization.
#include "gpu_symbolic.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace {

void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}
#define CK(x) ck((x), #x)

void fail(const char* msg)
{
    std::fprintf(stderr, "gpuSymbolic: %s\n", msg);
    std::exit(1);
}

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

template <class T>
struct Buf  // device array
{
    T* p = nullptr;
    size_t n = 0;
    explicit Buf(size_t n_ = 0) { alloc(n_); }
    ~Buf() { cudaFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void alloc(size_t n_)
    {
        cudaFree(p);
        p = nullptr;
        n = n_;
        CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    }
    void fill(int byte) { CK(cudaMemset(p, byte, std::max<size_t>(n, 1) * sizeof(T))); }
    T* release()  // hand the allocation over
    {
        T* r = p;
        p = nullptr;
        n = 0;
        return r;
    }
    T get(size_t i) const
    {
        T v;
        CK(cudaMemcpy(&v, p + i, sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

template <class F>
void cubRun(F f)
{
    size_t bytes = 0;
    CK(f(nullptr, bytes));
    void* t = nullptr;
    CK(cudaMalloc(&t, std::max<size_t>(bytes, 1)));
    CK(f(t, bytes));
    CK(cudaFree(t));
}

inline int gridFor(int64_t n, int threads = 256) { return (int)std::max<int64_t>(1, std::min<int64_t>((n + threads - 1) / threads, 1 << 20)); }
#define GRID_LOOP(i, n) for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)

__device__ __forceinline__ int hiOf(uint64_t k) { return (int)(k >> 32); }
__device__ __forceinline__ int loOf(uint64_t k) { return (int)(uint32_t)k; }

// ---------------------------------------------------------------- 1. lower pattern
// key = (row, col) of every strictly lower entry of P A P^T in plan numbering; the diagonal gets ~0
__global__ void lowerKeysKer(int64_t n, const int32_t* rowPtr, const int32_t* col, const int32_t* pos, uint64_t* key)
{
    GRID_LOOP(i, n)
    {
        const int a = pos[i];
        for (int32_t q = rowPtr[i]; q < rowPtr[i + 1]; ++q) {
            const int b = pos[col[q]];
            key[q] = a == b ? ~0ull : ((uint64_t)max(a, b) << 32) | (uint32_t)min(a, b);
        }
    }
}

__global__ void invertKer(int64_t n, const int32_t* p, int32_t* inv)
{
    GRID_LOOP(k, n) inv[p[k]] = (int32_t)k;
}

__global__ void rowCountKer(int64_t m, const uint64_t* key, int64_t* cnt)
{
    GRID_LOOP(t, m) atomicAdd(reinterpret_cast<unsigned long long*>(cnt + hiOf(key[t]) + 1), 1ull);
}

// ---------------------------------------------------------------- 2. elimination tree
__global__ void blockOfKer(int nb, const int32_t* start, int32_t* blockOf)
{
    // one CTA per block (blocks are contiguous row ranges in tree order)
    for (int b = blockIdx.x; b < nb; b += gridDim.x)
        for (int r = start[b] + threadIdx.x; r < start[b + 1]; r += blockDim.x) blockOf[r] = b;
}

// every edge must join a block to itself or to an ancestor (block k of level l: children 2k, 2k+1 at l-1)
__global__ void ndCheckKer(int64_t m, const uint64_t* key, const int32_t* blockOf, const int8_t* lev, const int32_t* kidx,
                           unsigned long long* bad)
{
    GRID_LOOP(t, m)
    {
        const int bh = blockOf[hiOf(key[t])], bl = blockOf[loOf(key[t])];
        const int d = lev[bh] - lev[bl];
        if (d < 0 || (kidx[bl] >> d) != kidx[bh]) atomicAdd(bad, 1ull);
    }
}

// root of x in the finished forest below the current level, with path halving (racing halvings are
// harmless: every write replaces an ancestor by a higher one)
__device__ __forceinline__ int findRoot(int32_t* anc, int x)
{
    for (;;) {
        const int a = anc[x];
        if (a < 0) return x;
        const int b = anc[a];
        if (b < 0) return a;
        anc[x] = b;
        x = b;
    }
}

// entries of this level's rows into finished subtrees: resolve to the subtree root r, and record the
// first row of the block that reaches r (it becomes r's parent)
__global__ void etreeRootsKer(int64_t e0, int64_t e1, const uint64_t* key, const int32_t* blockOf, const int32_t* start,
                              int32_t* anc, int32_t* ref, int32_t* firstRow)
{
    GRID_LOOP(tt, e1 - e0)
    {
        const int64_t t = e0 + tt;
        const int hi = hiOf(key[t]), lo = loOf(key[t]);
        if (lo >= start[blockOf[hi]]) continue;
        const int r = findRoot(anc, lo);
        ref[t] = r;
        atomicMin(firstRow + r, hi);
    }
}

// link each reached root to its first row, and turn the entry into one inside the block: from that first
// row (later rows of the block reach the same tree through it), or none for the first row itself
__global__ void etreeLinkKer(int64_t e0, int64_t e1, const uint64_t* key, const int32_t* blockOf, const int32_t* start,
                             int32_t* anc, int32_t* parent, int32_t* ref, const int32_t* firstRow)
{
    GRID_LOOP(tt, e1 - e0)
    {
        const int64_t t = e0 + tt;
        const int hi = hiOf(key[t]), lo = loOf(key[t]);
        if (lo >= start[blockOf[hi]]) continue;
        const int r = ref[t], f = firstRow[r];
        if (hi == f) {
            parent[r] = f;
            anc[r] = f;
            ref[t] = -1;
        } else ref[t] = f;
    }
}

constexpr int kChunk = 1024;
constexpr int kShRows = 16384;  // block rows kept in shared memory (64 KB); bigger blocks use global memory

// Liu's algorithm over the rows of one block per CTA, sequential in thread 0, on the block-local entries
__global__ void etreeBlockKer(int b0, const int32_t* start, const int64_t* rowPtr, const uint64_t* key, const int32_t* ref,
                              int32_t* anc, int32_t* parent, int useRef)
{
    extern __shared__ int32_t al[];  // anc of the block's rows, by row - s0
    __shared__ uint64_t kb[kChunk];
    __shared__ int32_t rb[kChunk];
    const int b = b0 + blockIdx.x, s0 = start[b], s1 = start[b + 1], size = s1 - s0;
    const bool sh = size <= kShRows;
    int32_t* A = sh ? al : anc + s0;
    if (sh)
        for (int i = threadIdx.x; i < size; i += blockDim.x) A[i] = -1;
    __syncthreads();
    const int64_t e0 = rowPtr[s0], e1 = rowPtr[s1];
    for (int64_t c = e0; c < e1; c += kChunk) {
        const int len = (int)min((int64_t)kChunk, e1 - c);
        for (int t = threadIdx.x; t < len; t += blockDim.x) {
            kb[t] = key[c + t];
            rb[t] = useRef ? ref[c + t] : 0;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            for (int t = 0; t < len; ++t) {
                const int i = hiOf(kb[t]);
                int k = loOf(kb[t]);
                if (k < s0) {
                    k = rb[t];
                    if (k < 0) continue;
                }
                for (int r = k; r != -1 && r < i;) {
                    const int nx = A[r - s0];
                    A[r - s0] = i;
                    if (nx == -1) parent[r] = i;
                    r = nx;
                }
            }
        __syncthreads();
    }
    if (sh)
        for (int i = threadIdx.x; i < size; i += blockDim.x) anc[s0 + i] = A[i];
}

// ---------------------------------------------------------------- 3. postorder (Euler tour)
// children sorted by key (parent << 32 | weight), stable in the node index
__global__ void childKeysKer(int64_t n, const int32_t* parent, const int32_t* weight, uint64_t* key, int32_t* val,
                             int nm1)
{
    GRID_LOOP(v, n)
    {
        const int p = parent[v];
        const uint32_t w = weight ? (uint32_t)min(max(weight[v], 0), nm1) : 0u;
        key[v] = p < 0 ? ~0ull : ((uint64_t)p << 32) | w;
        val[v] = (int32_t)v;
    }
}

__global__ void childLinksKer(int64_t nk, const uint64_t* key, const int32_t* val, int32_t* firstChild, int32_t* nextSib,
                              int32_t* nchild)
{
    GRID_LOOP(t, nk)
    {
        const int p = hiOf(key[t]);
        if (t == 0 || hiOf(key[t - 1]) != p) firstChild[p] = val[t];
        nextSib[val[t]] = t + 1 < nk && hiOf(key[t + 1]) == p ? val[t + 1] : -1;
        atomicAdd(nchild + p, 1);
    }
}

__global__ void rootLinksKer(int64_t n, const int32_t* parent, const int32_t* rootIdx, const int32_t* roots, int nroots,
                             int32_t* nextRoot)
{
    GRID_LOOP(v, n)
    if (parent[v] < 0) {
        const int j = rootIdx[v];
        nextRoot[v] = j + 1 < nroots ? roots[j + 1] : -1;
    }
}

// arcs 2v (down into v) and 2v+1 (up out of v); value = (weight, next); weight 1 on up arcs
__global__ void eulerKer(int64_t n, const int32_t* parent, const int32_t* firstChild, const int32_t* nextSib,
                         const int32_t* nextRoot, int2* arc)
{
    GRID_LOOP(v, n)
    {
        const int fc = firstChild[v], ns = nextSib[v], p = parent[v];
        arc[2 * v] = make_int2(0, fc >= 0 ? 2 * fc : 2 * (int)v + 1);
        const int up = ns >= 0 ? 2 * ns : p >= 0 ? 2 * p + 1 : nextRoot[v] >= 0 ? 2 * nextRoot[v] : -1;
        arc[2 * v + 1] = make_int2(1, up);
    }
}

// Wyllie's list ranking: suffix sums of the weights
__global__ void wyllieKer(int64_t N, const int2* in, int2* out)
{
    GRID_LOOP(x, N)
    {
        const int2 a = in[x];
        if (a.y < 0) out[x] = a;
        else {
            const int2 b = in[a.y];
            out[x] = make_int2(a.x + b.x, b.y);
        }
    }
}

// post[v] = n - suffix(up v), first[v] = n - suffix(down v), node[post] = v
__global__ void postFromRankKer(int64_t n, const int2* arc, int32_t* post, int32_t* first, int32_t* node)
{
    GRID_LOOP(v, n)
    {
        const int p = (int)n - arc[2 * v + 1].x;
        post[v] = p;
        first[v] = (int)n - arc[2 * v].x;
        node[p] = (int32_t)v;
    }
}

// ---------------------------------------------------------------- 4. supernodes and column counts
// entries of each row sorted by the P1 position of their column
__global__ void postKeysKer(int64_t m, const uint64_t* key, const int32_t* postRow, const int32_t* postCol, uint64_t* out)
{
    GRID_LOOP(t, m) out[t] = ((uint64_t)postRow[hiOf(key[t])] << 32) | (uint32_t)postCol[loOf(key[t])];
}

// column k is a leaf of row i's subtree when no earlier entry of the row lies in k's subtree; then k
// does not join its only child in a fundamental supernode
__global__ void rowLeafKer(int64_t m, const uint64_t* pk, const int32_t* node, const int32_t* first, int32_t* leafSomewhere)
{
    GRID_LOOP(t, m)
    {
        const int k = node[loOf(pk[t])];
        const bool prev = t > 0 && hiOf(pk[t - 1]) == hiOf(pk[t]);
        if (!prev || loOf(pk[t - 1]) < first[k]) leafSomewhere[k] = 1;
    }
}

// flag[q] = 1 where a supernode starts, q in the given numbering (node[q] = plan node at q)
__global__ void leaderKer(int64_t n, const int32_t* node, const int32_t* nchild, const int32_t* leafSomewhere, int32_t* flag)
{
    GRID_LOOP(q, n)
    {
        const int v = node[q];
        flag[q] = nchild[v] != 1 || leafSomewhere[v];
    }
}

__global__ void startsKer(int64_t n, const int32_t* flag, const int32_t* sid, int32_t* start, int ns)
{
    GRID_LOOP(q, n)
    if (flag[q]) start[sid[q]] = (int32_t)q;
    if (blockIdx.x == 0 && threadIdx.x == 0) start[ns] = (int32_t)n;
}

// supernodal tree: parent and first descendant (supernodes are numbered in a postorder of it)
__global__ void snTreeKer(int ns, const int32_t* start, const int32_t* node, const int32_t* parent, const int32_t* post,
                          const int32_t* first, const int32_t* sid, int32_t* sparent, int32_t* sfirst)
{
    GRID_LOOP(s, ns)
    {
        const int top = node[start[s + 1] - 1], p = parent[top];
        sparent[s] = p < 0 ? -1 : sid[post[p]];
        sfirst[s] = sid[first[top]];
    }
}

// Per row i: walk the supernodal tree from the supernode of every entry up to where the previous entry's
// path joins (the first ancestor of the previous entry's supernode), or up to i's own supernode for the
// row's first entry. Every supernode passed holds row i below its own columns. Keys sorted by (row,
// position of the column) in the numbering of sid. With out == nullptr only counts.
__global__ void rowWalkKer(int64_t m, const uint64_t* pk, const int32_t* sid, const int32_t* sparent, const int32_t* sfirst,
                           int32_t* cnt, const int64_t* off, const int32_t* cap, int32_t* out, unsigned long long* overflow)
{
    GRID_LOOP(t, m)
    {
        const int i = hiOf(pk[t]), st = sid[loOf(pk[t])], si = sid[i];
        const bool prev = t > 0 && hiOf(pk[t - 1]) == i;
        const int sp = prev ? sid[loOf(pk[t - 1])] : -1;
        if (prev && sp == st) continue;
        for (int u = st; prev ? !(sfirst[u] <= sp && sp <= u) : u != si; u = sparent[u]) {
            const int slot = atomicAdd(cnt + u, 1);
            if (out) {
                if (slot < cap[u]) out[off[u] + slot] = i;
                else atomicAdd(overflow, 1ull);
            }
        }
    }
}

// column counts of the plan nodes from the fundamental supernodes in P1 numbering
__global__ void colCountKer(int64_t n, const int32_t* post, const int32_t* sid, const int32_t* start, const int32_t* rows,
                            int32_t* colCount)
{
    GRID_LOOP(v, n)
    {
        const int q = post[v], s = sid[q];
        colCount[v] = (start[s + 1] - start[s]) + rows[s] - (q - start[s]);
    }
}

// ---------------------------------------------------------------- 5. final numbering
__global__ void finalKer(int64_t n, const int32_t* post2, const int32_t* parent, const int32_t* planPerm,
                         const int32_t* colCount, int32_t* perm, int32_t* parent2, int32_t* colCount2, int32_t* node2)
{
    GRID_LOOP(v, n)
    {
        const int q = post2[v];
        perm[q] = planPerm[v];
        parent2[q] = parent[v] < 0 ? -1 : post2[parent[v]];
        colCount2[q] = colCount[v];
        node2[q] = (int32_t)v;
    }
}

__global__ void snParentFinalKer(int ns, const int32_t* start, const int32_t* parent2, const int32_t* sid, int32_t* sparent)
{
    GRID_LOOP(s, ns)
    {
        const int p = parent2[start[s + 1] - 1];
        sparent[s] = p < 0 ? -1 : sid[p];
    }
}

__global__ void markStartsKer(int ns, const int32_t* start, int32_t* flag)
{
    GRID_LOOP(s, ns) flag[start[s]] = 1;
}

// ---------------------------------------------------------------- 6. Ls
__global__ void lsColsKer(int64_t n, const int32_t* sid, const int32_t* start, const int64_t* pi, int32_t* Ls)
{
    GRID_LOOP(q, n)
    {
        const int s = sid[q];
        Ls[pi[s] + (q - start[s])] = (int32_t)q;
    }
}

__global__ void lsRowsKer(int64_t total, int ns, const int64_t* off, const int32_t* start, const int64_t* pi,
                          const int32_t* rows, int32_t* Ls)
{
    GRID_LOOP(e, total)
    {
        int lo = 0, hi = ns - 1;  // last s with off[s] <= e
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (off[mid] <= e) lo = mid;
            else hi = mid - 1;
        }
        Ls[pi[lo] + (start[lo + 1] - start[lo]) + (e - off[lo])] = rows[e];
    }
}

__global__ void countKeyKer(int64_t nnz, const uint64_t* key, unsigned long long* diag)
{
    GRID_LOOP(t, nnz)
    if (key[t] == ~0ull) atomicAdd(diag, 1ull);
}

__global__ void addKer(int64_t n, int32_t* v, int32_t d)
{
    GRID_LOOP(i, n) v[i] += d;
}

__global__ void rootFlagKer(int64_t n, const int32_t* parent, int32_t* flag)
{
    GRID_LOOP(v, n) flag[v] = parent[v] < 0;
}

__global__ void rootListKer(int64_t n, const int32_t* flag, const int32_t* idx, int32_t* roots)
{
    GRID_LOOP(v, n)
    if (flag[v]) roots[idx[v]] = (int32_t)v;
}

__global__ void gatherKer(int64_t n, const int32_t* idx, const int32_t* src, int32_t* dst)
{
    GRID_LOOP(i, n) dst[i] = src[idx[i]];
}

// invPerm[original row] = pivot
__global__ void invPermKer(int64_t n, const int32_t* pos, const int32_t* post2, int32_t* invPerm)
{
    GRID_LOOP(i, n) invPerm[i] = post2[pos[i]];
}

__global__ void narrowKer(int64_t n, const int64_t* in, int32_t* out)
{
    GRID_LOOP(i, n) out[i] = (int32_t)in[i];
}

__device__ __forceinline__ int64_t lowerBound(const int32_t* a, int64_t lo, int64_t hi, int v)
{
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (a[mid] < v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// one warp per region supernode: positions of its update rows in the parent's row list
__global__ void relindKer(int R, const int32_t* regSuper, const int32_t* parentReg, const int64_t* relOff,
                          const int32_t* super, const int64_t* pi, const int32_t* Ls, int32_t* relind,
                          int32_t* urow, unsigned long long* missing)
{
    const int lane = threadIdx.x & 31, nw = (gridDim.x * blockDim.x) >> 5;
    for (int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; r < R; r += nw) {
        const int pr = parentReg[r];
        if (pr < 0) continue;
        const int s = regSuper[r], ps = regSuper[pr], k = super[s + 1] - super[s];
        const int64_t b = pi[s], e = pi[s + 1], pb = pi[ps], pe = pi[ps + 1];
        for (int64_t i = b + k + lane; i < e; i += 32) {
            const int row = Ls[i];
            const int64_t at = lowerBound(Ls, pb, pe, row);
            if (at == pe || Ls[at] != row) atomicAdd(missing, 1ull);
            relind[relOff[r] + (i - b - k)] = (int32_t)(at - pb);
            urow[relOff[r] + (i - b - k)] = row;
        }
    }
}

// every CSR entry of A: its region supernode (INT_MAX outside the region) and its position in that front
__global__ void assemblyKer(int64_t n, const int32_t* rowPtr, const int32_t* col, const int32_t* invPerm, const int32_t* sid,
                            const int32_t* reg, const int32_t* super, const int64_t* pi, const int32_t* Ls, int32_t* key,
                            int32_t* pos, unsigned long long* missing)
{
    GRID_LOOP(i, n)
    {
        const int fi = invPerm[i];
        for (int32_t q = rowPtr[i]; q < rowPtr[i + 1]; ++q) {
            const int fj = invPerm[col[q]], lo = min(fi, fj), hi = max(fi, fj), s = sid[lo], r = reg[s];
            key[q] = r < 0 ? INT_MAX : r;
            if (r < 0) continue;
            const int64_t b = pi[s], e = pi[s + 1], at = lowerBound(Ls, b, e, hi);
            if (at == e || Ls[at] != hi) atomicAdd(missing, 1ull);
            pos[q] = (int)(at - b) + (int)(e - b) * (lo - super[s]);
        }
    }
}

__global__ void iotaKer(int64_t n, int32_t* v)
{
    GRID_LOOP(i, n) v[i] = (int32_t)i;
}

__global__ void keyCountKer(int64_t n, const int32_t* key, int64_t* cnt)
{
    GRID_LOOP(q, n)
    if (key[q] != INT_MAX) atomicAdd(reinterpret_cast<unsigned long long*>(cnt + key[q] + 1), 1ull);
}

// supernode id of every position from the start flags (inclusive scan - 1); returns the count
int supernodeIds(int64_t n, const int32_t* flag, int32_t* sid)
{
    cubRun([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, flag, sid, n); });
    int ns = 0;
    CK(cudaMemcpy(&ns, sid + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    addKer<<<gridFor(n), 256>>>(n, sid, -1);
    return ns;
}

template <class T>
void sortKeys(Buf<T>& k, int64_t count)  // in place (through a second buffer)
{
    Buf<T> alt(count);
    cub::DoubleBuffer<T> dk(k.p, alt.p);
    cubRun([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, dk, count); });
    if (dk.Current() != k.p) std::swap(k.p, alt.p);
}

// Euler-tour postorder of the forest given by parent (roots in increasing index); children ordered by
// (weight, index), or by index without weights. post[v] = position of v, first[v] = position of the first
// node of v's subtree, node[position] = v, nchild[v] = children of v.
void postorder(int64_t n, const int32_t* parent, const int32_t* weight, int32_t* post, int32_t* first, int32_t* node,
               int32_t* nchild)
{
    Buf<int32_t> firstChild(n), nextSib(n), nextRoot(n);
    firstChild.fill(0xff);
    nextSib.fill(0xff);
    nextRoot.fill(0xff);
    CK(cudaMemset(nchild, 0, n * sizeof(int32_t)));
    {
        Buf<uint64_t> k0(n), k1(n);
        Buf<int32_t> v0(n), v1(n);
        childKeysKer<<<gridFor(n), 256>>>(n, parent, weight, k0.p, v0.p, (int)n - 1);
        cub::DoubleBuffer<uint64_t> dk(k0.p, k1.p);
        cub::DoubleBuffer<int32_t> dv(v0.p, v1.p);
        cubRun([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dv, (int64_t)n); });
        Buf<int32_t> flag(n), idx(n), roots(n);
        rootFlagKer<<<gridFor(n), 256>>>(n, parent, flag.p);
        cubRun([&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, flag.p, idx.p, (int64_t)n); });
        const int nroots = idx.get(n - 1) + flag.get(n - 1);
        rootListKer<<<gridFor(n), 256>>>(n, flag.p, idx.p, roots.p);
        rootLinksKer<<<gridFor(n), 256>>>(n, parent, idx.p, roots.p, nroots, nextRoot.p);
        const int64_t nkids = n - nroots;  // roots (key ~0) sort last
        childLinksKer<<<gridFor(nkids), 256>>>(nkids, dk.Current(), dv.Current(), firstChild.p, nextSib.p, nchild);
    }
    Buf<int2> a0(2 * n), a1(2 * n);
    eulerKer<<<gridFor(n), 256>>>(n, parent, firstChild.p, nextSib.p, nextRoot.p, a0.p);
    int2 *in = a0.p, *out = a1.p;
    for (int64_t len = 1; len < 2 * n; len *= 2) {
        wyllieKer<<<gridFor(2 * n), 256>>>(2 * n, in, out);
        std::swap(in, out);
    }
    postFromRankKer<<<gridFor(n), 256>>>(n, in, post, first, node);
    CK(cudaGetLastError());
}

// CHOLMOD's merge rule (cholmod_super_symbolic, default nrelax {4, 16, 48}, zrelax {0.8, 0.1, 0.05}) for
// supernode s (nscol0 columns, snz0 rows) and the current node s + 1 (nscol1, snz1, totzeros zeros);
// totzeros becomes the merged node's zeros. Every quantity before the one division is an integer below
// 2^53, so the result does not depend on fused multiply-adds.
__device__ bool relaxMerge(int64_t nscol0, int64_t nscol1, int64_t snz0, int64_t snz1, int64_t& totzeros)
{
    const int64_t ns = nscol0 + nscol1;
    if (ns <= 4) return true;
    const double lnz0 = (double)snz0, lnz1 = (double)snz1;
    const double xnewzeros = nscol0 * (lnz1 + nscol0 - lnz0);
    const int64_t newzeros = nscol0 * (snz1 + nscol0 - snz0);
    if (xnewzeros == 0) return true;
    const double xtotzeros = (double)totzeros + xnewzeros;
    const double xns = (double)ns;
    const double xtotsize = (xns * (xns + 1) / 2) + xns * (lnz1 - nscol1);
    const double z = xtotzeros / xtotsize;
    totzeros += newzeros;
    return ((ns <= 16 && z < 0.8) || (ns <= 48 && z < 0.1) || (z < 0.05)) && (xtotsize < (double)INT64_MAX / sizeof(double));
}

// CHOLMOD's amalgamation is a pass from the last fundamental supernode down: node s joins the current
// node s + 1 (s + 1 and the nodes merged into it, up to e) if s's parent lies in it and the rule above
// agrees. So the state of s (e, columns, rows, zeros) is a function of the state of s + 1 alone, and
// becomes s's own fresh state whenever s does not merge. The pass is run in chunks of C supernodes,
// one thread each: the first pass guesses that each chunk starts from a fresh node, later passes start
// from the bottom state the chunk above had at the end of the previous pass and stop where the state
// stops changing. When no chunk bottom changes, the states are those of the sequential pass.
__global__ void amalgChunkKer(int nf, int C, int pass, const int32_t* start, const int32_t* sp, const int32_t* snzF,
                              int32_t* E, int32_t* NS, int32_t* SZ, int64_t* Z, int8_t* M, const int32_t* bE,
                              const int32_t* bNS, const int32_t* bSZ, const int64_t* bZ)
{
    const int nch = (nf + C - 1) / C;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < nch; c += gridDim.x * blockDim.x) {
        const int lo = c * C, hi = min(nf, lo + C) - 1;
        int inE = -1, inNs = 0, inSz = 0;
        int64_t inZ = 0;
        if (hi + 1 < nf) {
            if (pass == 0) {  // guess: the node above starts fresh
                inE = hi + 1;
                inNs = start[hi + 2] - start[hi + 1];
                inSz = snzF[hi + 1];
            } else {
                inE = bE[c + 1];
                inNs = bNS[c + 1];
                inSz = bSZ[c + 1];
                inZ = bZ[c + 1];
            }
        }
        for (int s = hi; s >= lo; --s) {
            int e = s, ns = start[s + 1] - start[s], sz = snzF[s];
            int64_t z = 0;
            int8_t merged = 0;
            const int p = sp[s];
            if (s + 1 < nf && p >= 0 && p <= inE) {  // the parent lies in the current node s + 1
                int64_t tz = inZ;
                if (relaxMerge(ns, inNs, sz, inSz, tz)) {
                    merged = 1;
                    z = tz;
                    e = inE;
                    sz = ns + inSz;
                    ns += inNs;
                }
            }
            if (pass > 0 && E[s] == e && NS[s] == ns && SZ[s] == sz && Z[s] == z) break;  // unchanged from here down
            E[s] = e;
            NS[s] = ns;
            SZ[s] = sz;
            Z[s] = z;
            M[s] = merged;
            inE = e;
            inNs = ns;
            inSz = sz;
            inZ = z;
        }
    }
}

// chunk bottoms for the next pass; counts those that changed
__global__ void amalgBottomKer(int nf, int C, const int32_t* E, const int32_t* NS, const int32_t* SZ, const int64_t* Z,
                               int32_t* bE, int32_t* bNS, int32_t* bSZ, int64_t* bZ, int* changed)
{
    const int nch = (nf + C - 1) / C;
    GRID_LOOP(c, nch)
    {
        const int lo = (int)c * C;
        if (bE[c] != E[lo] || bNS[c] != NS[lo] || bSZ[c] != SZ[lo] || bZ[c] != Z[lo]) {
            bE[c] = E[lo];
            bNS[c] = NS[lo];
            bSZ[c] = SZ[lo];
            bZ[c] = Z[lo];
            atomicAdd(changed, 1);
        }
    }
}

// a fundamental supernode starts a relaxed one unless it was merged into the one before
__global__ void liveKer(int nf, const int8_t* M, int32_t* live)
{
    GRID_LOOP(s, nf) live[s] = s == 0 || !M[s - 1];
}

__global__ void relaxedListKer(int nf, const int32_t* live, const int32_t* rid, const int32_t* startF, const int32_t* SZ,
                               int32_t* superR, int64_t* snzR, int ns, int n)
{
    GRID_LOOP(s, nf)
    if (live[s]) {
        superR[rid[s]] = startF[s];
        snzR[rid[s]] = SZ[s];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) superR[ns] = n;
}

// rows below the columns of each relaxed supernode
__global__ void capKer(int ns, const int32_t* superR, const int64_t* snzR, int32_t* cap, int64_t* cap64)
{
    GRID_LOOP(s, ns)
    {
        cap[s] = (int32_t)(snzR[s] - (superR[s + 1] - superR[s]));
        cap64[s] = cap[s];
    }
}

__global__ void mismatchKer(int ns, const int32_t* a, const int32_t* b, unsigned long long* bad)
{
    GRID_LOOP(s, ns)
    if (a[s] != b[s]) atomicAdd(bad, 1ull);
}

__global__ void sumKer(int64_t n, const int32_t* v, unsigned long long* sum)
{
    unsigned long long acc = 0;
    GRID_LOOP(i, n) acc += (unsigned long long)v[i];
    atomicAdd(sum, acc);
}

}  // namespace

void gpuSymbolic(int64_t n, const int32_t* hRowPtr, const int32_t* hCol, const int32_t* hPlanPerm,
                 const std::vector<int32_t>& tree, SymbolicResult& res, SymbolicDevice* dev, bool hostLs, bool verbose)
{
    if (n >= INT_MAX / 2) fail("n too large for 32-bit indices");
    res = SymbolicResult();
    res.n = n;
    const int64_t nnz = hRowPtr[n];
    auto t = Clock::now();
    Buf<int32_t> rowPtr(n + 1), col(nnz), planPerm(n), pos(n);
    CK(cudaMemcpy(rowPtr.p, hRowPtr, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(col.p, hCol, nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(planPerm.p, hPlanPerm, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    res.tUpload = since(t);

    // ---- 1. strictly lower pattern in plan numbering, sorted by (row, col)
    t = Clock::now();
    invertKer<<<gridFor(n), 256>>>(n, planPerm.p, pos.p);
    Buf<uint64_t> key(nnz);
    lowerKeysKer<<<gridFor(n), 256>>>(n, rowPtr.p, col.p, pos.p, key.p);
    if (!dev) {
        rowPtr.alloc(0);
        col.alloc(0);
    }
    sortKeys(key, nnz);
    int64_t m = 0;  // strictly lower entries (the diagonal keys sort last)
    {
        Buf<unsigned long long> diag(1);
        diag.fill(0);
        countKeyKer<<<gridFor(nnz), 256>>>(nnz, key.p, diag.p);
        m = nnz - (int64_t)diag.get(0);
    }
    Buf<int64_t> lrow(n + 1);  // row pointers of the lower pattern (plan numbering)
    lrow.fill(0);
    rowCountKer<<<gridFor(m), 256>>>(m, key.p, lrow.p);
    cubRun([&](void* tt, size_t& b) { return cub::DeviceScan::InclusiveSum(tt, b, lrow.p, lrow.p, n + 1); });
    CK(cudaDeviceSynchronize());
    res.tPattern = since(t);

    // ---- 2. elimination tree, ND level by level
    t = Clock::now();
    const int nb = (int)tree.size();
    int L = 0;
    while ((1 << L) - 1 < nb) ++L;
    if ((1 << L) - 1 != nb) fail("the ND tree must have 2^L - 1 blocks");
    std::vector<int32_t> hStart(nb + 1, 0), hKidx(nb);
    std::vector<int8_t> hLev(nb);
    std::vector<int> lvOff(L + 1, 0);
    for (int l = 0; l < L; ++l) lvOff[l + 1] = lvOff[l] + (1 << (L - 1 - l));
    for (int l = 0; l < L; ++l)
        for (int b = lvOff[l]; b < lvOff[l + 1]; ++b) {
            hLev[b] = (int8_t)l;
            hKidx[b] = b - lvOff[l];
        }
    for (int b = 0; b < nb; ++b) hStart[b + 1] = hStart[b] + tree[b];
    if (hStart[nb] != n) fail("the ND tree does not match the matrix");
    Buf<int32_t> start(nb + 1), kidx(nb), blockOf(n);
    Buf<int8_t> lev(nb);
    CK(cudaMemcpy(start.p, hStart.data(), (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kidx.p, hKidx.data(), nb * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(lev.p, hLev.data(), nb * sizeof(int8_t), cudaMemcpyHostToDevice));
    blockOfKer<<<std::min(nb, 65535), 256>>>(nb, start.p, blockOf.p);
    {
        Buf<unsigned long long> bad(1);
        bad.fill(0);
        ndCheckKer<<<gridFor(m), 256>>>(m, key.p, blockOf.p, lev.p, kidx.p, bad.p);
        if (bad.get(0)) fail("an edge of A joins two ND blocks of which neither is an ancestor of the other");
    }
    Buf<int32_t> parent(n);
    parent.fill(0xff);
    {
        Buf<int32_t> anc(n), firstRow(n), ref(m);
        anc.fill(0xff);
        firstRow.fill(0x7f);  // 0x7f7f7f7f > any row
        std::vector<int64_t> hLrow(n + 1);
        CK(cudaMemcpy(hLrow.data(), lrow.p, (n + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CK(cudaFuncSetAttribute(etreeBlockKer, cudaFuncAttributeMaxDynamicSharedMemorySize, kShRows * (int)sizeof(int32_t)));
        for (int l = 0; l < L; ++l) {
            const int b0 = lvOff[l], cnt = lvOff[l + 1] - lvOff[l];
            const int64_t e0 = hLrow[hStart[b0]], e1 = hLrow[hStart[b0 + cnt]];
            if (l > 0 && e1 > e0) {
                etreeRootsKer<<<gridFor(e1 - e0), 256>>>(e0, e1, key.p, blockOf.p, start.p, anc.p, ref.p, firstRow.p);
                etreeLinkKer<<<gridFor(e1 - e0), 256>>>(e0, e1, key.p, blockOf.p, start.p, anc.p, parent.p, ref.p, firstRow.p);
            }
            int maxRows = 0;
            for (int b = b0; b < b0 + cnt; ++b) maxRows = std::max(maxRows, tree[b]);
            const int shBytes = std::min(maxRows, kShRows) * (int)sizeof(int32_t);
            etreeBlockKer<<<cnt, 128, shBytes>>>(b0, start.p, lrow.p, key.p, ref.p, anc.p, parent.p, l > 0);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
    blockOf.alloc(0);
    lrow.alloc(0);
    res.tEtree = since(t);

    // ---- 3. postorder P1 and fundamental supernodes
    t = Clock::now();
    Buf<int32_t> post1(n), first1(n), node1(n), nchild(n);
    postorder(n, parent.p, nullptr, post1.p, first1.p, node1.p, nchild.p);
    Buf<uint64_t> pk(m);  // (P1 row, P1 col) of every lower entry, sorted
    postKeysKer<<<gridFor(m), 256>>>(m, key.p, post1.p, post1.p, pk.p);
    sortKeys(pk, m);
    Buf<int32_t> leafSomewhere(n), flag(n), sid1(n);
    leafSomewhere.fill(0);
    rowLeafKer<<<gridFor(m), 256>>>(m, pk.p, node1.p, first1.p, leafSomewhere.p);
    leaderKer<<<gridFor(n), 256>>>(n, node1.p, nchild.p, leafSomewhere.p, flag.p);
    const int nf = supernodeIds(n, flag.p, sid1.p);
    res.nfsuper = nf;
    Buf<int32_t> fstart1(nf + 1), sparent1(nf), sfirst1(nf);
    startsKer<<<gridFor(n), 256>>>(n, flag.p, sid1.p, fstart1.p, nf);
    snTreeKer<<<gridFor(nf), 256>>>(nf, fstart1.p, node1.p, parent.p, post1.p, first1.p, sid1.p, sparent1.p, sfirst1.p);
    CK(cudaDeviceSynchronize());
    res.tPost = since(t);

    // ---- 4. column counts: rows below the columns of every fundamental supernode
    t = Clock::now();
    Buf<int32_t> rows1(nf), colCount(n);
    rows1.fill(0);
    rowWalkKer<<<gridFor(m), 256>>>(m, pk.p, sid1.p, sparent1.p, sfirst1.p, rows1.p, nullptr, nullptr, nullptr, nullptr);
    colCountKer<<<gridFor(n), 256>>>(n, post1.p, sid1.p, fstart1.p, rows1.p, colCount.p);
    pk.alloc(0);
    CK(cudaDeviceSynchronize());
    res.tCounts = since(t);

    // ---- 5. weighted postorder = final numbering; fundamental supernodes in it (the same column sets)
    t = Clock::now();
    Buf<int32_t> post2(n), first2(n), node2(n), nchild2(n);
    postorder(n, parent.p, colCount.p, post2.p, first2.p, node2.p, nchild2.p);
    Buf<int32_t> perm(n), parent2(n), colCount2(n);
    finalKer<<<gridFor(n), 256>>>(n, post2.p, parent.p, planPerm.p, colCount.p, perm.p, parent2.p, colCount2.p, node2.p);
    leaderKer<<<gridFor(n), 256>>>(n, node2.p, nchild.p, leafSomewhere.p, flag.p);
    Buf<int32_t> sidF(n);
    if (supernodeIds(n, flag.p, sidF.p) != nf) fail("fundamental supernodes differ between the two postorders");
    Buf<int32_t> fstartF(nf + 1), sparentF(nf), snzF(nf);
    startsKer<<<gridFor(n), 256>>>(n, flag.p, sidF.p, fstartF.p, nf);
    snParentFinalKer<<<gridFor(nf), 256>>>(nf, fstartF.p, parent2.p, sidF.p, sparentF.p);
    gatherKer<<<gridFor(nf), 256>>>(nf, fstartF.p, colCount2.p, snzF.p);
    CK(cudaDeviceSynchronize());
    res.tPost += since(t);

    // ---- 6. relaxed amalgamation
    t = Clock::now();
    int ns = 0;
    std::vector<int64_t> hPi;
    int64_t total = 0;
    Buf<int32_t> superR, cap(0);
    Buf<int64_t> off, pi;
    {
        constexpr int C = 256;
        const int nch = (nf + C - 1) / C;
        Buf<int32_t> E(nf), NS(nf), SZ(nf), bE(nch), bNS(nch), bSZ(nch), live(nf), rid(nf);
        Buf<int64_t> Z(nf), bZ(nch);
        Buf<int8_t> M(nf);
        Buf<int> changed(1);
        bE.fill(0xff);  // no valid state: the first pass always changes every bottom
        int passes = 0;
        for (int pass = 0;; ++pass) {
            amalgChunkKer<<<gridFor(nch, 128), 128>>>(nf, C, pass, fstartF.p, sparentF.p, snzF.p, E.p, NS.p, SZ.p, Z.p, M.p,
                                                       bE.p, bNS.p, bSZ.p, bZ.p);
            changed.fill(0);
            amalgBottomKer<<<gridFor(nch), 256>>>(nf, C, E.p, NS.p, SZ.p, Z.p, bE.p, bNS.p, bSZ.p, bZ.p, changed.p);
            passes = pass + 1;
            if (pass > 0 && changed.get(0) == 0) break;
        }
        liveKer<<<gridFor(nf), 256>>>(nf, M.p, live.p);
        ns = supernodeIds(nf, live.p, rid.p);
        superR.alloc(ns + 1);
        cap.alloc(ns);
        off.alloc(ns + 1);
        pi.alloc(ns + 1);
        Buf<int64_t> snzR(ns + 1), cap64(ns + 1);
        relaxedListKer<<<gridFor(nf), 256>>>(nf, live.p, rid.p, fstartF.p, SZ.p, superR.p, snzR.p, ns, (int)n);
        capKer<<<gridFor(ns), 256>>>(ns, superR.p, snzR.p, cap.p, cap64.p);
        cubRun([&](void* tt, size_t& b) { return cub::DeviceScan::ExclusiveSum(tt, b, cap64.p, off.p, (int64_t)ns + 1); });
        cubRun([&](void* tt, size_t& b) { return cub::DeviceScan::ExclusiveSum(tt, b, snzR.p, pi.p, (int64_t)ns + 1); });
        hPi.resize(ns + 1);
        CK(cudaMemcpy(hPi.data(), pi.p, (ns + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));
        total = off.get(ns);
        res.amalgPasses = passes;
    }
    res.nsuper = ns;
    CK(cudaDeviceSynchronize());
    res.tAmalg = since(t);

    // ---- 7. row structures of the relaxed supernodes
    t = Clock::now();
    Buf<int32_t> fill(ns), sidR(n), sparentR(ns), sfirstR(ns);
    flag.fill(0);
    markStartsKer<<<gridFor(ns), 256>>>(ns, superR.p, flag.p);
    if (supernodeIds(n, flag.p, sidR.p) != ns) fail("relaxed supernode map");
    snTreeKer<<<gridFor(ns), 256>>>(ns, superR.p, node2.p, parent.p, post2.p, first2.p, sidR.p, sparentR.p, sfirstR.p);
    Buf<uint64_t> fk(m);  // (final row, final col) of every lower entry, sorted
    postKeysKer<<<gridFor(m), 256>>>(m, key.p, post2.p, post2.p, fk.p);
    key.alloc(0);
    sortKeys(fk, m);
    Buf<int32_t> rows(total), rowsSorted(total);
    Buf<unsigned long long> overflow(1);
    overflow.fill(0);
    fill.fill(0);
    rowWalkKer<<<gridFor(m), 256>>>(m, fk.p, sidR.p, sparentR.p, sfirstR.p, fill.p, off.p, cap.p, rows.p, overflow.p);
    fk.alloc(0);
    {
        Buf<unsigned long long> bad(1);
        bad.fill(0);
        mismatchKer<<<gridFor(ns), 256>>>(ns, fill.p, cap.p, bad.p);
        if (overflow.get(0) || bad.get(0)) fail("row structures do not match the column counts");
    }
    cubRun([&](void* tt, size_t& b) {
        return cub::DeviceSegmentedSort::SortKeys(tt, b, rows.p, rowsSorted.p, total, (int64_t)ns, off.p, off.p + 1);
    });
    rows.alloc(0);
    Buf<int32_t> Ls(hPi[ns]);
    lsColsKer<<<gridFor(n), 256>>>(n, sidR.p, superR.p, pi.p, Ls.p);
    lsRowsKer<<<gridFor(total), 256>>>(total, ns, off.p, superR.p, pi.p, rowsSorted.p, Ls.p);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    res.tStruct = since(t);

    // ---- results to the host (CHOLMOD's layout: 64-bit)
    t = Clock::now();
    auto down = [](const Buf<int32_t>& d, int64_t count, std::vector<int64_t>& h) {
        std::vector<int32_t> tmp(count);
        CK(cudaMemcpy(tmp.data(), d.p, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
        h.resize(count);
#pragma omp parallel for
        for (int64_t i = 0; i < count; ++i) h[i] = tmp[i];
    };
    down(perm, n, res.perm);
    if (hostLs) {  // for checks and comparisons only
        down(colCount2, n, res.colCount);
        down(Ls, hPi[ns], res.Ls);
    }
    res.sparent.resize(ns);
    CK(cudaMemcpy(res.sparent.data(), sparentR.p, ns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    {
        std::vector<int32_t> hSuperR(ns + 1);
        CK(cudaMemcpy(hSuperR.data(), superR.p, (ns + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
        res.super.assign(hSuperR.begin(), hSuperR.end());
        Buf<unsigned long long> sum(1);
        sum.fill(0);
        sumKer<<<gridFor(n), 256>>>(n, colCount2.p, sum.p);
        res.lnz = (double)sum.get(0);
    }
    res.pi = hPi;
    if (dev) {  // keep the device copies
        dev->release();
        dev->n = n;
        dev->nnz = nnz;
        dev->nsuper = ns;
        invPermKer<<<gridFor(n), 256>>>(n, pos.p, post2.p, planPerm.p);  // planPerm is free by now
        dev->invPerm = planPerm.release();
        dev->rowPtr = rowPtr.release();
        dev->col = col.release();
        dev->super = superR.release();
        dev->pi = pi.release();
        dev->Ls = Ls.release();
        dev->sid = sidR.release();
        CK(cudaDeviceSynchronize());
    }
    res.tDownload = since(t);
    if (verbose)
        std::printf("GPU symbolic: %lld fundamental / %lld relaxed supernodes, nnz(L) %.4g, %.2f s (upload %.2f, pattern "
                    "%.2f, etree %.2f, postorders + supernodes %.2f, column counts %.2f, amalgamation %.2f (%d passes), "
                    "structures %.2f, download %.2f)\n",
                    (long long)res.nfsuper, (long long)res.nsuper, res.lnz,
                    res.tUpload + res.tPattern + res.tEtree + res.tPost + res.tCounts + res.tAmalg + res.tStruct + res.tDownload,
                    res.tUpload, res.tPattern, res.tEtree, res.tPost, res.tCounts, res.tAmalg, res.amalgPasses, res.tStruct,
                    res.tDownload);
}

void SymbolicDevice::release()
{
    for (void* p : {(void*)rowPtr, (void*)col, (void*)invPerm, (void*)super, (void*)pi, (void*)Ls, (void*)sid}) cudaFree(p);
    rowPtr = col = invPerm = super = Ls = sid = nullptr;
    pi = nullptr;
    n = nnz = nsuper = 0;
}

void uploadSymbolic(int64_t n, const int32_t* hRowPtr, const int32_t* hCol, const int64_t* perm, int64_t ns,
                    const int64_t* hSuper, const int64_t* hPi, const int64_t* hLs, SymbolicDevice& d)
{
    d.release();
    d.n = n;
    d.nnz = hRowPtr[n];
    d.nsuper = ns;
    auto up32 = [](const int64_t* h, int64_t len) {  // 64-bit host -> 32-bit device
        Buf<int64_t> w(len);
        Buf<int32_t> out(len);
        CK(cudaMemcpy(w.p, h, len * sizeof(int64_t), cudaMemcpyHostToDevice));
        narrowKer<<<gridFor(len), 256>>>(len, w.p, out.p);
        CK(cudaDeviceSynchronize());
        return out.release();
    };
    Buf<int32_t> rowPtr(n + 1), col(d.nnz), sid(n), flag(n), inv(n);
    CK(cudaMemcpy(rowPtr.p, hRowPtr, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(col.p, hCol, d.nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    d.rowPtr = rowPtr.release();
    d.col = col.release();
    d.super = up32(hSuper, ns + 1);
    d.Ls = up32(hLs, hPi[ns]);
    Buf<int64_t> pi(ns + 1);
    CK(cudaMemcpy(pi.p, hPi, (ns + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    d.pi = pi.release();
    {
        Buf<int32_t> p32(n);
        int32_t* tmp = up32(perm, n);
        CK(cudaMemcpy(p32.p, tmp, n * sizeof(int32_t), cudaMemcpyDeviceToDevice));
        cudaFree(tmp);
        invertKer<<<gridFor(n), 256>>>(n, p32.p, inv.p);
    }
    d.invPerm = inv.release();
    flag.fill(0);
    markStartsKer<<<gridFor(ns), 256>>>((int)ns, d.super, flag.p);
    supernodeIds(n, flag.p, sid.p);
    d.sid = sid.release();
    CK(cudaDeviceSynchronize());
}

void gpuFrontMaps(const SymbolicDevice& d, const std::vector<int32_t>& reg, const std::vector<int64_t>& regSuper,
                  const std::vector<int32_t>& parentReg, const std::vector<int64_t>& relOff, int64_t relLen,
                  int32_t** relind, int32_t** urow, int32_t** aPos, int32_t** aSrc, std::vector<int64_t>& aBeg)
{
    const int R = (int)regSuper.size();
    const int64_t n = d.n, nnz = d.nnz;
    Buf<int32_t> dReg(d.nsuper), dRegSuper(R), dParentReg(R), rel(relLen), rows(relLen);
    Buf<int64_t> dRelOff(R);
    std::vector<int32_t> rs32(regSuper.begin(), regSuper.end());
    CK(cudaMemcpy(dReg.p, reg.data(), d.nsuper * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRegSuper.p, rs32.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dParentReg.p, parentReg.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRelOff.p, relOff.data(), R * sizeof(int64_t), cudaMemcpyHostToDevice));
    Buf<unsigned long long> missing(2);
    missing.fill(0);
    relindKer<<<gridFor((int64_t)R * 32), 256>>>(R, dRegSuper.p, dParentReg.p, dRelOff.p, d.super, d.pi, d.Ls, rel.p, rows.p,
                                                 missing.p);
    Buf<int32_t> key(nnz), pos(nnz);
    assemblyKer<<<gridFor(n), 256>>>(n, d.rowPtr, d.col, d.invPerm, d.sid, dReg.p, d.super, d.pi, d.Ls, key.p, pos.p,
                                     missing.p + 1);
    if (missing.get(0)) fail("a child row is missing from its parent's structure");
    if (missing.get(1)) fail("an entry of A is missing from its supernode's structure");
    // group by region supernode, in CSR order (stable sort)
    Buf<int32_t> k1(nnz), v0(nnz), v1(nnz);
    iotaKer<<<gridFor(nnz), 256>>>(nnz, v0.p);
    Buf<int64_t> cnt(R + 1);
    cnt.fill(0);
    keyCountKer<<<gridFor(nnz), 256>>>(nnz, key.p, cnt.p);
    cubRun([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, cnt.p, cnt.p, (int64_t)R + 1); });
    aBeg.resize(R + 1);
    CK(cudaMemcpy(aBeg.data(), cnt.p, (R + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));
    const int64_t len = aBeg[R];
    cub::DoubleBuffer<int32_t> dk(key.p, k1.p), dv(v0.p, v1.p);
    cubRun([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dv, nnz); });
    Buf<int32_t> outPos(len), outSrc(len);
    CK(cudaMemcpy(outSrc.p, dv.Current(), len * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    gatherKer<<<gridFor(len), 256>>>(len, outSrc.p, pos.p, outPos.p);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    *relind = rel.release();
    *urow = rows.release();
    *aPos = outPos.release();
    *aSrc = outSrc.release();
}
