// GPU multilevel vertex separators for the top levels of nested dissection (see gpu_nd.h).
//
// Compiled with --default-stream per-thread: the separator trials of a part run in parallel host
// threads (their CPU work, METIS on the coarsest graph and the band FM, then overlaps), and every
// thread's stream-0 work goes to its own stream.
#include "gpu_nd.h"

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <metis.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

static_assert(sizeof(idx_t) == sizeof(int32_t), "METIS must be built with 32-bit idx_t");

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}
#define CK(x) ck((x), #x)

[[noreturn]] void fail(const char* msg)
{
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

// Constants of the algorithm. None depends on the input: they are the usual multilevel settings.
// Development overrides (environment): REF_ND_COARSEST, REF_ND_RATING (0 = plain heavy edge),
// REF_ND_TRIALS (separators tried at depth d: max(1, trials >> d), the smallest is kept), REF_ND_SEED.
int envInt(const char* name, int def)
{
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : def;
}
// Defaults chosen by a 4-seed study: the rating avoids coarsening stalls that made some seeds' top
// separators 1.2-1.9x too large, and best-of-4/2 separators at depths 0/1 plus a larger coarsest graph
// for METIS keep the flops close to the best known ordering.
const int32_t kCoarsest = envInt("REF_ND_COARSEST", 65536);  // stop coarsening here; METIS takes over
const int kRating = envInt("REF_ND_RATING", 1);
const int kTrials = envInt("REF_ND_TRIALS", 4);
const uint32_t kSeed = (uint32_t)envInt("REF_ND_SEED", 0);  // varies all matching seeds, for variance studies
constexpr double kCoarsenStall = 0.85;   // ... or when a level shrinks the graph by less than 15% (METIS's value)
constexpr int kMatchRounds = 10;         // matching rounds per level (stops earlier when no more edges match)
constexpr int kBandWidth = 3;            // refinement band: vertices within 3 hops of the separator (Scotch's default)
// Balance: each side at most 0.5 + ufactor / 2000 of the total weight; METIS ND's default ufactor 200
// gives 0.6. REF_ND_UFACTOR changes it here and in ref_order's METIS calls (development knob).
const int kUfactor = envInt("REF_ND_UFACTOR", 200);
const double kPartMax = 0.5 + kUfactor / 2000.0;

// Device buffer on the stream-ordered pool, so that the many small allocations are cheap
template <class T>
struct DBuf
{
    T* p = nullptr;
    size_t n = 0;
    DBuf() = default;
    explicit DBuf(size_t count) { alloc(count); }
    ~DBuf() { release(); }
    DBuf(DBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DBuf& operator=(DBuf&& o) noexcept
    {
        if (this != &o) { release(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    void alloc(size_t count)
    {
        release();
        n = count;
        if (count) CK(cudaMallocAsync(reinterpret_cast<void**>(&p), count * sizeof(T), 0));
    }
    void release()
    {
        if (p) cudaFreeAsync(p, 0);
        p = nullptr;
        n = 0;
    }
    void toHost(T* h, size_t count, size_t offset = 0) const
    {
        if (count) CK(cudaMemcpy(h, p + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void fromHost(const T* h, size_t count)
    {
        if (count) CK(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    T get(size_t i) const
    {
        T v;
        toHost(&v, 1, i);
        return v;
    }
};

// Runs a CUB device algorithm: size query, temporary allocation, call
template <class F>
void cubRun(F f)
{
    size_t bytes = 0;
    CK(f(nullptr, bytes));
    DBuf<char> tmp(std::max<size_t>(bytes, 1));
    CK(f(tmp.p, bytes));
}

template <class T>
void exclusiveSum(const T* in, T* out, int64_t n)
{
    cubRun([&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, in, out, n); });
}

constexpr int kThreads = 256;
unsigned gridFor(int64_t n) { return (unsigned)std::max<int64_t>(1, (n + kThreads - 1) / kThreads); }

struct DGraph
{
    int32_t n = 0;
    int64_t m = 0, tvw = 0;
    DBuf<int64_t> xadj;
    DBuf<int32_t> adj, ew, vw, label;  // ew empty: unit edge weights; label only on dissection-level graphs
};

__device__ __forceinline__ uint32_t mix32(uint32_t a, uint32_t b, uint32_t seed)
{
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u + (seed << 6) + (seed >> 2));
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// ------------------------------------------------------------------ coarsening
// Every unmatched vertex picks its best unmatched neighbour. Edges are rated ew^2 / (w_u w_v) (the
// "expansion*2" rating of KaHIP: heavy edges between light vertices first, which keeps coarse vertex
// weights even) and totally ordered by (rating, hash, ends), so the best remaining edge is the mutual
// choice of both ends.
__global__ void kPropose(int32_t n, const int64_t* xadj, const int32_t* adj, const int32_t* ew, const int32_t* vw,
                         int32_t maxvw, const int32_t* mate, int32_t* cand, uint32_t seed, int rating)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (mate[v] >= 0) { cand[v] = -1; return; }
    int32_t best = -1, bestLo = 0, bestHi = 0;
    uint64_t bestKey = 0;
    const int32_t wv = vw[v];
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q) {
        int32_t u = adj[q];
        if (mate[u] >= 0 || wv + vw[u] > maxvw) continue;
        int32_t lo = min(u, v), hi = max(u, v);
        const float w = ew ? (float)ew[q] : 1.0f;
        const float r = rating ? w * w / ((float)wv * (float)vw[u]) : w;  // > 0, so its bits order like the value
        uint64_t key = ((uint64_t)__float_as_uint(r) << 32) | mix32(lo, hi, seed);
        if (best < 0 || key > bestKey || (key == bestKey && (lo > bestLo || (lo == bestLo && hi > bestHi)))) {
            best = u;
            bestKey = key;
            bestLo = lo;
            bestHi = hi;
        }
    }
    cand[v] = best;
}

__global__ void kAccept(int32_t n, const int32_t* cand, int32_t* mate, unsigned long long* count)
{
    __shared__ int c;
    if (threadIdx.x == 0) c = 0;
    __syncthreads();
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && mate[v] < 0) {
        int32_t u = cand[v];
        if (u >= 0 && cand[u] == v) { mate[v] = u; atomicAdd(&c, 1); }
    }
    __syncthreads();
    if (threadIdx.x == 0 && c) atomicAdd(count, (unsigned long long)c);
}

__global__ void kFinishMatch(int32_t n, int32_t* mate, int32_t* leader)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (mate[v] < 0) mate[v] = v;
    leader[v] = mate[v] >= v;  // singleton, or the smaller end of a pair
}

__global__ void kCmap(int32_t n, const int32_t* mate, const int32_t* cid, const int32_t* vw, int32_t* cmap,
                      int32_t* cvw)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t u = mate[v], lead = min(u, v);
    cmap[v] = cid[lead];
    if (lead == v) cvw[cid[v]] = vw[v] + (u != v ? vw[u] : 0);
}

__global__ void kEdgeKeys(int32_t n, const int64_t* xadj, const int32_t* adj, const int32_t* ew, const int32_t* cmap,
                          int bits, uint64_t sentinel, uint64_t* keys, int32_t* vals)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    uint64_t cu = (uint64_t)cmap[v];
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q) {
        uint64_t cw = (uint64_t)cmap[adj[q]];
        keys[q] = cu == cw ? sentinel : (cu << bits) | cw;
        vals[q] = ew ? ew[q] : 1;
    }
}

__global__ void kRowCount(int64_t m, const uint64_t* keys, int bits, unsigned long long* cnt)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) atomicAdd(&cnt[keys[i] >> bits], 1ull);
}

__global__ void kColumns(int64_t m, const uint64_t* keys, uint64_t mask, int32_t* adj)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) adj[i] = (int32_t)(keys[i] & mask);
}

// Contracts the matched pairs of g (mate[v] = partner or -1; released here) into c, with cmap mapping
// every vertex to its coarse vertex. Coarse vertices are numbered by their smaller member, vertex and
// edge weights are summed. Returns false (c untouched) if nothing was matched.
bool contract(const DGraph& g, DBuf<int32_t>& mate, DGraph& c, DBuf<int32_t>& cmap)
{
    const int32_t n = g.n;
    DBuf<int32_t> leader(n), cid(n);
    kFinishMatch<<<gridFor(n), kThreads>>>(n, mate.p, leader.p);
    exclusiveSum(leader.p, cid.p, n);
    const int32_t nc = cid.get(n - 1) + leader.get(n - 1);
    if (nc == n) return false;

    c.n = nc;
    c.tvw = g.tvw;
    cmap.alloc(n);
    c.vw.alloc(nc);
    kCmap<<<gridFor(n), kThreads>>>(n, mate.p, cid.p, g.vw.p, cmap.p, c.vw.p);
    mate.release(); leader.release(); cid.release();

    int bits = 1;
    while ((int64_t(1) << bits) < nc) ++bits;
    const uint64_t sentinel = (uint64_t(1) << (2 * bits)) - 1;  // above every real key, which has cu != cw
    DBuf<uint64_t> k0(g.m), k1(g.m);
    DBuf<int32_t> v0(g.m), v1(g.m);
    kEdgeKeys<<<gridFor(n), kThreads>>>(n, g.xadj.p, g.adj.p, g.ew.p, cmap.p, bits, sentinel, k0.p, v0.p);
    cub::DoubleBuffer<uint64_t> dk(k0.p, k1.p);
    cub::DoubleBuffer<int32_t> dv(v0.p, v1.p);
    cubRun([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dv, g.m, 0, 2 * bits); });
    // reduce into the spare buffers
    uint64_t* ukeys = dk.Current() == k0.p ? k1.p : k0.p;
    int32_t* uvals = dv.Current() == v0.p ? v1.p : v0.p;
    DBuf<int64_t> nruns(1);
    cubRun([&](void* t, size_t& b) {
        return cub::DeviceReduce::ReduceByKey(t, b, dk.Current(), ukeys, dv.Current(), uvals, nruns.p,
                                              ::cuda::std::plus<int32_t>{}, g.m);
    });
    int64_t mc = nruns.get(0);
    if (mc > 0) {
        uint64_t last;
        CK(cudaMemcpy(&last, ukeys + mc - 1, sizeof(last), cudaMemcpyDeviceToHost));
        if (last == sentinel) --mc;
    }
    c.m = mc;
    DBuf<unsigned long long> cnt(nc + 1);
    CK(cudaMemsetAsync(cnt.p, 0, (nc + 1) * sizeof(unsigned long long), 0));
    kRowCount<<<gridFor(mc), kThreads>>>(mc, ukeys, bits, cnt.p);
    c.xadj.alloc(nc + 1);
    exclusiveSum(reinterpret_cast<const int64_t*>(cnt.p), c.xadj.p, nc + 1);
    c.adj.alloc(mc);
    c.ew.alloc(mc);
    kColumns<<<gridFor(mc), kThreads>>>(mc, ukeys, (uint64_t(1) << bits) - 1, c.adj.p);
    CK(cudaMemcpy(c.ew.p, uvals, mc * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    return true;
}

// One coarsening level: heavy-edge matching, then contraction. Returns false if the graph did not shrink.
bool coarsen(const DGraph& g, int32_t maxvw, uint32_t seed, DGraph& c, DBuf<int32_t>& cmap)
{
    const int32_t n = g.n;
    DBuf<int32_t> mate(n), cand(n);
    CK(cudaMemsetAsync(mate.p, 0xff, n * sizeof(int32_t), 0));
    DBuf<unsigned long long> count(1);
    for (int r = 0; r < kMatchRounds; ++r) {
        CK(cudaMemsetAsync(count.p, 0, sizeof(unsigned long long), 0));
        kPropose<<<gridFor(n), kThreads>>>(n, g.xadj.p, g.adj.p, g.ew.p, g.vw.p, maxvw, mate.p, cand.p, seed + r, kRating);
        kAccept<<<gridFor(n), kThreads>>>(n, cand.p, mate.p, count.p);
        if (count.get(0) == 0) break;
    }
    cand.release();
    return contract(g, mate, c, cmap);
}

// ------------------------------------------------------------------ steps 1-3: graph, matching, compression
// Two directed entries per off-diagonal entry of the upper triangle, keyed (row << bits) | col; the
// diagonal only records log|a_ii| (its keys are the sentinel, sorted past every real key)
// the row of every CSR entry, one thread per row
__global__ void kRowOfEntry(int64_t n, const int32_t* rowPtr, int32_t* row)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; ++k) row[k] = (int32_t)i;
}

__global__ void kEntryKeys(int64_t nnz, const int32_t* row, const int32_t* col, const double2* val, int bits,
                           uint64_t sentinel, uint64_t* keys, float* loga, float* logdiag, unsigned long long* ndiag)
{
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    const int32_t i = row[k], j = col[k];
    const float la = (float)log(hypot(val[k].x, val[k].y));  // -inf for an explicit zero
    if (i == j) {
        logdiag[i] = la;
        keys[2 * k] = keys[2 * k + 1] = sentinel;
        atomicAdd(ndiag, 1ull);
        return;
    }
    keys[2 * k] = ((uint64_t)i << bits) | (uint64_t)j;
    keys[2 * k + 1] = ((uint64_t)j << bits) | (uint64_t)i;
    loga[2 * k] = loga[2 * k + 1] = la;
}

__global__ void kFill(int64_t n, float* a, float v)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__ void kFillInt(int64_t n, int32_t* a, int32_t v)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__ void kDuplicates(int64_t m, const uint64_t* keys, unsigned long long* dup)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 < m && keys[i] == keys[i + 1]) atomicAdd(dup, 1ull);
}

// gain of entry e = 2 log|a_ij| - log|a_ii| - log|a_jj|: +inf next to a missing diagonal, NaN (never
// positive) for an explicit zero next to one
__global__ void kGain(int64_t m, const uint64_t* keys, int bits, const float* loga, const float* logdiag, float* gain)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m) return;
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    gain[e] = 2 * loga[e] - logdiag[keys[e] >> bits] - logdiag[keys[e] & mask];
}

// Locally dominant matching on the edges of positive gain, edges totally ordered by (gain, min, max)
__global__ void kGainPropose(int32_t n, const int64_t* xadj, const int32_t* adj, const float* gain, const int32_t* mate,
                             int32_t* cand)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (mate[v] >= 0) { cand[v] = -1; return; }
    int32_t best = -1, bestLo = 0, bestHi = 0;
    float bg = 0;
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q) {
        int32_t u = adj[q];
        if (mate[u] >= 0 || !(gain[q] > 0)) continue;
        int32_t lo = min(u, v), hi = max(u, v);
        if (best < 0 || gain[q] > bg || (gain[q] == bg && (lo > bestLo || (lo == bestLo && hi > bestHi)))) {
            best = u;
            bg = gain[q];
            bestLo = lo;
            bestHi = hi;
        }
    }
    cand[v] = best;
}

// ------------------------------------------------------------------ separator refinement
__global__ void kProject(int32_t n, const int32_t* cmap, const int32_t* cwhere, int32_t* where)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) where[v] = cwhere[cmap[v]];
}

__global__ void kDistInit(int32_t n, const int32_t* where, int32_t* dist)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) dist[v] = where[v] == 2 ? 0 : INT_MAX;
}

__global__ void kDistStep(int32_t n, const int64_t* xadj, const int32_t* adj, int32_t d, int32_t* dist)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || dist[v] != d) return;
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q)
        if (dist[adj[q]] > d + 1) dist[adj[q]] = d + 1;  // every writer stores the same value
}

__global__ void kBandFlag(int32_t n, const int32_t* dist, int32_t* flag)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) flag[v] = dist[v] <= kBandWidth;
}

// band vertices, and the weight of each side outside the band
__global__ void kBandList(int32_t n, const int32_t* flag, const int32_t* lid, const int32_t* where, const int32_t* vw,
                          int32_t* blist, unsigned long long* outside)
{
    __shared__ unsigned long long w[2];
    if (threadIdx.x < 2) w[threadIdx.x] = 0;
    __syncthreads();
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        if (flag[v]) blist[lid[v]] = v;
        else atomicAdd(&w[where[v]], (unsigned long long)vw[v]);  // never 2 outside the band
    }
    __syncthreads();
    if (threadIdx.x < 2 && w[threadIdx.x]) atomicAdd(&outside[threadIdx.x], w[threadIdx.x]);
}

// Band rows: neighbours inside the band by local id, neighbours outside as anchor nb (side 0) or nb + 1
__global__ void kBandRows(int32_t nb, const int32_t* blist, const int64_t* xadj, const int32_t* adj,
                          const int32_t* where, const int32_t* flag, const int32_t* lid, const int32_t* bx,
                          int32_t* badj, int32_t* cnt)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int32_t v = blist[i], c = 0;
    bool a[2] = {false, false};
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q) {
        int32_t u = adj[q];
        if (flag[u]) { if (badj) badj[bx[i] + c] = lid[u]; ++c; }
        else a[where[u]] = true;
    }
    for (int s = 0; s < 2; ++s)
        if (a[s]) { if (badj) badj[bx[i] + c] = nb + s; ++c; }
    if (cnt) cnt[i] = c;
}

template <class T>
__global__ void kGather(int32_t n, const int32_t* idx, const T* src, T* dst)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

template <class T>
__global__ void kScatter(int32_t n, const int32_t* idx, const T* src, T* dst)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[idx[i]] = src[i];
}

// Node FM (METIS_NodeRefine) on the band around the separator, with one fixed anchor per side
void refineBand(const DGraph& g, DBuf<int32_t>& where)
{
    const int32_t n = g.n;
    DBuf<int32_t> dist(n), flag(n), lid(n);
    kDistInit<<<gridFor(n), kThreads>>>(n, where.p, dist.p);
    for (int d = 0; d < kBandWidth; ++d) kDistStep<<<gridFor(n), kThreads>>>(n, g.xadj.p, g.adj.p, d, dist.p);
    kBandFlag<<<gridFor(n), kThreads>>>(n, dist.p, flag.p);
    exclusiveSum(flag.p, lid.p, n);
    const int32_t nb = lid.get(n - 1) + flag.get(n - 1);
    if (nb == 0) return;  // empty separator (disconnected graph): nothing to refine
    DBuf<int32_t> blist(nb);
    DBuf<unsigned long long> outside(2);
    CK(cudaMemsetAsync(outside.p, 0, 2 * sizeof(unsigned long long), 0));
    kBandList<<<gridFor(n), kThreads>>>(n, flag.p, lid.p, where.p, g.vw.p, blist.p, outside.p);
    DBuf<int32_t> cnt(nb + 1), bx(nb + 1);
    CK(cudaMemsetAsync(cnt.p + nb, 0, sizeof(int32_t), 0));
    kBandRows<<<gridFor(nb), kThreads>>>(nb, blist.p, g.xadj.p, g.adj.p, where.p, flag.p, lid.p, nullptr, nullptr, cnt.p);
    exclusiveSum(cnt.p, bx.p, nb + 1);
    const int32_t be = bx.get(nb);
    DBuf<int32_t> badj(be), bvw(nb), bwhere(nb);
    kBandRows<<<gridFor(nb), kThreads>>>(nb, blist.p, g.xadj.p, g.adj.p, where.p, flag.p, lid.p, bx.p, badj.p, nullptr);
    kGather<<<gridFor(nb), kThreads>>>(nb, blist.p, g.vw.p, bvw.p);
    kGather<<<gridFor(nb), kThreads>>>(nb, blist.p, where.p, bwhere.p);

    // host band graph: nb band vertices, then the two anchors
    std::vector<int32_t> xadj(nb + 3), adj(be), vw(nb + 2), wh(nb + 2), hm(nb + 2, -1);
    unsigned long long out[2];
    bx.toHost(xadj.data(), nb + 1);
    badj.toHost(adj.data(), be);
    bvw.toHost(vw.data(), nb);
    bwhere.toHost(wh.data(), nb);
    outside.toHost(out, 2);
    std::vector<int32_t> anchorAdj[2];
    for (int32_t i = 0; i < nb; ++i)
        for (int32_t q = xadj[i]; q < xadj[i + 1]; ++q)
            if (adj[q] >= nb) anchorAdj[adj[q] - nb].push_back(i);
    for (int s = 0; s < 2; ++s) {
        adj.insert(adj.end(), anchorAdj[s].begin(), anchorAdj[s].end());
        xadj[nb + 1 + s] = (int32_t)adj.size();
        vw[nb + s] = (int32_t)out[s];
        wh[nb + s] = s;
        hm[nb + s] = s;  // an anchor may only stay on its own side
    }
    int64_t pw[3] = {0, 0, 0};
    for (int32_t i = 0; i < nb + 2; ++i) pw[wh[i]] += vw[i];
    // METIS_NodeRefine bounds the larger side by ubfactor * its current weight; convert kPartMax
    const double cap = kPartMax * (double)(pw[0] + pw[1] + pw[2]);
    const real_t ub = (real_t)std::max(1.0, cap / (double)std::max<int64_t>(1, std::max(pw[0], pw[1])));
    if (METIS_NodeRefine(nb + 2, xadj.data(), vw.data(), adj.data(), wh.data(), hm.data(), ub) != METIS_OK)
        fail("METIS_NodeRefine failed");
    bwhere.fromHost(wh.data(), nb);
    kScatter<<<gridFor(nb), kThreads>>>(nb, blist.p, bwhere.p, where.p);
}

// Vertex separator of the coarsest graph by METIS on the host
void initialSeparator(const DGraph& g, DBuf<int32_t>& where)
{
    std::vector<int64_t> x64(g.n + 1);
    g.xadj.toHost(x64.data(), g.n + 1);
    std::vector<idx_t> xadj(x64.begin(), x64.end()), adj(g.m), vw(g.n), part(g.n);
    g.adj.toHost(adj.data(), g.m);
    g.vw.toHost(vw.data(), g.n);
    idx_t nv = g.n, sepsize = 0, options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_UFACTOR] = kUfactor;
    if (METIS_ComputeVertexSeparator(&nv, xadj.data(), adj.data(), vw.data(), options, &sepsize, part.data()) !=
        METIS_OK)
        fail("METIS_ComputeVertexSeparator failed");
    where.alloc(g.n);
    where.fromHost(part.data(), g.n);
}

__global__ void kBadEdges(int32_t n, const int64_t* xadj, const int32_t* adj, const int32_t* where,
                          unsigned long long* bad)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || where[v] == 2) return;
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q)
        if (where[adj[q]] == 1 - where[v]) atomicAdd(bad, 1ull);
}

// Multilevel vertex separator of g: where[v] = 0, 1 or 2 (separator)
void separatorOnce(const DGraph& g, uint32_t seed, DBuf<int32_t>& where, GpuNdStats& st, std::vector<int32_t>& sizes)
{
    auto t = Clock::now();
    std::vector<std::unique_ptr<DGraph>> levels;
    std::vector<DBuf<int32_t>> cmaps;
    const DGraph* cur = &g;
    const int32_t maxvw = (int32_t)std::min<int64_t>(INT32_MAX, (int64_t)(1.5 * g.tvw / kCoarsest));
    while (cur->n > kCoarsest) {
        auto c = std::make_unique<DGraph>();
        DBuf<int32_t> cmap;
        if (!coarsen(*cur, maxvw, seed * 64 + (uint32_t)levels.size() * kMatchRounds, *c, cmap)) break;
        const bool stalled = c->n > kCoarsenStall * cur->n;
        levels.push_back(std::move(c));
        cmaps.push_back(std::move(cmap));
        cur = levels.back().get();
        if (stalled) break;
    }
    sizes.assign(1, g.n);
    for (const auto& l : levels) sizes.push_back(l->n);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    st.coarsen += since(t);

    t = Clock::now();
    initialSeparator(*cur, where);
    st.initial += since(t);

    t = Clock::now();
    for (int l = (int)levels.size() - 1; l >= 0; --l) {
        const DGraph& fine = l > 0 ? *levels[l - 1] : g;
        DBuf<int32_t> fw(fine.n);
        kProject<<<gridFor(fine.n), kThreads>>>(fine.n, cmaps[l].p, where.p, fw.p);
        where = std::move(fw);
        levels[l].reset();
        cmaps[l].release();
        refineBand(fine, where);
    }
    DBuf<unsigned long long> bad(1);
    CK(cudaMemsetAsync(bad.p, 0, sizeof(unsigned long long), 0));
    kBadEdges<<<gridFor(g.n), kThreads>>>(g.n, g.xadj.p, g.adj.p, where.p, bad.p);
    if (bad.get(0) != 0) fail("GPU separator is invalid (edges join the two sides)");
    st.refine += since(t);
}

__global__ void kSepWeight(int32_t n, const int32_t* where, const int32_t* vw, unsigned long long* w)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && where[v] == 2) atomicAdd(w, (unsigned long long)vw[v]);
}

// Best of several multilevel separators (different matching seeds) by separator weight. The trials run
// in parallel threads, as many at a time as the free device memory allows (a trial needs about 50 bytes
// per adjacency entry at its first coarsening level); the result does not depend on that number.
void separator(const DGraph& g, int depth, uint32_t seed, DBuf<int32_t>& where, GpuNdStats& st,
               std::vector<int32_t>& sizes)
{
    const int trials = std::max(1, kTrials >> depth);
    struct Trial { DBuf<int32_t> w; std::vector<int32_t> sz; GpuNdStats st; unsigned long long sep = ULLONG_MAX; };
    std::vector<Trial> tr(trials);
    auto run = [&](int t) {
        separatorOnce(g, seed + 7919u * t, tr[t].w, tr[t].st, tr[t].sz);
        DBuf<unsigned long long> sw(1);
        CK(cudaMemsetAsync(sw.p, 0, sizeof(unsigned long long), 0));
        kSepWeight<<<gridFor(g.n), kThreads>>>(g.n, tr[t].w.p, g.vw.p, sw.p);
        tr[t].sep = sw.get(0);
    };
    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));
    const double need = 50.0 * (double)g.m + 64.0 * g.n;
    const int conc = (int)std::max<double>(1, std::min<double>(trials, 0.8 * freeB / need));
    for (int t0 = 0; t0 < trials; t0 += conc) {
        std::vector<std::thread> th;
        for (int t = t0; t < std::min(trials, t0 + conc); ++t) th.emplace_back(run, t);
        for (auto& x : th) x.join();
    }
    int b = 0;
    for (int t = 1; t < trials; ++t)
        if (tr[t].sep < tr[b].sep) b = t;  // ties: the first trial, independent of the thread schedule
    where = std::move(tr[b].w);
    sizes = std::move(tr[b].sz);
    for (const Trial& x : tr) {  // thread time summed over the trials
        st.coarsen += x.st.coarsen;
        st.initial += x.st.initial;
        st.refine += x.st.refine;
    }
}

// ------------------------------------------------------------------ split
__global__ void kSideFlag(int32_t n, const int32_t* where, int32_t side, int32_t* flag)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) flag[v] = where[v] == side;
}

__global__ void kSubRows(int32_t n, const int64_t* xadj, const int32_t* adj, const int32_t* where, int32_t side,
                         const int32_t* nid, const int64_t* sx, int32_t* sadj, int64_t* cnt)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || where[v] != side) return;
    int64_t c = 0;
    for (int64_t q = xadj[v]; q < xadj[v + 1]; ++q)
        if (where[adj[q]] == side) {
            if (sadj) sadj[sx[nid[v]] + c] = nid[adj[q]];
            ++c;
        }
    if (cnt) cnt[nid[v]] = c;
}

template <class T>
__global__ void kCompact(int32_t n, const int32_t* flag, const int32_t* nid, const T* src, T* dst)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && flag[v]) dst[nid[v]] = src[v];
}

// Induced subgraph of one side (vertex weights and labels carried along)
void extractSide(const DGraph& g, const DBuf<int32_t>& where, int32_t side, DGraph& s)
{
    const int32_t n = g.n;
    DBuf<int32_t> flag(n), nid(n);
    kSideFlag<<<gridFor(n), kThreads>>>(n, where.p, side, flag.p);
    exclusiveSum(flag.p, nid.p, n);
    s.n = nid.get(n - 1) + flag.get(n - 1);
    s.xadj.alloc(s.n + 1);
    s.vw.alloc(s.n);
    s.label.alloc(s.n);
    if (s.n == 0) { s.m = 0; s.tvw = 0; CK(cudaMemsetAsync(s.xadj.p, 0, sizeof(int64_t), 0)); return; }
    DBuf<int64_t> cnt(s.n + 1);
    CK(cudaMemsetAsync(cnt.p + s.n, 0, sizeof(int64_t), 0));
    kSubRows<<<gridFor(n), kThreads>>>(n, g.xadj.p, g.adj.p, where.p, side, nid.p, nullptr, nullptr, cnt.p);
    exclusiveSum(cnt.p, s.xadj.p, s.n + 1);
    s.m = s.xadj.get(s.n);
    s.adj.alloc(s.m);
    kSubRows<<<gridFor(n), kThreads>>>(n, g.xadj.p, g.adj.p, where.p, side, nid.p, s.xadj.p, s.adj.p, nullptr);
    kCompact<<<gridFor(n), kThreads>>>(n, flag.p, nid.p, g.vw.p, s.vw.p);
    kCompact<<<gridFor(n), kThreads>>>(n, flag.p, nid.p, g.label.p, s.label.p);
    int64_t tvw = 0;
    {
        DBuf<int64_t> sum(1);
        cubRun([&](void* t, size_t& b) {
            return cub::DeviceReduce::Sum(t, b, s.vw.p, sum.p, (int64_t)s.n);
        });
        tvw = sum.get(0);
    }
    s.tvw = tvw;
}

std::vector<int32_t> separatorLabels(const DGraph& g, const DBuf<int32_t>& where)
{
    DBuf<int32_t> flag(g.n), out(g.n);
    kSideFlag<<<gridFor(g.n), kThreads>>>(g.n, where.p, 2, flag.p);
    DBuf<int64_t> nsel(1);
    cubRun([&](void* t, size_t& b) {
        return cub::DeviceSelect::Flagged(t, b, g.label.p, flag.p, out.p, nsel.p, (int64_t)g.n);
    });
    std::vector<int32_t> sep(nsel.get(0));
    out.toHost(sep.data(), sep.size());
    return sep;
}

Sub toHost(const DGraph& g)
{
    if (g.m > INT32_MAX) fail("subgraph too large for 32-bit METIS idx_t");
    Sub s;
    std::vector<int64_t> x64(g.n + 1);
    g.xadj.toHost(x64.data(), g.n + 1);
    s.xadj.assign(x64.begin(), x64.end());
    s.adj.resize(g.m);
    s.vwgt.resize(g.n);
    s.label.resize(g.n);
    g.adj.toHost(s.adj.data(), g.m);
    g.vw.toHost(s.vwgt.data(), g.n);
    g.label.toHost(s.label.data(), g.n);
    return s;
}

__global__ void kIota(int32_t n, int32_t* a)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) a[v] = v;
}

}  // namespace

struct GpuGraph
{
    DGraph g;
};

void GpuGraphDeleter::operator()(GpuGraph* g) const { delete g; }

GpuGraphPtr gpuCompressedGraph(int64_t n, int64_t nnz, const int32_t* rowPtr, const int32_t* col,
                               const std::complex<double>* val, bool matching, std::vector<int32_t>& hostMate,
                               GpuPrepStats& st)
{
    if (n > INT32_MAX) fail("too many vertices for 32-bit vertex ids");
    auto t = Clock::now();
    DBuf<int32_t> dptr(n + 1), drow(nnz), dcol(nnz);
    DBuf<double2> dval(nnz);
    dptr.fromHost(rowPtr, n + 1);
    dcol.fromHost(col, nnz);
    dval.fromHost(reinterpret_cast<const double2*>(val), nnz);
    st.upload = since(t);

    // 1. symmetric adjacency and edge gains
    t = Clock::now();
    int bits = 1;
    while ((int64_t(1) << bits) < n) ++bits;
    const uint64_t sentinel = (uint64_t(1) << (2 * bits)) - 1;
    const int64_t m2 = 2 * nnz;
    DBuf<uint64_t> k0(m2), k1(m2);
    DBuf<float> a0(m2), a1(m2), logdiag(n);
    DBuf<unsigned long long> counter(1);
    CK(cudaMemsetAsync(counter.p, 0, sizeof(unsigned long long), 0));
    kFill<<<gridFor(n), kThreads>>>(n, logdiag.p, -INFINITY);
    kRowOfEntry<<<gridFor(n), kThreads>>>(n, dptr.p, drow.p);
    kEntryKeys<<<gridFor(nnz), kThreads>>>(nnz, drow.p, dcol.p, dval.p, bits, sentinel, k0.p, a0.p, logdiag.p,
                                           counter.p);
    dptr.release(); drow.release(); dcol.release(); dval.release();
    const int64_t m = 2 * (nnz - (int64_t)counter.get(0));
    cub::DoubleBuffer<uint64_t> dk(k0.p, k1.p);
    cub::DoubleBuffer<float> da(a0.p, a1.p);
    cubRun([&](void* tmp, size_t& b) { return cub::DeviceRadixSort::SortPairs(tmp, b, dk, da, m2, 0, 2 * bits); });
    CK(cudaMemsetAsync(counter.p, 0, sizeof(unsigned long long), 0));
    kDuplicates<<<gridFor(m), kThreads>>>(m, dk.Current(), counter.p);
    if (counter.get(0) != 0) fail("duplicate entries in the matrix");
    DGraph g0;
    g0.n = (int32_t)n;
    g0.m = m;
    g0.tvw = n;
    DBuf<unsigned long long> cnt(n + 1);
    CK(cudaMemsetAsync(cnt.p, 0, (n + 1) * sizeof(unsigned long long), 0));
    kRowCount<<<gridFor(m), kThreads>>>(m, dk.Current(), bits, cnt.p);
    g0.xadj.alloc(n + 1);
    exclusiveSum(reinterpret_cast<const int64_t*>(cnt.p), g0.xadj.p, n + 1);
    cnt.release();
    g0.adj.alloc(m);
    kColumns<<<gridFor(m), kThreads>>>(m, dk.Current(), (uint64_t(1) << bits) - 1, g0.adj.p);
    DBuf<float> gain(m);
    kGain<<<gridFor(m), kThreads>>>(m, dk.Current(), bits, da.Current(), logdiag.p, gain.p);
    k0.release(); k1.release(); a0.release(); a1.release(); logdiag.release();
    g0.vw.alloc(n);
    kFillInt<<<gridFor(n), kThreads>>>(n, g0.vw.p, 1);
    CK(cudaDeviceSynchronize());
    st.graph = since(t);
    st.edges = m;

    // 2. matching on the positive gains
    t = Clock::now();
    DBuf<int32_t> mate(n), cand(n);
    CK(cudaMemsetAsync(mate.p, 0xff, n * sizeof(int32_t), 0));
    for (st.rounds = 0; matching; ++st.rounds) {
        CK(cudaMemsetAsync(counter.p, 0, sizeof(unsigned long long), 0));
        kGainPropose<<<gridFor(n), kThreads>>>((int32_t)n, g0.xadj.p, g0.adj.p, gain.p, mate.p, cand.p);
        kAccept<<<gridFor(n), kThreads>>>((int32_t)n, cand.p, mate.p, counter.p);
        const unsigned long long newly = counter.get(0);
        st.pairs += newly / 2;
        if (newly == 0) break;  // the total order guarantees progress until no positive edge is left
    }
    gain.release(); cand.release();
    hostMate.resize(n);
    mate.toHost(hostMate.data(), n);
    st.matching = since(t);

    // 3. contract the pairs
    t = Clock::now();
    auto top = GpuGraphPtr(new GpuGraph);
    DBuf<int32_t> cmap;
    if (!contract(g0, mate, top->g, cmap)) top->g = std::move(g0);
    DGraph& c = top->g;
    c.label.alloc(c.n);
    kIota<<<gridFor(c.n), kThreads>>>(c.n, c.label.p);
    CK(cudaDeviceSynchronize());
    st.compress = since(t);
    st.compressedN = c.n;
    st.compressedEdges = c.m;
    return top;
}

void gpuInit()
{
    CK(cudaFree(nullptr));
    cudaMemPool_t pool;  // keep freed blocks in the pool instead of returning them to the driver
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetDefaultMemPool(&pool, dev));
    uint64_t threshold = UINT64_MAX;
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

void gpuDissectTop(GpuGraphPtr top, int gpuDepth, int64_t minGpuVertices,
                   const std::function<void(int, int64_t, std::vector<int32_t>&&)>& sepOut,
                   const std::function<void(int, int64_t, Sub&&)>& handoff, GpuNdStats& st)
{
    // A few GPU worker threads take parts from a shared depth-first stack, so that sibling parts are
    // bisected concurrently (their CPU steps overlap). handoff() is called only from this thread, which
    // may create OpenMP tasks there. The seed of each bisection depends only on its place in the tree,
    // so the result does not depend on the schedule.
    constexpr int kWorkers = 4;
    auto t0 = Clock::now();
    struct Node { std::unique_ptr<DGraph> g; int depth; int64_t k; };
    struct Ready { int depth; int64_t k; Sub s; };
    std::vector<Node> stack;
    std::deque<Ready> ready;
    std::mutex mu;
    std::condition_variable cv;
    int busy = 0;
    bool finished = false;
    {
        auto g = std::make_unique<DGraph>(std::move(top->g));
        top.reset();
        stack.push_back({std::move(g), 0, 0});
    }
    auto worker = [&]() {
        for (;;) {
            Node node;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !stack.empty() || busy == 0; });
                if (stack.empty()) {  // nothing queued and nobody working: done
                    finished = true;
                    cv.notify_all();
                    return;
                }
                node = std::move(stack.back());
                stack.pop_back();
                ++busy;
            }
            DGraph& g = *node.g;
            if (g.n > 0 && (node.depth >= gpuDepth || g.n < minGpuVertices)) {
                auto t = Clock::now();
                Sub s = toHost(g);
                node.g.reset();
                const double dt = since(t);
                std::lock_guard<std::mutex> lk(mu);
                st.handoff += dt;
                st.handoffs++;
                ready.push_back({node.depth, node.k, std::move(s)});
            } else if (g.n > 0) {
                auto tb = Clock::now();
                DBuf<int32_t> where;
                std::vector<int32_t> sizes;
                GpuNdStats ls;
                const uint32_t seed = (uint32_t)(((uint64_t(1) << node.depth) + node.k) * 2654435761u) + 104729u * kSeed;
                separator(g, node.depth, seed, where, ls, sizes);
                auto t = Clock::now();
                sepOut(node.depth, node.k, separatorLabels(g, where));  // distinct block per part
                auto a = std::make_unique<DGraph>(), b = std::make_unique<DGraph>();
                extractSide(g, where, 0, *a);
                extractSide(g, where, 1, *b);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
                const double dt = since(t);
                where.release();
                const int32_t na = a->n, nb = b->n, n0 = g.n;
                node.g.reset();
                std::lock_guard<std::mutex> lk(mu);
                st.coarsen += ls.coarsen;
                st.initial += ls.initial;
                st.refine += ls.refine;
                st.split += dt;
                st.bisections++;
                if (node.depth == 0) {
                    st.topSeconds = since(tb);
                    st.topLeft = na;
                    st.topRight = nb;
                    st.topSep = n0 - na - nb;
                    st.topLevelN = sizes;
                }
                stack.push_back({std::move(b), node.depth + 1, 2 * node.k + 1});
                stack.push_back({std::move(a), node.depth + 1, 2 * node.k});  // part 0 first
            }
            std::lock_guard<std::mutex> lk(mu);
            --busy;
            cv.notify_all();
        }
    };
    std::vector<std::thread> workers;
    for (int w = 0; w < kWorkers; ++w) workers.emplace_back(worker);
    for (;;) {  // hand finished parts to the caller as they come
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !ready.empty() || finished; });
        while (!ready.empty()) {
            Ready r = std::move(ready.front());
            ready.pop_front();
            lk.unlock();
            handoff(r.depth, r.k, std::move(r.s));
            lk.lock();
        }
        if (finished) break;
    }
    for (auto& w : workers) w.join();
    CK(cudaDeviceSynchronize());
    st.total = since(t0);
}
