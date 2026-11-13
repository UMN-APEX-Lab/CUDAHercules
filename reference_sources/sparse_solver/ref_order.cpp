// Nested-dissection ordering with a block tree, CPU version (stage 1 of the reordering; the GPU
// version replaces the separator step of the top levels).
//
// Everything is a graph algorithm on the pattern of A, plus one value-dependent step:
//   1. graph     symmetric adjacency of the upper-stored A, without the diagonal
//   2. matching  pair rows i, j whose 2x2 pivot beats two 1x1 pivots: |a_ij|^2 > |a_ii| |a_jj|, i.e.
//                edge gain log(|a_ij|^2 / (|a_ii| |a_jj|)) > 0. The gain is invariant under symmetric
//                diagonal scaling, so there is no threshold. Parallel locally dominant matching: an
//                edge is taken when it is the best remaining choice of both ends.
//   3. compress  each pair becomes one vertex of weight 2
//   4. dissect   recursive vertex separators (METIS_ComputeVertexSeparator) down to L levels, the two
//                halves of every split in parallel (OpenMP tasks); leaves ordered by METIS_NodeND.
//                With --gpu, the depths where there are fewer parts than threads (2^depth < threads,
//                or depths < D with --gpu-depth D) are dissected on the GPU instead (gpu_nd.cu): one
//                thread runs the GPU depth first and hands every finished part to the CPU threads,
//                which continue while the GPU works on the next part.
//   5. expand    vertices back to rows (pairs adjacent), tree sizes in rows
//
// Output, for ref_solve --user-perm P.perm --user-tree P.rowtree:
//   P.perm     int32, gather convention (perm[k] = original row at position k)
//   P.rowtree  2^L - 1 int32 block sizes in rows: the 2^(L-1) leaves, then the separators level by
//              level bottom up. Block k of a level has children 2k and 2k+1 one level down, and the
//              positions in P.perm follow this block order .
//   P.match    int32 involution: match[i] = row paired with i, or i
//
// --symbolic runs CHOLMOD symbolic analysis on the result (nnz(L), flops, largest front), and
// --eval-perm F does the same for another gather permutation, for comparison.
//
// METIS 5.1 takes its random numbers from the C library's rand(), whose state is global and behind a
// lock: concurrent METIS calls spent most of their time in the kernel on that lock, and their results
// depended on scheduling. rand()/srand() are therefore redefined below with per-thread state. They
// take precedence over libc's for libmetis, and since METIS calls srand() at the start of every API
// call, each call's result depends only on its input.
//
// usage: ref_order <dir|A.bin> [-o P] [--nd-levels L] [--threads N] [--no-matching]
//                 [--gpu] [--gpu-depth D] [--symbolic] [--eval-perm F]
#include "bench_common.h"
#include "gpu_nd.h"

#include <metis.h>
#include <omp.h>
#include <suitesparse/cholmod.h>

#include <climits>
#include <cstring>
#include <mutex>
#include <sys/stat.h>

// Per-thread rand()/srand() for METIS (see the top of the file); splitmix64, 31-bit output
static thread_local uint64_t tlsRandState = 1;
extern "C" int rand(void) noexcept
{
    uint64_t z = (tlsRandState += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (int)((z ^ (z >> 31)) >> 33);
}
extern "C" void srand(unsigned seed) noexcept { tlsRandState = seed; }

namespace {

struct Graph  // symmetric adjacency, no self loops
{
    int64_t n = 0;
    std::vector<int64_t> xadj;
    std::vector<int32_t> adj;
};

// ---------------------------------------------------------------- 1. graph + edge gains
// Builds the adjacency of A (upper CSR) with sorted rows, and the matching gain of every entry
void buildGraph(const Csr& A, Graph& g, std::vector<float>& gain)
{
    const int64_t n = A.n;
    std::vector<double> logDiag(n, -HUGE_VAL);  // log 0 = -inf: a missing diagonal always wants a pair
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k)
            if (A.col[k] == i) logDiag[i] = std::log(std::abs(A.val[k]));

    std::vector<int64_t> deg(n + 1, 0);
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k) {
            const int32_t j = A.col[k];
            if (j == i) continue;
            __atomic_fetch_add(&deg[i + 1], 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&deg[j + 1], 1, __ATOMIC_RELAXED);
        }
    for (int64_t i = 0; i < n; ++i) deg[i + 1] += deg[i];
    g.n = n;
    g.xadj = deg;
    struct Entry { int32_t nb; float gain; };
    std::vector<Entry> e(g.xadj[n]);
    std::vector<int64_t>& next = deg;  // reuse as fill pointers
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k) {
            const int32_t j = A.col[k];
            if (j == i) continue;
            // NaN (0 entry next to a 0 diagonal) compares false with 0 and is never matched
            float gk = (float)(2 * std::log(std::abs(A.val[k])) - logDiag[i] - logDiag[j]);
            e[__atomic_fetch_add(&next[i], 1, __ATOMIC_RELAXED)] = {j, gk};
            e[__atomic_fetch_add(&next[j], 1, __ATOMIC_RELAXED)] = {(int32_t)i, gk};
        }
    g.adj.resize(e.size());
    gain.resize(e.size());
#pragma omp parallel for schedule(dynamic, 4096)
    for (int64_t i = 0; i < n; ++i) {
        std::sort(e.begin() + g.xadj[i], e.begin() + g.xadj[i + 1],
                  [](const Entry& a, const Entry& b) { return a.nb < b.nb; });
        for (int64_t q = g.xadj[i]; q < g.xadj[i + 1]; ++q) { g.adj[q] = e[q].nb; gain[q] = e[q].gain; }
    }
}

// ---------------------------------------------------------------- 2. matching
// Locally dominant matching on the edges of positive gain. Edges are totally ordered by
// (gain, min end, max end), so the best remaining edge is always the mutual choice of its ends.
std::vector<int32_t> matchPairs(const Graph& g, const std::vector<float>& gain, int& rounds)
{
    const int64_t n = g.n;
    std::vector<int32_t> mate(n, -1), cand(n, -1), active;
    for (int64_t v = 0; v < n; ++v)
        for (int64_t q = g.xadj[v]; q < g.xadj[v + 1]; ++q)
            if (gain[q] > 0) { active.push_back((int32_t)v); break; }
    auto better = [](float g1, int32_t v, int32_t u1, float g2, int32_t u2) {
        if (g1 != g2) return g1 > g2;
        return std::make_pair(std::min(v, u1), std::max(v, u1)) > std::make_pair(std::min(v, u2), std::max(v, u2));
    };
    for (rounds = 0; !active.empty(); ++rounds) {
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t a = 0; a < active.size(); ++a) {
            int32_t v = active[a], best = -1;
            float bg = 0;
            for (int64_t q = g.xadj[v]; q < g.xadj[v + 1]; ++q) {
                int32_t u = g.adj[q];
                if (mate[u] >= 0 || !(gain[q] > 0)) continue;
                if (best < 0 || better(gain[q], v, u, bg, best)) { best = u; bg = gain[q]; }
            }
            cand[v] = best;
        }
        int64_t matched = 0;
#pragma omp parallel for reduction(+ : matched)
        for (size_t a = 0; a < active.size(); ++a) {
            int32_t v = active[a], u = cand[v];
            if (u >= 0 && cand[u] == v) { mate[v] = u; matched++; }
        }
        std::vector<int32_t> still;
        for (int32_t v : active)
            if (mate[v] < 0 && cand[v] >= 0) still.push_back(v);
        active.swap(still);
        if (matched == 0) break;  // cannot happen with a total order, kept as a guard
    }
    return mate;
}

// ---------------------------------------------------------------- 3. compress
// Contracts every pair to one vertex. cmap: row -> vertex; first/second: rows of each vertex.
void compress(const Graph& g, const std::vector<int32_t>& mate, Graph& c, std::vector<int32_t>& cmap,
              std::vector<int32_t>& first, std::vector<int32_t>& second)
{
    const int64_t n = g.n;
    cmap.resize(n);
    first.clear();
    second.clear();
    for (int64_t v = 0; v < n; ++v)
        if (mate[v] < 0 || v < mate[v]) {
            cmap[v] = (int32_t)first.size();
            if (mate[v] >= 0) cmap[mate[v]] = (int32_t)first.size();
            first.push_back((int32_t)v);
            second.push_back(mate[v]);
        }
    const int64_t nc = first.size();
    c.n = nc;
    c.xadj.assign(nc + 1, 0);
    // two passes: count, then fill; each vertex merges the mapped neighbours of its rows
    for (int pass = 0; pass < 2; ++pass) {
#pragma omp parallel
        {
            std::vector<int32_t> buf;
#pragma omp for schedule(dynamic, 4096)
            for (int64_t s = 0; s < nc; ++s) {
                buf.clear();
                for (int32_t v : {first[s], second[s]}) {
                    if (v < 0) continue;
                    for (int64_t q = g.xadj[v]; q < g.xadj[v + 1]; ++q)
                        if (cmap[g.adj[q]] != s) buf.push_back(cmap[g.adj[q]]);
                }
                std::sort(buf.begin(), buf.end());
                buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
                if (pass == 0) c.xadj[s + 1] = (int64_t)buf.size();
                else std::copy(buf.begin(), buf.end(), c.adj.begin() + c.xadj[s]);
            }
        }
        if (pass == 0) {
            for (int64_t s = 0; s < nc; ++s) c.xadj[s + 1] += c.xadj[s];
            c.adj.resize(c.xadj[nc]);
        }
    }
}

// ---------------------------------------------------------------- 4. dissect
// Sub (gpu_nd.h): a subgraph in METIS form; label = vertex id in the compressed graph
static_assert(sizeof(idx_t) == sizeof(int32_t), "METIS must be built with 32-bit idx_t");

// METIS balance (ufactor), METIS ND's default 200 unless REF_ND_UFACTOR overrides it (development knob)
inline int ufactor()
{
    static const int u = [] {
        const char* v = std::getenv("REF_ND_UFACTOR");
        return v && *v ? std::atoi(v) : 200;
    }();
    return u;
}

class Dissector
{
public:
    explicit Dissector(int levels) : L_(levels), blocks_(((size_t)1 << levels) - 1), depthStats_(levels) {}

    // array position of block k at depth d (root depth 0): leaves first, then bottom up
    size_t pos(int depth, int64_t k) const { return ((size_t)1 << L_) - ((size_t)1 << (depth + 1)) + k; }

    void run(Sub&& top)
    {
#pragma omp parallel
#pragma omp single
        dissect(std::move(top), 0, 0);
    }

    std::vector<std::vector<int32_t>>& blocks() { return blocks_; }
    void setBlock(int depth, int64_t k, std::vector<int32_t>&& v) { blocks_[pos(depth, k)] = std::move(v); }
    double topSeconds = 0;
    idx_t topSep = 0, topLeft = 0, topRight = 0;

    void printDepthStats() const
    {
        for (int d = 0; d < L_; ++d) {
            const DepthStats& s = depthStats_[d];
            if (!s.calls) continue;
            std::printf("    depth %2d %-9s: %4d parts on the CPU, %7.2f s total, %6.2f s max, %lld vertices\n", d,
                        d == L_ - 1 ? "(leaves)" : "", s.calls, s.sum, s.max, (long long)s.vertices);
        }
    }

    // Dissects part k at the given depth and everything below it (spawns OpenMP tasks)
    void dissect(Sub&& g, int depth, int64_t k)
    {
        if (g.n() == 0) return;
        if (depth == L_ - 1) {
            auto t = Clock::now();
            orderLeaf(g, blocks_[pos(depth, k)]);
            record(depth, since(t), g.n());
            return;
        }
        if (g.n() < kMinSplit) { dissect(std::move(g), depth + 1, 2 * k); return; }

        auto t = Clock::now();
        idx_t nv = g.n(), sepsize = 0, options[METIS_NOPTIONS];
        std::vector<idx_t> part(nv);
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_UFACTOR] = ufactor();
        if (METIS_ComputeVertexSeparator(&nv, g.xadj.data(), g.adj.data(), g.vwgt.data(), options, &sepsize,
                                         part.data()) != METIS_OK)
            die("METIS_ComputeVertexSeparator failed");
        record(depth, since(t), nv);
        std::vector<int32_t>& sep = blocks_[pos(depth, k)];
        for (idx_t v = 0; v < nv; ++v)
            if (part[v] == 2) sep.push_back(g.label[v]);
        Sub* half[2] = {new Sub, new Sub};
        split(g, part, *half[0], *half[1]);
        if (depth == 0) {
            topSeconds = since(t);
            topSep = (idx_t)sep.size();
            topLeft = half[0]->n();
            topRight = half[1]->n();
        }
        g = Sub();  // release before recursing
        for (int side = 0; side < 2; ++side) {
            Sub* h = half[side];
#pragma omp task firstprivate(h, side)
            {
                dissect(std::move(*h), depth + 1, 2 * k + side);
                delete h;
            }
        }
    }

    // Induced subgraphs of parts 0 and 1 (separator vertices dropped), keeping relative order
    static void split(const Sub& g, const std::vector<idx_t>& part, Sub& a, Sub& b)
    {
        std::vector<idx_t> lid(g.n());
        Sub* out[2] = {&a, &b};
        for (idx_t v = 0; v < g.n(); ++v)
            if (part[v] < 2) {
                Sub& s = *out[part[v]];
                lid[v] = s.n();
                s.label.push_back(g.label[v]);
                s.vwgt.push_back(g.vwgt[v]);
            }
        for (idx_t v = 0; v < g.n(); ++v) {
            if (part[v] == 2) continue;
            Sub& s = *out[part[v]];
            for (idx_t q = g.xadj[v]; q < g.xadj[v + 1]; ++q)
                if (part[g.adj[q]] == part[v]) s.adj.push_back(lid[g.adj[q]]);
            s.xadj.push_back((idx_t)s.adj.size());
        }
    }

    static void orderLeaf(Sub& g, std::vector<int32_t>& out)
    {
        idx_t nv = g.n();
        if (g.adj.empty()) { out = g.label; return; }
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_UFACTOR] = ufactor();
        std::vector<idx_t> perm(nv), iperm(nv);
        if (METIS_NodeND(&nv, g.xadj.data(), g.adj.data(), g.vwgt.data(), options, perm.data(), iperm.data()) !=
            METIS_OK)
            die("METIS_NodeND failed");
        out.resize(nv);
        for (idx_t i = 0; i < nv; ++i) out[i] = g.label[perm[i]];  // METIS perm: gather
    }

private:
    // Parts too small to split are passed whole down the left branch (empty sibling and separator)
    static constexpr idx_t kMinSplit = 64;

    struct DepthStats { int calls = 0; double sum = 0, max = 0; int64_t vertices = 0; };
    void record(int depth, double sec, idx_t n)
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        DepthStats& s = depthStats_[depth];
        s.calls++;
        s.sum += sec;
        s.max = std::max(s.max, sec);
        s.vertices += n;
    }

    int L_;
    std::vector<std::vector<int32_t>> blocks_;
    std::vector<DepthStats> depthStats_;
    std::mutex statsMutex_;
};

// ---------------------------------------------------------------- evaluation (CHOLMOD symbolic)
void evaluate(const char* name, const Csr& A, const std::vector<int32_t>& perm32)
{
    const int64_t n = A.n, nnz = A.nnz;
    cholmod_common c;
    cholmod_l_start(&c);
    // upper CSR == lower CSC: wrap as a lower-stored symmetric pattern
    cholmod_sparse* S = cholmod_l_allocate_sparse(n, n, nnz, 1, 1, -1, CHOLMOD_PATTERN, &c);
    std::copy(A.rowPtr.begin(), A.rowPtr.end(), static_cast<int64_t*>(S->p));
    std::copy(A.col.begin(), A.col.end(), static_cast<int64_t*>(S->i));

    std::vector<int64_t> perm(perm32.begin(), perm32.end());
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_GIVEN;
    c.postorder = 1;
    c.supernodal = CHOLMOD_SUPERNODAL;
    auto t = Clock::now();
    cholmod_factor* L = cholmod_l_analyze_p(S, perm.data(), nullptr, 0, &c);
    if (!L || c.status < CHOLMOD_OK) die("cholmod_analyze_p failed");
    const int64_t* super = static_cast<const int64_t*>(L->super);
    const int64_t* pi = static_cast<const int64_t*>(L->pi);
    int64_t maxFront = 0;
    for (int64_t s = 0; s < (int64_t)L->nsuper; ++s) maxFront = std::max(maxFront, pi[s + 1] - pi[s]);
    std::printf("  symbolic [%s]: nnz(L) %.4g, LL' flops %.4g, largest front %lld, root supernode %lld cols "
                "(CHOLMOD %.1f s)\n",
                name, c.lnz, c.fl, (long long)maxFront, (long long)(super[L->nsuper] - super[L->nsuper - 1]),
                since(t));
    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&S, &c);
    cholmod_l_finish(&c);
}

template <class T>
void writeVec(const std::string& path, const std::vector<T>& v)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) die(("cannot write " + path).c_str());
    std::fclose(f);
}

// Parts smaller than this go to the CPU even at GPU depths: METIS is then faster than a GPU bisection
constexpr int64_t kMinGpuVertices = 1 << 17;

std::vector<int32_t> readPerm(const std::string& path, int64_t n)
{
    std::vector<int32_t> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(v.data(), 4, n, f) != (size_t)n) die(("cannot read " + path).c_str());
    std::fclose(f);
    return v;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string in, out, evalPerm;
    int levels = 0, threads = omp_get_max_threads(), gpuDepth = -1;  // levels 0: from n, see below
    bool matching = true, symbolic = false, gpu = false;
    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "-o") && a + 1 < argc) out = argv[++a];
        else if (!std::strcmp(argv[a], "--nd-levels") && a + 1 < argc) levels = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc) threads = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--no-matching")) matching = false;
        else if (!std::strcmp(argv[a], "--gpu")) gpu = true;
        else if (!std::strcmp(argv[a], "--gpu-depth") && a + 1 < argc) gpu = true, gpuDepth = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--symbolic")) symbolic = true;
        else if (!std::strcmp(argv[a], "--eval-perm") && a + 1 < argc) evalPerm = argv[++a];
        else if (argv[a][0] != '-' && in.empty()) in = argv[a];
        else in.clear(), a = argc;
    }
    if (in.empty() || levels < 0 || levels > 20) {
        std::fprintf(stderr, "usage: %s <dir|A.bin> [-o P] [--nd-levels L] [--threads N] [--no-matching] "
                     "[--gpu] [--gpu-depth D] [--symbolic] [--eval-perm F]\n", argv[0]);
        return 1;
    }
    struct stat st;
    if (stat(in.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) in += "/A.bin";
    omp_set_num_threads(threads);
    if (!gpu) gpuDepth = 0;
    else if (gpuDepth < 0)  // depths with fewer parts than threads
        for (gpuDepth = 0; ((int64_t)1 << gpuDepth) < threads; ++gpuDepth) {}

    if (gpuDepth > 0) {
        auto t = Clock::now();
        gpuInit();
        std::printf("  CUDA context             : %8.2f s  (not part of the ordering time)\n", since(t));
    }
    auto t0 = Clock::now();
    Csr m;
    int sym = 0;
    if (!readMatrixACsr(in, m, sym)) die("read A failed");
    const int64_t n = m.n;
    if (levels == 0) {
        // Deeper trees factorize and solve faster at the same fill (2^(L-1) leaves of about 2000 rows on
        // average) but cost device memory for the per-node workspaces, hence the cap.
        levels = 1 + (int)std::lround(std::log2(std::max<double>(1.0, n / 2000.0)));
        levels = std::min(16, std::max(10, levels));
    }
    std::printf("file: %s\nn = %lld, nnz(upper) = %lld, %d threads, %d levels, GPU separators for depth < %d\n",
                in.c_str(), (long long)n, (long long)m.nnz, threads, levels, gpuDepth);
    std::printf("  read (CRC-checked)       : %8.2f s  (not part of the ordering time)\n", since(t0));

    auto tAll = Clock::now(), t = tAll;
    std::vector<int32_t> mate, first, second;  // first/second: rows of each compressed vertex (second -1 if single)
    Dissector nd(levels);
    if (gpuDepth > 0) {
        GpuPrepStats ps;
        GpuGraphPtr top = gpuCompressedGraph(n, m.nnz, m.rowPtr.data(), m.col.data(), m.val.data(),
                                             matching, mate, ps);
        std::printf("  1 graph (GPU)            : %8.2f s  (%lld adjacency entries; upload %.2f s)\n", ps.upload + ps.graph,
                    (long long)ps.edges, ps.upload);
        std::printf("  2 matching (GPU)         : %8.2f s  (%lld pairs, %d rounds)\n", ps.matching, (long long)ps.pairs,
                    ps.rounds);
        std::printf("  3 compress (GPU)         : %8.2f s  (%d vertices, %lld adjacency entries)\n", ps.compress,
                    (int)ps.compressedN, (long long)ps.compressedEdges);
        for (int64_t v = 0; v < n; ++v)
            if (mate[v] < 0 || v < mate[v]) {
                first.push_back((int32_t)v);
                second.push_back(mate[v]);
            }

        t = Clock::now();
        GpuNdStats gs;
#pragma omp parallel
#pragma omp single
        gpuDissectTop(
            std::move(top), gpuDepth, kMinGpuVertices,
            [&](int d, int64_t k, std::vector<int32_t>&& sep) { nd.setBlock(d, k, std::move(sep)); },
            [&](int d, int64_t k, Sub&& s) {
                Sub* p = new Sub(std::move(s));
                Dissector* ndp = &nd;
#pragma omp task firstprivate(p, d, k, ndp)
                {
                    ndp->dissect(std::move(*p), d, k);
                    delete p;
                }
            },
            gs);
        std::printf("  4 dissect                : %8.2f s  (GPU part %.2f s: %d bisections, %d parts handed to the CPU)\n",
                    since(t), gs.total, gs.bisections, gs.handoffs);
        std::printf("    GPU                    : coarsen %.2f s, initial %.2f s, refine %.2f s, split %.2f s, "
                    "handoff %.2f s\n", gs.coarsen, gs.initial, gs.refine, gs.split, gs.handoff);
        std::printf("    top separator on GPU   : %.2f s, %d | %d | sep %d vertices, levels",
                    gs.topSeconds, (int)gs.topLeft, (int)gs.topRight, (int)gs.topSep);
        for (int32_t lv : gs.topLevelN) std::printf(" %d", (int)lv);
        std::printf("\n");
    } else {
        Graph g;
        std::vector<float> gain;
        buildGraph(m, g, gain);
        std::printf("  1 graph                  : %8.2f s  (%lld adjacency entries)\n", since(t), (long long)g.adj.size());

        t = Clock::now();
        mate.assign(n, -1);
        int rounds = 0;
        if (matching) mate = matchPairs(g, gain, rounds);
        gain = std::vector<float>();
        int64_t pairs = 0;
        for (int64_t v = 0; v < n; ++v) pairs += mate[v] > v;
        std::printf("  2 matching               : %8.2f s  (%lld pairs, %d rounds)\n", since(t), (long long)pairs, rounds);

        t = Clock::now();
        Graph c;
        std::vector<int32_t> cmap;
        compress(g, mate, c, cmap, first, second);
        g = Graph();
        std::printf("  3 compress               : %8.2f s  (%lld vertices, %lld adjacency entries)\n", since(t),
                    (long long)c.n, (long long)c.adj.size());
        if ((int64_t)c.adj.size() > INT32_MAX) die("compressed graph too large for 32-bit METIS idx_t");

        t = Clock::now();
        Sub top;
        top.xadj.assign(c.xadj.begin(), c.xadj.end());
        top.adj.assign(c.adj.begin(), c.adj.end());
        top.vwgt.resize(c.n);
        top.label.resize(c.n);
        for (int64_t v = 0; v < c.n; ++v) {
            top.vwgt[v] = second[v] >= 0 ? 2 : 1;
            top.label[v] = (int32_t)v;
        }
        c = Graph();
        nd.run(std::move(top));
        std::printf("  4 dissect                : %8.2f s  (top separator %.2f s: %d | %d | sep %d vertices)\n",
                    since(t), nd.topSeconds, (int)nd.topLeft, (int)nd.topRight, (int)nd.topSep);
    }
    nd.printDepthStats();

    t = Clock::now();
    std::vector<int32_t> perm;
    perm.reserve(n);
    const auto& blocks = nd.blocks();
    std::vector<int32_t> tree(blocks.size());
    for (size_t b = 0; b < blocks.size(); ++b) {
        size_t start = perm.size();
        for (int32_t s : blocks[b]) {
            perm.push_back(first[s]);
            if (second[s] >= 0) perm.push_back(second[s]);
        }
        tree[b] = (int32_t)(perm.size() - start);
    }
    if ((int64_t)perm.size() != n) die("ordering lost rows");
    std::printf("  5 expand                 : %8.2f s\n", since(t));
    std::printf("  ordering total           : %8.2f s\n", since(tAll));
    {
        const size_t nleaves = (blocks.size() + 1) / 2;
        int64_t lmin = INT64_MAX, lmax = 0, sepRows = 0;
        for (size_t b = 0; b < blocks.size(); ++b) {
            if (b < nleaves) { lmin = std::min<int64_t>(lmin, tree[b]); lmax = std::max<int64_t>(lmax, tree[b]); }
            else sepRows += tree[b];
        }
        std::printf("  tree                     : %zu leaves %lld..%lld rows, separators %lld rows, top separator %d rows\n",
                    nleaves, (long long)lmin, (long long)lmax, (long long)sepRows, tree.back());
    }

    if (!out.empty()) {
        std::vector<int32_t> match(n);
        for (int64_t v = 0; v < n; ++v) match[v] = mate[v] >= 0 ? mate[v] : (int32_t)v;
        writeVec(out + ".perm", perm);
        writeVec(out + ".rowtree", tree);
        writeVec(out + ".match", match);
        std::printf("  wrote %s.perm, %s.rowtree, %s.match\n", out.c_str(), out.c_str(), out.c_str());
    }
    if (symbolic) evaluate("this ordering", m, perm);
    if (!evalPerm.empty()) evaluate(evalPerm.c_str(), m, readPerm(evalPerm, n));
    return 0;
}
