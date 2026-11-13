// Top levels of nested dissection on the GPU, for ref_order.
//
// Each bisection is multilevel: the graph is coarsened on the GPU (locally dominant heavy-edge
// matching + contraction by sort and reduce), the coarsest graph gets a vertex separator from METIS on
// the CPU, and on the way back up the separator is projected level by level and refined. Refinement is
// METIS's node FM (METIS_NodeRefine) on the CPU, applied only to a band: the vertices within a few hops
// of the separator, plus one fixed anchor vertex per side that stands for the rest of that side. The
// graph is then split on the GPU and the halves are dissected the same way, depth first.
#ifndef GPU_ND_H
#define GPU_ND_H

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// A graph in host memory in METIS form (idx_t is 32-bit); label = vertex id in the full graph
struct Sub
{
    std::vector<int32_t> xadj{0}, adj, vwgt, label;
    int32_t n() const { return (int32_t)label.size(); }
};

struct GpuNdStats
{
    int bisections = 0, handoffs = 0;
    double coarsen = 0, initial = 0, refine = 0, split = 0, handoff = 0, total = 0;
    double topSeconds = 0;               // first bisection
    int32_t topLeft = 0, topRight = 0, topSep = 0;
    std::vector<int32_t> topLevelN;      // vertices of each coarsening level of the first bisection
};

// Creates the CUDA context and sets up the memory pool (call once, before timing anything)
void gpuInit();

struct GpuGraph;  // a graph resident on the device
struct GpuGraphDeleter { void operator()(GpuGraph* g) const; };
using GpuGraphPtr = std::unique_ptr<GpuGraph, GpuGraphDeleter>;

struct GpuPrepStats
{
    double upload = 0, graph = 0, matching = 0, compress = 0;
    int rounds = 0;
    int64_t edges = 0, pairs = 0, compressedEdges = 0;
    int32_t compressedN = 0;
};

// Steps 1-3 of ref_order on the GPU, from the upper triangle of A in CSR (values used only for the matching
// gains): symmetric adjacency (sort of the (i, j) keys), pairing of the rows with |a_ij|^2 > |a_ii| |a_jj| by
// locally dominant matching on the gain, contraction of the pairs. Returns the compressed graph (vertex
// weights 1 or 2, numbered by their smaller row) on the device, and in mate[] the partner of every row or -1.
// With matching = false no rows are paired.
GpuGraphPtr gpuCompressedGraph(int64_t n, int64_t nnz, const int32_t* rowPtr, const int32_t* col,
                               const std::complex<double>* val, bool matching, std::vector<int32_t>& mate,
                               GpuPrepStats& stats);

// Dissects the graph on the GPU for depths < gpuDepth (root depth 0); vertex labels are 0..n-1. Every
// separator found goes to sepOut(depth, k, labels); every part that reaches gpuDepth, or has fewer than
// minGpuVertices vertices, is copied to the host and handed to handoff(depth, k, sub), which may
// process it asynchronously. Children of (depth, k) are (depth + 1, 2k) for part 0 and (depth + 1, 2k + 1)
// for part 1.
void gpuDissectTop(GpuGraphPtr top, int gpuDepth, int64_t minGpuVertices,
                   const std::function<void(int, int64_t, std::vector<int32_t>&&)>& sepOut,
                   const std::function<void(int, int64_t, Sub&&)>& handoff, GpuNdStats& stats);

#endif
