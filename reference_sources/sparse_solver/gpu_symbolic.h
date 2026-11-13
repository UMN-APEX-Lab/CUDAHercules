// Supernodal symbolic analysis of P A P^T on the GPU, for an ref_order plan (permutation + ND tree).
//
// The result is CHOLMOD's (cholmod_l_analyze_p with the plan as the given ordering, postorder on, default
// relaxed amalgamation), entry for entry: the same final permutation, supernodes and row structures, so
// either can feed the factorization. Steps:
//   1. strictly lower pattern of P A P^T as sorted (row, col) keys
//   2. elimination tree, ND level by level: the blocks of a level are independent (no edge joins two
//      blocks unless one is an ancestor of the other); one CTA per block runs Liu's algorithm over the
//      block's rows in shared memory, after the entries into finished subtrees have been resolved in
//      parallel to the subtree roots and their first rows in the block
//   3. a postorder (Euler tour + list ranking), and the fundamental supernodes from it: j joins its only
//      child unless j is a leaf of some row subtree, which shows as a gap in that row's sorted entries
//   4. column counts: per row, the supernodal row subtree is walked from every entry up to where the
//      previous entry's path joins it, counting rows per fundamental supernode
//   5. CHOLMOD's weighted postorder (children by increasing column count) as the final ordering
//   6. relaxed amalgamation: CHOLMOD's sequential pass over the fundamental supernodes, run in parallel
//      chunks that are corrected by the exact state at their top until no chunk changes
//   7. row structures: the same walks on the relaxed supernodal tree, writing the rows, sorted per
//      supernode
#ifndef GPU_SYMBOLIC_H
#define GPU_SYMBOLIC_H

#include <cstdint>
#include <vector>

struct SymbolicResult
{
    int64_t n = 0, nsuper = 0, nfsuper = 0;
    double lnz = 0;                // sum of the column counts
    std::vector<int64_t> perm;     // final ordering: perm[q] = original row of pivot q (CHOLMOD's Perm)
    std::vector<int64_t> colCount; // per pivot, diagonal included (CHOLMOD's ColCount); with hostLs only
    std::vector<int64_t> super;    // nsuper + 1: pivot columns of each supernode
    std::vector<int64_t> pi;       // nsuper + 1: offsets into Ls
    std::vector<int64_t> Ls;       // rows of each supernode (pivot numbering), ascending, own columns first
                                   // (left empty when not asked for)
    std::vector<int32_t> sparent;  // nsuper: parent supernode, -1 for a root
    int amalgPasses = 0;           // passes of the chunked amalgamation
    // seconds per step
    double tUpload = 0, tPattern = 0, tEtree = 0, tPost = 0, tCounts = 0, tAmalg = 0, tStruct = 0, tDownload = 0;
};

// The analysis on the device, kept for building the factorization's maps without a round trip
struct SymbolicDevice
{
    int64_t n = 0, nnz = 0, nsuper = 0;
    int32_t *rowPtr = nullptr, *col = nullptr;  // upper CSR of A (original numbering)
    int32_t* invPerm = nullptr;                 // original row -> pivot
    int32_t* super = nullptr;                   // nsuper + 1
    int64_t* pi = nullptr;                      // nsuper + 1
    int32_t* Ls = nullptr;                      // pi[nsuper]
    int32_t* sid = nullptr;                     // pivot -> supernode
    SymbolicDevice() = default;
    SymbolicDevice(const SymbolicDevice&) = delete;
    SymbolicDevice& operator=(const SymbolicDevice&) = delete;
    ~SymbolicDevice() { release(); }
    void release();
};

// rowPtr/col: upper CSR of A (original numbering, diagonal included, columns sorted); planPerm: gather
// permutation of the plan (planPerm[k] = original row at position k); tree: block sizes of the plan's ND
// tree in rows (ref_order's .rowtree: leaves first, then separators level by level, bottom up). The rows
// (out.Ls) and column counts are copied to the host only with hostLs; dev, if given, receives the device
// copies.
void gpuSymbolic(int64_t n, const int32_t* rowPtr, const int32_t* col, const int32_t* planPerm,
                 const std::vector<int32_t>& tree, SymbolicResult& out, SymbolicDevice* dev, bool hostLs, bool verbose);

// The same device copies from an analysis done on the host (CHOLMOD's arrays)
void uploadSymbolic(int64_t n, const int32_t* rowPtr, const int32_t* col, const int64_t* perm, int64_t nsuper,
                    const int64_t* super, const int64_t* pi, const int64_t* Ls, SymbolicDevice& dev);

// Maps of the multifrontal factorization over a region of supernodes (reg[s] = region index or -1, in
// increasing s), built on the device:
//   relind  for every region supernode whose parent is in the region, the positions of its rows below its
//           own columns in the parent's row list, at relOff[r] (the caller's offsets, relLen in total)
//   urow    the same rows' pivot numbers (for the backward solve)
//   aPos    every entry of P A P^T whose column lies in the region: its position in that supernode's
//           front (column-major, m rows), grouped by region supernode in CSR order (aBeg, R + 1 offsets)
//   aSrc    the CSR slot of that entry, so that each frequency's values are one gather
// The arrays are device allocations owned by the caller (cudaFree).
void gpuFrontMaps(const SymbolicDevice& d, const std::vector<int32_t>& reg, const std::vector<int64_t>& regSuper,
                  const std::vector<int32_t>& parentReg, const std::vector<int64_t>& relOff, int64_t relLen,
                  int32_t** relind, int32_t** urow, int32_t** aPos, int32_t** aSrc, std::vector<int64_t>& aBeg);

#endif
