#pragma once
/**
 * sparse_formats.cuh — Pre-built sparse format conversion utilities for SpMM.
 *
 * Usage: #include "sparse_formats.cuh" in kernel.cu
 *   - Call conversion functions in spmm_preprocess()
 *   - Use converted data in spmm_compute()
 *   - Call free_*() in spmm_cleanup()
 *
 * All input CSR arrays live on the GPU (device pointers from the harness).
 * All output arrays are allocated on the GPU by these functions.
 * Conversions run on the host (not timed), so they can be arbitrarily complex.
 */

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================
//  COO (Coordinate)
// ============================================================
//  Expands CSR row_ptr into per-element row indices.
//  Use for: atomics-based or element-parallel SpMM.
//  Memory: 12 bytes/nnz (same as CSR + row_ind).

struct COOData {
    int*   d_row_ind;   // [nnz]
    int*   d_col_ind;   // [nnz]  (copy)
    float* d_values;    // [nnz]  (copy)
    int    num_rows;
    int    num_cols;
    int    nnz;
};

COOData csr_to_coo(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz);
void free_coo(COOData& m);

// ============================================================
//  ELL (ELLPACK) — Column-major
// ============================================================
//  Pads every row to max_nnz_per_row.  Column-major layout enables
//  coalesced reads: data[pos * num_rows + row].
//  Padding: col = -1, val = 0.
//
//  Best when row lengths are uniform (std/avg < 0.5).
//  WARNING: memory = max_nnz_per_row × num_rows × 8 bytes.
//           Avoid when max >> avg (use SELL instead).

struct ELLData {
    int*    d_col_ind;        // [max_nnz_per_row * num_rows], column-major
    float*  d_values;         // [max_nnz_per_row * num_rows], column-major
    int     num_rows;
    int     num_cols;
    int     max_nnz_per_row;
    int     nnz;              // original nnz
    int64_t total_elements;   // max_nnz_per_row * num_rows
};

ELLData csr_to_ell(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz);
void free_ell(ELLData& m);

// ============================================================
//  SELL-C-σ  (Sliced ELLPACK with optional row sorting)
// ============================================================
//  Divides rows into slices of C rows.  Within each slice, rows can
//  be sorted by decreasing length (sigma = sort scope).
//  Each slice is independently padded to its own max row length.
//
//  Layout (column-major within each slice):
//    data[ slice_ptr[s] + j * C + local_row ]
//  where j = position in row (0..slice_width[s]-1),
//        local_row = 0..C-1
//
//  Parameters:
//    C     = slice height (32 recommended for warp-aligned access)
//    sigma = sort scope (0 = no sort, C = per-slice sort)
//
//  d_row_perm[i] = original row index for sorted position i.
//  Your compute kernel must use d_row_perm to write output to the
//  correct row of d_C:  d_C[ row_perm[sorted_row] * K + k ] = result

struct SELLData {
    int*    d_col_ind;     // packed, column-major within slices
    float*  d_values;      // packed, same layout
    int*    d_slice_ptr;   // [num_slices + 1], element offsets
    int*    d_slice_width; // [num_slices], max nnz/row per slice
    int*    d_row_perm;    // [num_rows_padded], sorted_pos → original_row
    int     num_rows;      // original num_rows
    int     num_cols;
    int     nnz;           // original nnz
    int     C;             // slice height
    int     sigma;
    int     num_slices;
    int     num_rows_padded; // num_slices * C (≥ num_rows)
    int64_t total_padded;  // total elements in packed arrays
};

SELLData csr_to_sell(const int* d_row_ptr, const int* d_col_ind,
                     const float* d_values,
                     int num_rows, int num_cols, int nnz,
                     int C = 32, int sigma = 0);
void free_sell(SELLData& m);

// ============================================================
//  BSR (Block Sparse Row) — Dense zero-padded blocks
// ============================================================
//  Groups CSR non-zeros into block_dim × block_dim dense blocks.
//  A block exists if it contains at least one non-zero.
//  Blocks stored row-major:
//    bsr_values[ block_idx * block_dim² + local_row * block_dim + local_col ]
//
//  Good for: matrices with natural block structure, Tensor Core
//            (block_dim=16 → 16×16 blocks for wmma).
//
//  fill_ratio = (nnzb * block_dim²) / original_nnz
//    ~1.0 = perfect block structure (every position filled)
//    >5   = blocks are mostly zeros — BSR adds overhead

struct BSRData {
    int*   d_bsr_row_ptr;  // [mb + 1]
    int*   d_bsr_col_ind;  // [nnzb]
    float* d_bsr_values;   // [nnzb * block_dim * block_dim], row-major per block
    int    num_rows;       // original matrix dimensions
    int    num_cols;
    int    mb;             // block-rows = ceil(num_rows / block_dim)
    int    nb;             // block-cols = ceil(num_cols / block_dim)
    int    nnzb;           // number of non-zero blocks
    int    block_dim;
    float  fill_ratio;     // padding overhead indicator
};

BSRData csr_to_bsr(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz,
                   int block_dim = 16);
void free_bsr(BSRData& m);

// ============================================================
//  Row Partitioning by nnz per row
// ============================================================
//  Splits rows into three groups based on nnz per row.
//  Use for: hybrid kernels — launch a different kernel per group.
//  Default thresholds: sparse < 32, medium [32, 256), dense ≥ 256.

struct RowBin {
    int*  d_row_ids;   // [count], device array of original row indices
    int   count;
    int   nnz_min;
    int   nnz_max;
    float nnz_avg;
};

struct RowPartitionData {
    RowBin sparse;         // nnz/row < threshold_low
    RowBin medium;         // threshold_low ≤ nnz/row < threshold_high
    RowBin dense;          // nnz/row ≥ threshold_high
    int    num_rows;
    int    threshold_low;
    int    threshold_high;
};

RowPartitionData partition_rows(const int* d_row_ptr, int num_rows,
                                int threshold_low  = 32,
                                int threshold_high = 256);
void free_row_partition(RowPartitionData& rp);

// ============================================================
//  Tiled2D — 16×16 sub-blocks with per-block format selection
//  (CB-SpMV style: cache-friendly packed blocks + load balance)
// ============================================================
//  The matrix is split into 16×16 tiles.  Each non-empty tile is one
//  unit of work, intended for ONE WARP.  Tiles are packed into a single
//  contiguous byte buffer so a warp's working set is one sequential
//  byte range (indices and values adjacent → high L1 hit rate).
//
//  Per-tile format by nnz:
//    TILED2D_COO   (nnz < 32):    uint8 coords[nnz]  — (col<<4)|row
//                                 [pad to 4B]  float vals[nnz]
//    TILED2D_CSR   (32–128):      uint8 rp[17] (local row_ptr, rp[16]=nnz)
//                                 uint8 cols[nnz] (local col 0..15)
//                                 [pad to 4B]  float vals[nnz]
//    TILED2D_DENSE (nnz > 128):   float vals[256] row-major, zero-filled,
//                                 no index data at all
//
//  Load balance (optional, default on): tiles are sorted by nnz
//  descending and greedily assigned to thread blocks of
//  TILED2D_WARPS_PER_TB warps via a min-heap on accumulated nnz.
//  Only the metadata arrays are permuted; the packed buffer is not.
//
//  Kernel skeleton:
//    slot = blockIdx.x * TILED2D_WARPS_PER_TB + (threadIdx.x / 32);
//    if (slot >= t.num_slots || d_blk_type[slot] == TILED2D_EMPTY) return;
//    const char* p = t.d_data + t.d_blk_vp[slot];
//    rows covered: [d_blk_row[slot]*16, +16), cols: [d_blk_col[slot]*16, +16)
//  Tiles in the same block-row land in different warps/TBs, so
//  accumulate into y (SpMV) or C (SpMM) with atomicAdd.
//
//  Column aggregation from the CB-SpMV paper is NOT implemented.

#define TILED2D_COO   0
#define TILED2D_CSR   1
#define TILED2D_DENSE 2
#define TILED2D_EMPTY 3

#define TILED2D_TILE        16
#define TILED2D_WARPS_PER_TB 8

struct Tiled2DData {
    int*     d_blk_row;    // [num_slots] tile row index (rows blk_row*16 ..)
    int*     d_blk_col;    // [num_slots] tile col index
    int*     d_blk_nnz;    // [num_slots]
    int*     d_blk_type;   // [num_slots] TILED2D_COO/CSR/DENSE/EMPTY
    int64_t* d_blk_vp;     // [num_slots] byte offset into d_data ("virtual ptr")
    char*    d_data;       // packed per-tile payload, 4B-aligned per tile
    int*     d_blk_row_ptr; // [num_blk_rows + 1] tile-level row pointer —
                            // ONLY when balance=false (tiles are then grouped
                            // by block-row): tiles of block-row I occupy slots
                            // [d_blk_row_ptr[I], d_blk_row_ptr[I+1]).
                            // nullptr when balance=true (slots are permuted).
                            // Use for block-wise products (SpGEMM): index
                            // tiles of any block-row directly.
    int      num_blk_rows; // ceil(num_rows / 16)
    int      num_blocks;   // non-empty tiles
    int      num_slots;    // num_tbs * TILED2D_WARPS_PER_TB (≥ num_blocks)
    int      num_tbs;      // thread blocks to launch
    int      num_rows;
    int      num_cols;
    int      nnz;
    int64_t  data_bytes;
    float    super_sparse_frac;  // fraction of tiles with nnz < 32
};

Tiled2DData csr_to_tiled2d(const int* d_row_ptr, const int* d_col_ind,
                           const float* d_values,
                           int num_rows, int num_cols, int nnz,
                           bool balance = true);
void free_tiled2d(Tiled2DData& m);

// ============================================================
//  Windowed — column-windowed packed stream (Sextans style)
// ============================================================
//  A is cut into a 2D grid of (row block R0 × column window K0)
//  segments.  Non-zeros are re-encoded with WINDOW-LOCAL indices and
//  packed into one 64-bit word each:
//
//    bits 63..50  col_local  (14 bits, col - w*K0,  requires K0 ≤ 16384)
//    bits 49..32  row_local  (18 bits, row - rb*R0, requires R0 ≤ 262144)
//    bits 31..0   value      (float bits)
//
//  Decode in-kernel:
//    int   col = (int)(p >> 50);
//    int   row = (int)((p >> 32) & 0x3FFFF);
//    float val = __int_as_float((int)(p & 0xFFFFFFFFu));
//
//  8 bytes/nnz instead of CSR's 12 → less A traffic, and every load is
//  a coalesced 64-bit read.  Segment (rb, w) occupies
//  d_packed[ d_seg_ptr[rb*num_windows + w] .. d_seg_ptr[...+1] ),
//  ordered row-major within the segment.
//
//  Intended dataflow (SpMM): one CTA owns a row block, keeps its C tile
//  resident (registers/smem), loops over windows; per window it stages
//  B[w*K0 .. w*K0+K0) × K_tile in shared memory (all random B reads hit
//  smem), and streams the segment's packed non-zeros once.
//  Pick K0 so the B window fits smem: K0 * K_tile * 4 bytes.

struct WindowedData {
    unsigned long long* d_packed;  // [nnz]
    int* d_seg_ptr;                // [num_row_blocks * num_windows + 1]
    int  num_row_blocks;           // ceil(num_rows / R0)
    int  num_windows;              // ceil(num_cols / K0)
    int  R0;                       // row block height
    int  K0;                       // column window width
    int  num_rows;
    int  num_cols;
    int  nnz;
};

WindowedData csr_to_windowed(const int* d_row_ptr, const int* d_col_ind,
                             const float* d_values,
                             int num_rows, int num_cols, int nnz,
                             int K0 = 4096, int R0 = 65536);
void free_windowed(WindowedData& m);

// ============================================================
//  NNZ-balanced chunks (merge-path style work partitioning)
// ============================================================
//  Splits the non-zero range into num_chunks equal pieces and records
//  the first row of each chunk.  Chunk c covers non-zeros
//  [c*S, min((c+1)*S, nnz)) with S = nnz_per_chunk, spanning rows
//  d_chunk_first_row[c] .. d_chunk_first_row[c+1] (boundary rows are
//  SHARED between adjacent chunks — combine partial sums with atomics
//  or a carry fix-up pass).
//  Use for: high row-length variance — work per warp/CTA is exactly S
//  non-zeros regardless of row structure.

struct NnzChunkData {
    int* d_chunk_first_row;  // [num_chunks + 1], last entry = num_rows
    int  num_chunks;
    int  nnz_per_chunk;      // S (last chunk may be smaller)
    int  num_rows;
    int  nnz;
};

NnzChunkData balance_nnz_chunks(const int* d_row_ptr, int num_rows, int nnz,
                                int num_chunks);
void free_nnz_chunks(NnzChunkData& m);

// ============================================================
//  Cyclic row interleaving (round-robin, MatRaptor style)
// ============================================================
//  Deals rows round-robin to P partitions and concatenates them:
//  partition p owns original rows p, p+P, p+2P, ...
//  Statistically evens out nnz across partitions for power-law /
//  banded matrices where contiguous row ranges are skewed.
//
//  d_perm[new_pos]     = original row
//  d_inv_perm[orig row] = new_pos
//  Process rows in new_pos order (warp/CTA w takes a contiguous range
//  of new_pos) but read/write y, C at the ORIGINAL row d_perm[new_pos]
//  so outputs stay in place and coalescing of C is preserved per row.

struct CyclicPermData {
    int* d_perm;      // [num_rows]
    int* d_inv_perm;  // [num_rows]
    int  P;
    int  num_rows;
};

CyclicPermData cyclic_row_perm(int num_rows, int P);
void free_cyclic_perm(CyclicPermData& m);
