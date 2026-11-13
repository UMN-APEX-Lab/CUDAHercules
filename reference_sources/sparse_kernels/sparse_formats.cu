/**
 * sparse_formats.cu — Format conversion implementations.
 *
 * All conversions copy CSR from device to host, convert on host,
 * then upload the result to device.  This is fine because
 * spmm_preprocess() is NOT timed.
 */

#include "sparse_formats.cuh"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cmath>


// ================================================================
//  Internal helpers
// ================================================================

static void copy_csr_to_host(const int* d_row_ptr, const int* d_col_ind,
                              const float* d_values,
                              int num_rows, int nnz,
                              std::vector<int>& h_rp,
                              std::vector<int>& h_ci,
                              std::vector<float>& h_v)
{
    h_rp.resize(num_rows + 1);
    h_ci.resize(nnz);
    h_v.resize(nnz);
    cudaMemcpy(h_rp.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (nnz > 0) {
        cudaMemcpy(h_ci.data(), d_col_ind, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v.data(),  d_values,  nnz * sizeof(float), cudaMemcpyDeviceToHost);
    }
}


// ================================================================
//  COO
// ================================================================

// Simple GPU kernel: expand row_ptr → per-element row_ind
__global__ void _coo_expand_kernel(const int* __restrict__ row_ptr,
                                    int* __restrict__ row_ind,
                                    int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int i = start; i < end; i++) {
        row_ind[i] = row;
    }
}

COOData csr_to_coo(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz)
{
    COOData coo = {};
    coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.nnz      = nnz;

    if (nnz == 0) return coo;

    cudaMalloc(&coo.d_row_ind, nnz * sizeof(int));
    cudaMalloc(&coo.d_col_ind, nnz * sizeof(int));
    cudaMalloc(&coo.d_values,  nnz * sizeof(float));

    // Expand row_ptr → row_ind on GPU
    int threads = 256;
    int blocks  = (num_rows + threads - 1) / threads;
    _coo_expand_kernel<<<blocks, threads>>>(d_row_ptr, coo.d_row_ind, num_rows);

    // Copy col_ind and values (device-to-device)
    cudaMemcpy(coo.d_col_ind, d_col_ind, nnz * sizeof(int),   cudaMemcpyDeviceToDevice);
    cudaMemcpy(coo.d_values,  d_values,  nnz * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
    return coo;
}

void free_coo(COOData& m) {
    if (m.d_row_ind) cudaFree(m.d_row_ind);
    if (m.d_col_ind) cudaFree(m.d_col_ind);
    if (m.d_values)  cudaFree(m.d_values);
    m = {};
}


// ================================================================
//  ELL  (column-major)
// ================================================================

// GPU kernel: convert CSR → ELL column-major
__global__ void _ell_fill_kernel(const int* __restrict__ row_ptr,
                                  const int* __restrict__ csr_col,
                                  const float* __restrict__ csr_val,
                                  int* __restrict__ ell_col,
                                  float* __restrict__ ell_val,
                                  int num_rows, int max_nnz)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = row_ptr[row];
    int len   = row_ptr[row + 1] - start;

    for (int j = 0; j < max_nnz; j++) {
        int64_t idx = (int64_t)j * num_rows + row;   // column-major
        if (j < len) {
            ell_col[idx] = csr_col[start + j];
            ell_val[idx] = csr_val[start + j];
        } else {
            ell_col[idx] = -1;
            ell_val[idx] = 0.0f;
        }
    }
}

ELLData csr_to_ell(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz)
{
    ELLData ell = {};
    ell.num_rows = num_rows;
    ell.num_cols = num_cols;
    ell.nnz      = nnz;

    if (nnz == 0 || num_rows == 0) return ell;

    // Copy only row_ptr to host to find max_nnz_per_row
    std::vector<int> h_rp(num_rows + 1);
    cudaMemcpy(h_rp.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int max_nnz = 0;
    for (int i = 0; i < num_rows; i++) {
        max_nnz = std::max(max_nnz, h_rp[i + 1] - h_rp[i]);
    }
    ell.max_nnz_per_row = max_nnz;

    int64_t total = (int64_t)max_nnz * num_rows;
    ell.total_elements = total;

    // Guard: skip if ELL would use > 4 GB
    if (total * (int64_t)sizeof(float) > 4LL * 1024 * 1024 * 1024) {
        fprintf(stderr, "[sparse_formats] ELL skipped: would require %lld elements (%.1f GB)\n",
                (long long)total, (double)total * 8.0 / (1024.0 * 1024 * 1024));
        ell.d_col_ind = nullptr;
        ell.d_values  = nullptr;
        return ell;
    }

    cudaMalloc(&ell.d_col_ind, total * sizeof(int));
    cudaMalloc(&ell.d_values,  total * sizeof(float));

    if (!ell.d_col_ind || !ell.d_values) {
        fprintf(stderr, "[sparse_formats] ELL allocation failed\n");
        if (ell.d_col_ind) cudaFree(ell.d_col_ind);
        if (ell.d_values)  cudaFree(ell.d_values);
        ell.d_col_ind = nullptr;
        ell.d_values  = nullptr;
        return ell;
    }

    int threads = 256;
    int blocks  = (num_rows + threads - 1) / threads;
    _ell_fill_kernel<<<blocks, threads>>>(d_row_ptr, d_col_ind, d_values,
                                           ell.d_col_ind, ell.d_values,
                                           num_rows, max_nnz);
    cudaDeviceSynchronize();
    return ell;
}

void free_ell(ELLData& m) {
    if (m.d_col_ind) cudaFree(m.d_col_ind);
    if (m.d_values)  cudaFree(m.d_values);
    m = {};
}


// ================================================================
//  SELL-C-σ  (Sliced ELLPACK)
// ================================================================

SELLData csr_to_sell(const int* d_row_ptr, const int* d_col_ind,
                     const float* d_values,
                     int num_rows, int num_cols, int nnz,
                     int C, int sigma)
{
    SELLData sell = {};
    sell.num_rows = num_rows;
    sell.num_cols = num_cols;
    sell.nnz      = nnz;
    sell.C        = C;
    sell.sigma    = sigma;

    if (nnz == 0 || num_rows == 0 || C <= 0) return sell;

    // --- Copy CSR to host ---
    std::vector<int> h_rp, h_ci;
    std::vector<float> h_v;
    copy_csr_to_host(d_row_ptr, d_col_ind, d_values, num_rows, nnz, h_rp, h_ci, h_v);

    // --- Compute row lengths ---
    std::vector<int> row_len(num_rows);
    for (int i = 0; i < num_rows; i++) {
        row_len[i] = h_rp[i + 1] - h_rp[i];
    }

    // --- Pad num_rows to multiple of C ---
    int padded_rows = ((num_rows + C - 1) / C) * C;
    sell.num_slices     = padded_rows / C;
    sell.num_rows_padded = padded_rows;

    // --- Build row permutation ---
    // row_perm[sorted_pos] = original_row
    std::vector<int> row_perm(padded_rows);
    std::iota(row_perm.begin(), row_perm.begin() + num_rows, 0);
    // Pad with "ghost rows" (will have row_len = 0)
    for (int i = num_rows; i < padded_rows; i++) {
        row_perm[i] = i;  // ghost row index (≥ num_rows)
    }

    // Sort within windows of sigma rows (by decreasing row length)
    if (sigma > 0) {
        // Extend row_len for ghost rows
        row_len.resize(padded_rows, 0);

        for (int w = 0; w < padded_rows; w += sigma) {
            int end = std::min(w + sigma, padded_rows);
            std::sort(row_perm.begin() + w, row_perm.begin() + end,
                      [&](int a, int b) {
                          return row_len[a] > row_len[b];
                      });
        }
    } else {
        row_len.resize(padded_rows, 0);
    }

    // --- Compute per-slice max width ---
    std::vector<int> h_slice_width(sell.num_slices);
    for (int s = 0; s < sell.num_slices; s++) {
        int w = 0;
        for (int r = 0; r < C; r++) {
            int orig = row_perm[s * C + r];
            int len  = (orig < num_rows) ? row_len[orig] : 0;
            w = std::max(w, len);
        }
        h_slice_width[s] = w;
    }

    // --- Build slice_ptr (prefix sum of C × slice_width) ---
    std::vector<int> h_slice_ptr(sell.num_slices + 1);
    h_slice_ptr[0] = 0;
    for (int s = 0; s < sell.num_slices; s++) {
        h_slice_ptr[s + 1] = h_slice_ptr[s] + h_slice_width[s] * C;
    }
    int64_t total = h_slice_ptr[sell.num_slices];
    sell.total_padded = total;

    // --- Pack data (column-major within each slice) ---
    // data[ slice_ptr[s] + j * C + local_row ]
    std::vector<int>   h_sell_col(total, -1);
    std::vector<float> h_sell_val(total, 0.0f);

    for (int s = 0; s < sell.num_slices; s++) {
        int W = h_slice_width[s];
        for (int r = 0; r < C; r++) {
            int orig = row_perm[s * C + r];
            if (orig >= num_rows) continue;  // ghost row

            int start = h_rp[orig];
            int len   = row_len[orig];
            for (int j = 0; j < len; j++) {
                int64_t idx = h_slice_ptr[s] + (int64_t)j * C + r;
                h_sell_col[idx] = h_ci[start + j];
                h_sell_val[idx] = h_v[start + j];
            }
        }
    }

    // --- Upload to device ---
    cudaMalloc(&sell.d_col_ind,     total * sizeof(int));
    cudaMalloc(&sell.d_values,      total * sizeof(float));
    cudaMalloc(&sell.d_slice_ptr,   (sell.num_slices + 1) * sizeof(int));
    cudaMalloc(&sell.d_slice_width, sell.num_slices * sizeof(int));
    cudaMalloc(&sell.d_row_perm,    padded_rows * sizeof(int));

    cudaMemcpy(sell.d_col_ind,     h_sell_col.data(),     total * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(sell.d_values,      h_sell_val.data(),     total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.d_slice_ptr,   h_slice_ptr.data(),   (sell.num_slices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.d_slice_width, h_slice_width.data(), sell.num_slices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.d_row_perm,    row_perm.data(),       padded_rows * sizeof(int), cudaMemcpyHostToDevice);

    return sell;
}

void free_sell(SELLData& m) {
    if (m.d_col_ind)     cudaFree(m.d_col_ind);
    if (m.d_values)      cudaFree(m.d_values);
    if (m.d_slice_ptr)   cudaFree(m.d_slice_ptr);
    if (m.d_slice_width) cudaFree(m.d_slice_width);
    if (m.d_row_perm)    cudaFree(m.d_row_perm);
    m = {};
}


// ================================================================
//  BSR (Block Sparse Row)
// ================================================================

BSRData csr_to_bsr(const int* d_row_ptr, const int* d_col_ind,
                   const float* d_values,
                   int num_rows, int num_cols, int nnz,
                   int block_dim)
{
    BSRData bsr = {};
    bsr.num_rows  = num_rows;
    bsr.num_cols  = num_cols;
    bsr.block_dim = block_dim;

    int bs = block_dim;
    bsr.mb = (num_rows + bs - 1) / bs;
    bsr.nb = (num_cols + bs - 1) / bs;

    if (nnz == 0 || num_rows == 0 || bs <= 0) {
        bsr.nnzb = 0;
        bsr.fill_ratio = 0.0f;
        return bsr;
    }

    // --- Copy CSR to host ---
    std::vector<int> h_rp, h_ci;
    std::vector<float> h_v;
    copy_csr_to_host(d_row_ptr, d_col_ind, d_values, num_rows, nnz, h_rp, h_ci, h_v);

    // --- Find unique block columns per block row ---
    int nbr = bsr.mb;
    std::vector<std::vector<int>> blk_cols(nbr);

    for (int i = 0; i < num_rows; i++) {
        int br = i / bs;
        for (int j = h_rp[i]; j < h_rp[i + 1]; j++) {
            blk_cols[br].push_back(h_ci[j] / bs);
        }
    }

    // Sort and deduplicate each block row
    for (auto& cols : blk_cols) {
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }

    // --- Build bsr_row_ptr ---
    std::vector<int> h_bsr_rp(nbr + 1);
    h_bsr_rp[0] = 0;
    for (int br = 0; br < nbr; br++) {
        h_bsr_rp[br + 1] = h_bsr_rp[br] + (int)blk_cols[br].size();
    }
    bsr.nnzb = h_bsr_rp[nbr];

    // --- Build bsr_col_ind ---
    std::vector<int> h_bsr_ci(bsr.nnzb);
    for (int br = 0; br < nbr; br++) {
        for (int k = 0; k < (int)blk_cols[br].size(); k++) {
            h_bsr_ci[h_bsr_rp[br] + k] = blk_cols[br][k];
        }
    }

    // --- Fill dense block values (zero-initialized) ---
    int64_t total_vals = (int64_t)bsr.nnzb * bs * bs;
    std::vector<float> h_bsr_vals(total_vals, 0.0f);

    for (int i = 0; i < num_rows; i++) {
        int br      = i / bs;
        int local_r = i % bs;
        for (int j = h_rp[i]; j < h_rp[i + 1]; j++) {
            int bc      = h_ci[j] / bs;
            int local_c = h_ci[j] % bs;

            // Binary search for block index within this block row
            auto it = std::lower_bound(blk_cols[br].begin(), blk_cols[br].end(), bc);
            int bidx = h_bsr_rp[br] + (int)(it - blk_cols[br].begin());

            // Row-major within block
            h_bsr_vals[(int64_t)bidx * bs * bs + local_r * bs + local_c] = h_v[j];
        }
    }

    bsr.fill_ratio = (bsr.nnzb > 0)
        ? (float)((double)bsr.nnzb * bs * bs / nnz)
        : 0.0f;

    // --- Upload to device ---
    cudaMalloc(&bsr.d_bsr_row_ptr, (nbr + 1) * sizeof(int));
    cudaMalloc(&bsr.d_bsr_col_ind, bsr.nnzb * sizeof(int));
    cudaMalloc(&bsr.d_bsr_values,  total_vals * sizeof(float));

    cudaMemcpy(bsr.d_bsr_row_ptr, h_bsr_rp.data(),   (nbr + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(bsr.d_bsr_col_ind, h_bsr_ci.data(),    bsr.nnzb * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(bsr.d_bsr_values,  h_bsr_vals.data(),  total_vals * sizeof(float), cudaMemcpyHostToDevice);

    return bsr;
}

void free_bsr(BSRData& m) {
    if (m.d_bsr_row_ptr) cudaFree(m.d_bsr_row_ptr);
    if (m.d_bsr_col_ind) cudaFree(m.d_bsr_col_ind);
    if (m.d_bsr_values)  cudaFree(m.d_bsr_values);
    m = {};
}


// ================================================================
//  Row Partitioning
// ================================================================

RowPartitionData partition_rows(const int* d_row_ptr, int num_rows,
                                int threshold_low, int threshold_high)
{
    RowPartitionData rp = {};
    rp.num_rows       = num_rows;
    rp.threshold_low  = threshold_low;
    rp.threshold_high = threshold_high;

    if (num_rows == 0) return rp;

    // Copy row_ptr to host
    std::vector<int> h_rp(num_rows + 1);
    cudaMemcpy(h_rp.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Classify rows
    std::vector<int> s_ids, m_ids, d_ids;
    long long s_sum = 0, m_sum = 0, d_sum = 0;
    int s_min = INT_MAX, s_max = 0;
    int m_min = INT_MAX, m_max = 0;
    int d_min = INT_MAX, d_max = 0;

    for (int i = 0; i < num_rows; i++) {
        int len = h_rp[i + 1] - h_rp[i];
        if (len < threshold_low) {
            s_ids.push_back(i);
            s_sum += len;
            s_min = std::min(s_min, len);
            s_max = std::max(s_max, len);
        } else if (len < threshold_high) {
            m_ids.push_back(i);
            m_sum += len;
            m_min = std::min(m_min, len);
            m_max = std::max(m_max, len);
        } else {
            d_ids.push_back(i);
            d_sum += len;
            d_min = std::min(d_min, len);
            d_max = std::max(d_max, len);
        }
    }

    // Helper: fill one RowBin
    auto fill_bin = [](RowBin& bin, const std::vector<int>& ids,
                       int vmin, int vmax, long long vsum) {
        bin.count = (int)ids.size();
        if (bin.count == 0) {
            bin.d_row_ids = nullptr;
            bin.nnz_min = 0;
            bin.nnz_max = 0;
            bin.nnz_avg = 0.0f;
            return;
        }
        bin.nnz_min = vmin;
        bin.nnz_max = vmax;
        bin.nnz_avg = (float)((double)vsum / bin.count);
        cudaMalloc(&bin.d_row_ids, bin.count * sizeof(int));
        cudaMemcpy(bin.d_row_ids, ids.data(), bin.count * sizeof(int), cudaMemcpyHostToDevice);
    };

    fill_bin(rp.sparse, s_ids, s_min, s_max, s_sum);
    fill_bin(rp.medium, m_ids, m_min, m_max, m_sum);
    fill_bin(rp.dense,  d_ids, d_min, d_max, d_sum);

    return rp;
}

void free_row_partition(RowPartitionData& rp) {
    if (rp.sparse.d_row_ids) cudaFree(rp.sparse.d_row_ids);
    if (rp.medium.d_row_ids) cudaFree(rp.medium.d_row_ids);
    if (rp.dense.d_row_ids)  cudaFree(rp.dense.d_row_ids);
    rp = {};
}


// ================================================================
//  Tiled2D  (CB-SpMV style 16×16 blocks)
// ================================================================

namespace {

struct TileBuild {
    int blk_row;
    int blk_col;
    // local coords (row 0..15, col 0..15) in CSR traversal order
    std::vector<uint8_t> rows;
    std::vector<uint8_t> cols;
    std::vector<float>   vals;
};

inline int64_t align4(int64_t x) { return (x + 3) & ~(int64_t)3; }

}  // namespace

Tiled2DData csr_to_tiled2d(const int* d_row_ptr, const int* d_col_ind,
                           const float* d_values,
                           int num_rows, int num_cols, int nnz,
                           bool balance)
{
    const int T = TILED2D_TILE;
    const int W = TILED2D_WARPS_PER_TB;

    Tiled2DData t = {};
    t.num_rows = num_rows;
    t.num_cols = num_cols;
    t.nnz      = nnz;

    if (nnz == 0 || num_rows == 0) return t;

    std::vector<int> h_rp;
    std::vector<int> h_ci;
    std::vector<float> h_v;
    copy_csr_to_host(d_row_ptr, d_col_ind, d_values, num_rows, nnz, h_rp, h_ci, h_v);

    // --- Tile the matrix: one pass per block-row of 16 matrix rows ---
    int num_blk_rows = (num_rows + T - 1) / T;
    int num_blk_cols = (num_cols + T - 1) / T;
    std::vector<TileBuild> tiles;

    {
        // blk_col → index into `tiles` for the current block-row
        std::vector<int> slot_of_bc(num_blk_cols, -1);
        std::vector<int> touched;

        for (int br = 0; br < num_blk_rows; br++) {
            int r_lo = br * T;
            int r_hi = std::min(num_rows, r_lo + T);
            for (int r = r_lo; r < r_hi; r++) {
                for (int i = h_rp[r]; i < h_rp[r + 1]; i++) {
                    int c  = h_ci[i];
                    int bc = c / T;
                    int s  = slot_of_bc[bc];
                    if (s < 0) {
                        s = (int)tiles.size();
                        tiles.push_back(TileBuild{br, bc, {}, {}, {}});
                        slot_of_bc[bc] = s;
                        touched.push_back(bc);
                    }
                    tiles[s].rows.push_back((uint8_t)(r - r_lo));
                    tiles[s].cols.push_back((uint8_t)(c - bc * T));
                    tiles[s].vals.push_back(h_v[i]);
                }
            }
            for (int bc : touched) slot_of_bc[bc] = -1;
            touched.clear();
        }
    }

    int num_blocks = (int)tiles.size();
    t.num_blocks   = num_blocks;
    t.num_blk_rows = num_blk_rows;
    t.num_tbs      = (num_blocks + W - 1) / W;
    t.num_slots    = t.num_tbs * W;

    // --- Pick per-tile format, compute payload offsets ---
    std::vector<int>     type_of(num_blocks);
    std::vector<int64_t> vp_of(num_blocks);
    int     super_sparse = 0;
    int64_t offset = 0;

    for (int b = 0; b < num_blocks; b++) {
        int bn = (int)tiles[b].vals.size();
        int64_t bytes;
        if (bn > 128) {
            type_of[b] = TILED2D_DENSE;
            bytes = (int64_t)T * T * sizeof(float);
        } else if (bn >= 32) {
            type_of[b] = TILED2D_CSR;
            bytes = align4(17 + bn) + (int64_t)bn * sizeof(float);
        } else {
            type_of[b] = TILED2D_COO;
            bytes = align4(bn) + (int64_t)bn * sizeof(float);
            super_sparse++;
        }
        vp_of[b] = offset;
        offset += align4(bytes);
    }
    t.data_bytes = offset;
    t.super_sparse_frac = (float)super_sparse / num_blocks;

    // --- Pack payloads into one contiguous byte buffer ---
    std::vector<char> h_data(offset, 0);
    for (int b = 0; b < num_blocks; b++) {
        const TileBuild& tb = tiles[b];
        int bn  = (int)tb.vals.size();
        char* p = h_data.data() + vp_of[b];

        if (type_of[b] == TILED2D_DENSE) {
            float* dv = (float*)p;
            for (int i = 0; i < bn; i++)
                dv[tb.rows[i] * T + tb.cols[i]] = tb.vals[i];
        } else if (type_of[b] == TILED2D_CSR) {
            uint8_t* rp8  = (uint8_t*)p;
            uint8_t* col8 = rp8 + 17;
            float*   dv   = (float*)(p + align4(17 + bn));
            // entries are already row-major from the CSR traversal
            int cnt[T + 1] = {};
            for (int i = 0; i < bn; i++) cnt[tb.rows[i] + 1]++;
            for (int r = 0; r < T; r++) cnt[r + 1] += cnt[r];
            for (int r = 0; r <= T; r++) rp8[r] = (uint8_t)cnt[r];
            for (int i = 0; i < bn; i++) {
                col8[i] = tb.cols[i];
                dv[i]   = tb.vals[i];
            }
        } else {  // COO
            uint8_t* coord = (uint8_t*)p;
            float*   dv    = (float*)(p + align4(bn));
            for (int i = 0; i < bn; i++) {
                coord[i] = (uint8_t)((tb.cols[i] << 4) | tb.rows[i]);
                dv[i]    = tb.vals[i];
            }
        }
    }

    // --- Load balance: greedy min-heap of TBs, heaviest tiles first ---
    // slot_of[b] = slot the tile is assigned to (slot = tb_id*W + warp)
    std::vector<int> order(num_blocks);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> slot_of(num_blocks);

    if (balance && t.num_tbs > 1) {
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return tiles[a].vals.size() > tiles[b].vals.size();
        });
        // min-heap of (accumulated nnz, tb_id, warps_used)
        typedef std::pair<long long, std::pair<int, int> > HeapEntry;
        std::vector<HeapEntry> heap(t.num_tbs);
        for (int i = 0; i < t.num_tbs; i++)
            heap[i] = HeapEntry(0LL, std::make_pair(i, 0));
        auto cmp = [](const HeapEntry& a, const HeapEntry& b) {
            return a.first > b.first;  // min-heap on load
        };
        std::make_heap(heap.begin(), heap.end(), cmp);

        for (int b : order) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            HeapEntry e = heap.back();
            heap.pop_back();
            int tb_id = e.second.first;
            int warps = e.second.second;
            slot_of[b] = tb_id * W + warps;
            e.first += (long long)tiles[b].vals.size();
            e.second.second = warps + 1;
            if (e.second.second < W) {
                heap.push_back(e);
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
    } else {
        // Unbalanced: keep tile order (grouped by block-row from the build
        // pass) and publish a tile-level row pointer for block-wise products.
        for (int b = 0; b < num_blocks; b++) slot_of[b] = b;
        std::vector<int> brp(num_blk_rows + 1, 0);
        for (int b = 0; b < num_blocks; b++) brp[tiles[b].blk_row + 1]++;
        for (int r = 0; r < num_blk_rows; r++) brp[r + 1] += brp[r];
        cudaMalloc(&t.d_blk_row_ptr, (num_blk_rows + 1) * sizeof(int));
        cudaMemcpy(t.d_blk_row_ptr, brp.data(),
                   (num_blk_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // --- Emit metadata in slot order (empty slots: type = EMPTY) ---
    std::vector<int>     m_row(t.num_slots, 0);
    std::vector<int>     m_col(t.num_slots, 0);
    std::vector<int>     m_nnz(t.num_slots, 0);
    std::vector<int>     m_type(t.num_slots, TILED2D_EMPTY);
    std::vector<int64_t> m_vp(t.num_slots, 0);
    for (int b = 0; b < num_blocks; b++) {
        int s = slot_of[b];
        m_row[s]  = tiles[b].blk_row;
        m_col[s]  = tiles[b].blk_col;
        m_nnz[s]  = (int)tiles[b].vals.size();
        m_type[s] = type_of[b];
        m_vp[s]   = vp_of[b];
    }

    cudaMalloc(&t.d_blk_row,  t.num_slots * sizeof(int));
    cudaMalloc(&t.d_blk_col,  t.num_slots * sizeof(int));
    cudaMalloc(&t.d_blk_nnz,  t.num_slots * sizeof(int));
    cudaMalloc(&t.d_blk_type, t.num_slots * sizeof(int));
    cudaMalloc(&t.d_blk_vp,   t.num_slots * sizeof(int64_t));
    cudaMalloc(&t.d_data,     t.data_bytes);

    cudaMemcpy(t.d_blk_row,  m_row.data(),  t.num_slots * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(t.d_blk_col,  m_col.data(),  t.num_slots * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(t.d_blk_nnz,  m_nnz.data(),  t.num_slots * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(t.d_blk_type, m_type.data(), t.num_slots * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(t.d_blk_vp,   m_vp.data(),   t.num_slots * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(t.d_data,     h_data.data(), t.data_bytes,                  cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return t;
}

void free_tiled2d(Tiled2DData& m) {
    if (m.d_blk_row)     cudaFree(m.d_blk_row);
    if (m.d_blk_col)     cudaFree(m.d_blk_col);
    if (m.d_blk_nnz)     cudaFree(m.d_blk_nnz);
    if (m.d_blk_type)    cudaFree(m.d_blk_type);
    if (m.d_blk_vp)      cudaFree(m.d_blk_vp);
    if (m.d_data)        cudaFree(m.d_data);
    if (m.d_blk_row_ptr) cudaFree(m.d_blk_row_ptr);
    m = {};
}


// ================================================================
//  Windowed  (Sextans style packed column windows)
// ================================================================

WindowedData csr_to_windowed(const int* d_row_ptr, const int* d_col_ind,
                             const float* d_values,
                             int num_rows, int num_cols, int nnz,
                             int K0, int R0)
{
    WindowedData w = {};
    w.num_rows = num_rows;
    w.num_cols = num_cols;
    w.nnz      = nnz;

    if (K0 < 1 || K0 > 16384) {
        fprintf(stderr, "csr_to_windowed: K0=%d out of range (1..16384), clamping\n", K0);
        K0 = std::min(std::max(K0, 1), 16384);
    }
    if (R0 < 1 || R0 > 262144) {
        fprintf(stderr, "csr_to_windowed: R0=%d out of range (1..262144), clamping\n", R0);
        R0 = std::min(std::max(R0, 1), 262144);
    }
    w.K0 = K0;
    w.R0 = R0;
    w.num_row_blocks = (num_rows + R0 - 1) / R0;
    w.num_windows    = (num_cols + K0 - 1) / K0;

    if (nnz == 0 || num_rows == 0) return w;

    std::vector<int> h_rp;
    std::vector<int> h_ci;
    std::vector<float> h_v;
    copy_csr_to_host(d_row_ptr, d_col_ind, d_values, num_rows, nnz, h_rp, h_ci, h_v);

    int num_segs = w.num_row_blocks * w.num_windows;

    // Two passes: count per segment, then scatter (keeps row-major
    // order within each segment because rows are visited in order).
    std::vector<int> seg_ptr(num_segs + 1, 0);
    for (int r = 0; r < num_rows; r++) {
        int rb = r / R0;
        for (int i = h_rp[r]; i < h_rp[r + 1]; i++)
            seg_ptr[rb * w.num_windows + h_ci[i] / K0 + 1]++;
    }
    for (int s = 0; s < num_segs; s++) seg_ptr[s + 1] += seg_ptr[s];

    std::vector<unsigned long long> packed(nnz);
    std::vector<int> cursor(seg_ptr.begin(), seg_ptr.end() - 1);
    for (int r = 0; r < num_rows; r++) {
        int rb = r / R0;
        unsigned long long row_local = (unsigned long long)(r - rb * R0);
        for (int i = h_rp[r]; i < h_rp[r + 1]; i++) {
            int c   = h_ci[i];
            int win = c / K0;
            unsigned long long col_local = (unsigned long long)(c - win * K0);
            unsigned int vbits;
            memcpy(&vbits, &h_v[i], sizeof(float));
            packed[cursor[rb * w.num_windows + win]++] =
                (col_local << 50) | (row_local << 32) | (unsigned long long)vbits;
        }
    }

    cudaMalloc(&w.d_packed, (size_t)nnz * sizeof(unsigned long long));
    cudaMalloc(&w.d_seg_ptr, (num_segs + 1) * sizeof(int));
    cudaMemcpy(w.d_packed, packed.data(), (size_t)nnz * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(w.d_seg_ptr, seg_ptr.data(), (num_segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return w;
}

void free_windowed(WindowedData& m) {
    if (m.d_packed)  cudaFree(m.d_packed);
    if (m.d_seg_ptr) cudaFree(m.d_seg_ptr);
    m = {};
}


// ================================================================
//  NNZ-balanced chunks
// ================================================================

NnzChunkData balance_nnz_chunks(const int* d_row_ptr, int num_rows, int nnz,
                                int num_chunks)
{
    NnzChunkData m = {};
    m.num_rows = num_rows;
    m.nnz      = nnz;
    if (num_rows == 0 || nnz == 0 || num_chunks < 1) return m;

    num_chunks = std::min(num_chunks, nnz);
    m.num_chunks    = num_chunks;
    m.nnz_per_chunk = (nnz + num_chunks - 1) / num_chunks;

    std::vector<int> h_rp(num_rows + 1);
    cudaMemcpy(h_rp.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // first_row[c] = row containing nnz index c*S
    // (upper_bound gives the first row whose row_ptr exceeds the target)
    std::vector<int> first_row(num_chunks + 1);
    for (int c = 0; c < num_chunks; c++) {
        long long target = (long long)c * m.nnz_per_chunk;
        int row = (int)(std::upper_bound(h_rp.begin(), h_rp.end(), (int)target)
                        - h_rp.begin()) - 1;
        first_row[c] = std::min(row, num_rows - 1);
    }
    first_row[num_chunks] = num_rows;

    cudaMalloc(&m.d_chunk_first_row, (num_chunks + 1) * sizeof(int));
    cudaMemcpy(m.d_chunk_first_row, first_row.data(),
               (num_chunks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    return m;
}

void free_nnz_chunks(NnzChunkData& m) {
    if (m.d_chunk_first_row) cudaFree(m.d_chunk_first_row);
    m = {};
}


// ================================================================
//  Cyclic row interleaving
// ================================================================

CyclicPermData cyclic_row_perm(int num_rows, int P)
{
    CyclicPermData m = {};
    m.num_rows = num_rows;
    m.P        = std::max(1, std::min(P, num_rows));
    if (num_rows == 0) return m;

    std::vector<int> perm(num_rows), inv(num_rows);
    int pos = 0;
    for (int p = 0; p < m.P; p++)
        for (int r = p; r < num_rows; r += m.P) {
            perm[pos] = r;
            inv[r]    = pos;
            pos++;
        }

    cudaMalloc(&m.d_perm,     num_rows * sizeof(int));
    cudaMalloc(&m.d_inv_perm, num_rows * sizeof(int));
    cudaMemcpy(m.d_perm,     perm.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m.d_inv_perm, inv.data(),  num_rows * sizeof(int), cudaMemcpyHostToDevice);
    return m;
}

void free_cyclic_perm(CyclicPermData& m) {
    if (m.d_perm)     cudaFree(m.d_perm);
    if (m.d_inv_perm) cudaFree(m.d_inv_perm);
    m = {};
}
