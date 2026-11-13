#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    BSRData bsr;
    int* d_chunk_rows;
    int num_rows;
    int num_cols;
    int num_chunks;
};

__global__ void build_chunk_rows_kernel(
    const int* __restrict__ row_ptr,
    int* __restrict__ chunk_rows,
    int mb,
    int num_chunks)
{
    int chunk = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (chunk >= num_chunks) return;

    int block_index = chunk * 4;
    int lo = 0;
    int hi = mb - 1;

    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (row_ptr[mid] <= block_index) lo = mid;
        else hi = mid - 1;
    }

    chunk_rows[chunk] = lo;
}

__device__ __forceinline__ void cp_async_16(float* dst, const float* src)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void stage_a_chunk(
    float* __restrict__ s_a,
    const float* __restrict__ values,
    int block_begin,
    int block_end,
    int tid)
{
    constexpr int A_PER_CHUNK = 1024;

    for (int i = tid; i < A_PER_CHUNK; i += 128) {
        int slot = i >> 8;
        int elem = i & 255;
        int gb = block_begin + slot;
        s_a[i] = (gb < block_end) ? values[(size_t)gb * 256 + elem] : 0.0f;
    }
}

__device__ __forceinline__ void prefetch_b_chunk(
    float* __restrict__ s_b,
    const int* __restrict__ col_ind,
    const float* __restrict__ B,
    int num_cols,
    int block_begin,
    int block_end,
    int tid)
{
    constexpr int B_PER_CHUNK = 512;

    for (int vec = tid; vec < B_PER_CHUNK / 4; vec += 128) {
        int float_offset = vec * 4;
        int slot = float_offset >> 7;
        int within = float_offset & 127;
        int r = within >> 3;
        int k = within & 7;
        int gb = block_begin + slot;

        if (gb < block_end) {
            int brow = col_ind[gb] * 16 + r;

            if (brow < num_cols && k <= 4) {
                cp_async_16(
                    s_b + float_offset,
                    B + (size_t)brow * 8 + k);
            } else {
                float* d = s_b + float_offset;
                if (brow < num_cols) {
                    d[0] = (k + 0 < 8) ? B[(size_t)brow * 8 + k + 0] : 0.0f;
                    d[1] = (k + 1 < 8) ? B[(size_t)brow * 8 + k + 1] : 0.0f;
                    d[2] = (k + 2 < 8) ? B[(size_t)brow * 8 + k + 2] : 0.0f;
                    d[3] = (k + 3 < 8) ? B[(size_t)brow * 8 + k + 3] : 0.0f;
                } else {
                    d[0] = 0.0f;
                    d[1] = 0.0f;
                    d[2] = 0.0f;
                    d[3] = 0.0f;
                }
            }
        } else {
            float* d = s_b + float_offset;
            d[0] = 0.0f;
            d[1] = 0.0f;
            d[2] = 0.0f;
            d[3] = 0.0f;
        }
    }
}

__global__ void bsr16_balanced_spmm_k8_async_b(
    const int* __restrict__ bsr_row_ptr,
    const int* __restrict__ bsr_col_ind,
    const float* __restrict__ bsr_values,
    const int* __restrict__ chunk_rows,
    const float* __restrict__ B,
    float* __restrict__ C,
    int mb,
    int num_rows,
    int num_cols,
    int nnzb,
    int num_chunks)
{
    constexpr int A_PER_CHUNK = 1024;
    constexpr int B_PER_CHUNK = 512;
    constexpr int BLOCKS_PER_CHUNK = 4;

    __shared__ float s_a[A_PER_CHUNK];
    __shared__ float s_b[2][B_PER_CHUNK];
    __shared__ int s_segment_end;
    __shared__ int s_block_row;

    const int tid = (int)threadIdx.x;
    const int local_row = tid >> 3;
    const int out_k = tid & 7;
    const int stride = (int)gridDim.x;

    int chunk = (int)blockIdx.x;
    if (chunk >= num_chunks) return;

    int block_begin = chunk * BLOCKS_PER_CHUNK;
    int block_end = min(block_begin + BLOCKS_PER_CHUNK, nnzb);
    int bbuf = 0;

    prefetch_b_chunk(
        s_b[bbuf],
        bsr_col_ind,
        B,
        num_cols,
        block_begin,
        block_end,
        tid);

    cp_async_commit();
    cp_async_wait();

    stage_a_chunk(
        s_a,
        bsr_values,
        block_begin,
        block_end,
        tid);

    __syncthreads();

    while (chunk < num_chunks) {
        int next_chunk = chunk + stride;
        int next_begin = next_chunk * BLOCKS_PER_CHUNK;
        int next_end = min(next_begin + BLOCKS_PER_CHUNK, nnzb);
        int next_buf = bbuf ^ 1;

        if (next_chunk < num_chunks) {
            prefetch_b_chunk(
                s_b[next_buf],
                bsr_col_ind,
                B,
                num_cols,
                next_begin,
                next_end,
                tid);
            cp_async_commit();
        }

        int current = block_begin;
        int block_row = chunk_rows[chunk];

        while (current < block_end) {
            if (tid == 0) {
                while (block_row + 1 < mb &&
                       current >= bsr_row_ptr[block_row + 1]) {
                    ++block_row;
                }

                s_block_row = block_row;
                s_segment_end = min(block_end, bsr_row_ptr[block_row + 1]);
            }

            __syncthreads();

            int segment_end = s_segment_end;
            int output_block_row = s_block_row;
            float acc = 0.0f;

            for (int block = current; block < segment_end; ++block) {
                int slot = block - block_begin;
                const float* a = s_a + slot * 256 + local_row * 16;
                const float* b = s_b[bbuf] + slot * 128 + out_k;

                acc = fmaf(a[0],  b[0],   acc);
                acc = fmaf(a[1],  b[8],   acc);
                acc = fmaf(a[2],  b[16],  acc);
                acc = fmaf(a[3],  b[24],  acc);
                acc = fmaf(a[4],  b[32],  acc);
                acc = fmaf(a[5],  b[40],  acc);
                acc = fmaf(a[6],  b[48],  acc);
                acc = fmaf(a[7],  b[56],  acc);
                acc = fmaf(a[8],  b[64],  acc);
                acc = fmaf(a[9],  b[72],  acc);
                acc = fmaf(a[10], b[80],  acc);
                acc = fmaf(a[11], b[88],  acc);
                acc = fmaf(a[12], b[96],  acc);
                acc = fmaf(a[13], b[104], acc);
                acc = fmaf(a[14], b[112], acc);
                acc = fmaf(a[15], b[120], acc);
            }

            int output_row = output_block_row * 16 + local_row;
            if (output_row < num_rows) {
                atomicAdd(&C[(size_t)output_row * 8 + out_k], acc);
            }

            __syncthreads();
            current = segment_end;
            ++block_row;
        }

        if (next_chunk >= num_chunks) break;

        cp_async_wait();

        stage_a_chunk(
            s_a,
            bsr_values,
            next_begin,
            next_end,
            tid);

        __syncthreads();

        chunk = next_chunk;
        block_begin = next_begin;
        block_end = next_end;
        bbuf = next_buf;
    }
}

void* spmm_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz,
    int K)
{
    (void)K;

    SpMMContext* ctx = new SpMMContext;
    ctx->bsr = csr_to_bsr(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz,
        16);

    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->num_chunks = (ctx->bsr.nnzb + 3) / 4;
    ctx->d_chunk_rows = nullptr;

    if (ctx->num_chunks > 0 && ctx->bsr.mb > 0) {
        cudaMalloc(
            &ctx->d_chunk_rows,
            (size_t)ctx->num_chunks * sizeof(int));

        constexpr int threads = 256;
        int blocks = (ctx->num_chunks + threads - 1) / threads;

        build_chunk_rows_kernel<<<blocks, threads>>>(
            ctx->bsr.d_bsr_row_ptr,
            ctx->d_chunk_rows,
            ctx->bsr.mb,
            ctx->num_chunks);
    }

    return ctx;
}

void spmm_compute(
    void* context,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_B,
    float* d_C,
    int num_rows,
    int num_cols,
    int nnz,
    int K)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;
    (void)K;

    if (num_rows <= 0) return;

    cudaMemsetAsync(
        d_C,
        0,
        (size_t)num_rows * 8 * sizeof(float),
        0);

    SpMMContext* ctx = static_cast<SpMMContext*>(context);

    if (ctx == nullptr ||
        ctx->bsr.nnzb <= 0 ||
        ctx->bsr.mb <= 0 ||
        ctx->num_chunks <= 0 ||
        ctx->d_chunk_rows == nullptr) {
        return;
    }

    constexpr int persistent_ctas = 1880;

    bsr16_balanced_spmm_k8_async_b<<<persistent_ctas, 128>>>(
        ctx->bsr.d_bsr_row_ptr,
        ctx->bsr.d_bsr_col_ind,
        ctx->bsr.d_bsr_values,
        ctx->d_chunk_rows,
        d_B,
        d_C,
        ctx->bsr.mb,
        num_rows,
        num_cols,
        ctx->bsr.nnzb,
        ctx->num_chunks);
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);

    if (ctx != nullptr) {
        if (ctx->d_chunk_rows != nullptr) {
            cudaFree(ctx->d_chunk_rows);
        }

        free_bsr(ctx->bsr);
        delete ctx;
    }
}