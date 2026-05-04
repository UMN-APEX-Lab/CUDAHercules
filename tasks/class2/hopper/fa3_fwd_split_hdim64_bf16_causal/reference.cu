/*
 * Flash Attention 3 Forward Split-KV — hdim=64, bf16, causal
 *
 * Source: flash-attention/hopper/ (FA3 Hopper kernels)
 * Uses split-KV parallelism with 4 splits.
 */

#include "flash_fwd_launch_template.h"
#include "flash_fwd_combine_launch_template.h"

// Stub: referenced by dead varlen code path but never called in non-varlen mode
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream,
                               bool packgqa, int blockM, int blockN, bool enable_pdl) {}

// Template instantiation: <Arch=90, T, kHeadDim, kHeadDimV, Split=true, PagedKVNonTMA=false, Has_softcap=false, PackGQA=true>
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, true, false, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Combine kernel instantiation
template void run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_fa3_fwd_split(
    const void* Q,     // [B, H, Sq, D] bf16
    const void* K,     // [B, H, Sk, D] bf16
    const void* V,     // [B, H, Sk, D] bf16
    void* O,           // [B, H, Sq, D] bf16
    float* lse,        // [B, H, Sq]
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    int num_splits,
    cudaStream_t stream
) {
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K);
    params.v_ptr = const_cast<void*>(V);
    params.o_ptr = O;

    params.q_batch_stride = static_cast<int64_t>(H) * Sq * D;
    params.k_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.v_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.o_batch_stride = static_cast<int64_t>(H) * Sq * D;

    params.q_row_stride = D;
    params.k_row_stride = D;
    params.v_row_stride = D;
    params.o_row_stride = D;

    params.q_head_stride = static_cast<int64_t>(Sq) * D;
    params.k_head_stride = static_cast<int64_t>(Sk) * D;
    params.v_head_stride = static_cast<int64_t>(Sk) * D;
    params.o_head_stride = static_cast<int64_t>(Sq) * D;

    params.v_dim_stride = 1;

    params.h = H;
    params.h_k = H;

    params.b = B;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.d = D;
    params.dv = D;
    params.d_rounded = round_up(D, 32);
    params.dv_rounded = round_up(D, 32);
    params.seqlen_q_rounded = round_up(Sq, 128);
    params.seqlen_k_rounded = round_up(Sk, 128);
    params.total_q = B * Sq;
    params.total_k = B * Sk;

    params.scale_softmax = scale;
    params.softmax_lse_ptr = lse;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    params.is_causal = true;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.softcap = 0.0f;
    params.num_splits = num_splits;

    params.is_bf16 = true;
    params.is_e4m3 = false;

    // Device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    params.num_sm = prop.multiProcessorCount;
    params.arch = 90;

    // Scheduler semaphore
    static int* d_semaphore = nullptr;
    if (!d_semaphore) {
        cudaMalloc(&d_semaphore, sizeof(int));
    }
    cudaMemsetAsync(d_semaphore, 0, sizeof(int), stream);
    params.tile_count_semaphore = d_semaphore;

    // Split-KV accumulation buffers
    int dv_rounded = round_up(D, 32);
    static float* d_oaccum = nullptr;
    static float* d_lseaccum = nullptr;
    static size_t oaccum_size = 0;
    static size_t lseaccum_size = 0;

    size_t needed_oaccum = (size_t)B * H * Sq * dv_rounded * num_splits * sizeof(float);
    size_t needed_lseaccum = (size_t)B * H * Sq * num_splits * sizeof(float);

    if (needed_oaccum > oaccum_size) {
        if (d_oaccum) cudaFree(d_oaccum);
        cudaMalloc(&d_oaccum, needed_oaccum);
        oaccum_size = needed_oaccum;
    }
    if (needed_lseaccum > lseaccum_size) {
        if (d_lseaccum) cudaFree(d_lseaccum);
        cudaMalloc(&d_lseaccum, needed_lseaccum);
        lseaccum_size = needed_lseaccum;
    }

    params.oaccum_ptr = d_oaccum;
    params.softmax_lseaccum_ptr = d_lseaccum;
    params.oaccum_split_stride = static_cast<int64_t>(B) * H * Sq * dv_rounded;
    params.oaccum_batch_stride = static_cast<int64_t>(H) * Sq * dv_rounded;
    params.oaccum_row_stride = dv_rounded;
    params.oaccum_head_stride = static_cast<int64_t>(Sq) * dv_rounded;
    params.lseaccum_split_stride = static_cast<int64_t>(B) * H * Sq;
    params.lseaccum_batch_stride = static_cast<int64_t>(H) * Sq;
    params.lseaccum_head_stride = Sq;

    run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, true, false, false, true>(params, stream);

    // Combine split results
    run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, false);
}
