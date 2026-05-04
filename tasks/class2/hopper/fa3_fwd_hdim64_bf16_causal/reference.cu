/*
 * Flash Attention 3 Forward — hdim=64, bf16, causal
 *
 * Source: flash-attention/hopper/ (FA3 Hopper kernels)
 * Kernel source code in reference_sources/flash_attn_v3/.
 */

#include "flash_fwd_launch_template.h"

// Stub: referenced by dead varlen code path but never called in non-varlen mode
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream,
                               bool packgqa, int blockM, int blockN, bool enable_pdl) {}

// Template instantiation: <Arch=90, T, kHeadDim, kHeadDimV, Split=false, PagedKVNonTMA=false, Has_softcap=false, PackGQA=false>
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(
    Flash_fwd_params &params, cudaStream_t stream);

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_fa3_fwd(
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

    params.v_dim_stride = 1;  // Row-major V

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
    params.num_splits = 1;

    params.is_bf16 = true;
    params.is_e4m3 = false;

    // Device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    params.num_sm = prop.multiProcessorCount;
    params.arch = 90;

    // Persistent scheduler semaphore (required for SM90)
    static int* d_semaphore = nullptr;
    if (!d_semaphore) {
        cudaMalloc(&d_semaphore, sizeof(int));
    }
    cudaMemsetAsync(d_semaphore, 0, sizeof(int), stream);
    params.tile_count_semaphore = d_semaphore;

    run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(params, stream);
}
