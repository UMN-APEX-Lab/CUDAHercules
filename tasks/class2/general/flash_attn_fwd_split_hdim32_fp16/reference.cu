/*
 * Flash Attention Forward Split-KV — hdim=32, fp16, non-causal
 *
 * Source: flash-attention (Tri Dao)
 * Kernel source code in reference_sources/.
 * Split-KV variant: splits K/V across multiple thread block groups,
 * each computes partial results, then combines.
 */

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_flash_attn_fwd_split(
    const void* Q,     // [B, H, Sq, D] fp16
    const void* K,     // [B, H, Sk, D] fp16
    const void* V,     // [B, H, Sk, D] fp16
    void* O,           // [B, H, Sq, D] fp16
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
    FLASH_NAMESPACE::Flash_fwd_params params;
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

    params.h = H;
    params.h_k = H;
    params.h_h_k_ratio = 1;

    params.b = B;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.seqlen_knew = 0;
    params.d = D;
    int seqlen_q_rounded = round_up(Sq, 128);
    int seqlen_k_rounded = round_up(Sk, 128);
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d_rounded = round_up(D, 32);
    params.rotary_dim = 0;
    params.total_q = B * Sq;

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * M_LOG2E;

    params.softmax_lse_ptr = lse;
    params.p_ptr = nullptr;
    params.num_splits = num_splits;

    // Allocate temporary buffers for split-KV
    static float* oaccum = nullptr; static size_t oaccum_cached_size = 0;
    static float* lseaccum = nullptr; static size_t lseaccum_cached_size = 0;
    { size_t _need = (size_t)num_splits * B * H * seqlen_q_rounded * D * sizeof(float);
        if (oaccum_cached_size < _need) { if (oaccum) cudaFree(oaccum); cudaMalloc(&oaccum, _need); oaccum_cached_size = _need; } }
    { size_t _need = (size_t)num_splits * B * H * seqlen_q_rounded * sizeof(float);
        if (lseaccum_cached_size < _need) { if (lseaccum) cudaFree(lseaccum); cudaMalloc(&lseaccum, _need); lseaccum_cached_size = _need; } }
    params.oaccum_ptr = oaccum;
    params.softmax_lseaccum_ptr = lseaccum;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;

    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;

    params.is_bf16 = false;
    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, false>(params, stream);

    cudaStreamSynchronize(stream);
    /* cudaFree(oaccum); removed — buffer cached across invocations */
    /* cudaFree(lseaccum); removed — buffer cached across invocations */
}
