/*
 * Flash Attention Forward — hdim=256, bf16, causal
 *
 * Source: flash-attention (Tri Dao)
 * Kernel source code in reference_sources/, copied from:
 *   flash-attention/csrc/flash_attn/src/
 *
 * Compile with:
 *   -I reference/
 *   -I <cutlass>/include/
 *   -gencode arch=compute_80,code=compute_80
 */

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_flash_attn_fwd(
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
    params.seqlen_q_rounded = round_up(Sq, 128);
    params.seqlen_k_rounded = round_up(Sk, 128);
    params.d_rounded = round_up(D, 32);
    params.rotary_dim = 0;
    params.total_q = B * Sq;

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * M_LOG2E;

    params.softmax_lse_ptr = lse;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.p_ptr = nullptr;

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

    // Causal masking
    params.is_causal = true;
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.softcap = 0.0f;

    params.is_bf16 = true;
    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 256, true>(params, stream);
}
