// Reference fused multi-head attention forward pass for the 41_fused_multi_head_attention harness.
//
// The reference is FlashAttention-2 (Dao-AILab/flash-attention, csrc/flash_attn/src, BSD-3-Clause),
// compiled from reference_sources/flash_attn into fa2_ref.o by the Makefile (fa2_ref.cu).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/// True if the reference supports this head dimension (head_dim == head_dim_v in {64, 128}).
bool kh_ref_fmha_supported(int head_dim);

/// O = softmax(scale * Q K^T [causal mask]) V for FP16 tensors in BMHK layout
/// ([batch, seq, heads, head_dim]; strides in elements, innermost stride 1).
/// lse: FP32 scratch of batch * heads * seq_q elements (row log-sum-exp written by the kernel).
cudaError_t kh_ref_fmha_fwd(
    int batch, int seq_q, int seq_kv, int heads, int head_dim,
    void const *Q, void const *K, void const *V, void *O, float *lse,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float scale, bool causal, cudaStream_t stream);
