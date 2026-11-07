"""
Task: Flash Attention Forward — hdim=128, bf16, non-causal

Source: flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
Operation: O = softmax(Q @ K^T * scale) @ V
Also outputs log-sum-exp (LSE) per query row for backward pass.
"""
import torch
import math

BATCH_SIZE = 8
NUM_HEADS = 16
SEQ_LEN_Q = 8192
SEQ_LEN_K = 8192
HEAD_DIM = 128

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_flash_attn_fwd(
    const void* Q,     // [B, H, Sq, D] bf16
    const void* K,     // [B, H, Sk, D] bf16
    const void* V,     // [B, H, Sk, D] bf16
    void* O,           // [B, H, Sq, D] bf16
    float* lse,        // [B, H, Sq] log-sum-exp per query row
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {
    "B": BATCH_SIZE,
    "H": NUM_HEADS,
    "Sq": SEQ_LEN_Q,
    "Sk": SEQ_LEN_K,
    "D": HEAD_DIM,
    "scale": SCALE,
}

TOLERANCES = {"atol": 0.002, "rtol": 0.02}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/flash_attn",
    "reference_sources/flash-attention/csrc/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = ["-gencode", "arch=compute_80,code=compute_80"]


def get_inputs():
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    return [("Q", Q), ("K", K), ("V", V)]


def get_outputs(inputs):
    Q = inputs[0][1]
    B, H, Sq, D = Q.shape
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    O = torch.full((B, H, Sq, D), float('nan'), device='cuda', dtype=torch.bfloat16)
    lse = torch.full((B, H, Sq), float('nan'), device='cuda', dtype=torch.float32)
    return [("O", O), ("lse", lse)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    Q = inputs[0][1].float()
    K = inputs[1][1].float()
    V = inputs[2][1].float()

    B, H, Sq, D = Q.shape
    Sk = K.shape[2]

    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * SCALE

    lse = torch.logsumexp(attn_weights, dim=-1)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    O = torch.matmul(attn_probs, V).to(torch.bfloat16)

    return [("O", O), ("lse", lse)]


DESCRIPTION = """\
Implement a CUDA kernel for the forward pass of exact softmax attention.
The full Sq x Sk attention matrix must not be materialized in global memory.

For every batch b and head h (Q, K and V have the same number of heads H):
  S = Q @ K^T * scale
  P = softmax(S) along the key axis
  O = P @ V
  lse[i] = log(sum_j exp(S[i][j]))   (natural log over the unmasked keys of row i)

Inputs (bf16, contiguous row-major [B, H, S, D] layout, D fastest):
  Q: [B, H, Sq, D] = [%d, %d, %d, %d]
  K: [B, H, Sk, D] = [%d, %d, %d, %d]
  V: [B, H, Sk, D] = [%d, %d, %d, %d]
  All input elements are i.i.d. standard normal, rounded to bf16.

Outputs:
  O:   [B, H, Sq, D] bf16 — attention output
  lse: [B, H, Sq] fp32 — log-sum-exp per query row

scale = 1/sqrt(D) = %f

Precision: matrix products must use bf16 (or higher-precision) operands with FP32
accumulation, and the softmax statistics / lse must be computed in FP32. Quantizing
Q, K, V or P to FP8, INT8 or any lower precision is not allowed.

Correctness: O and lse are compared element-wise with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 0.002, rtol = 0.02
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       SCALE)
