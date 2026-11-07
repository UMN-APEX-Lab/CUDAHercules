"""
Task: Flash Attention Backward — hdim=96, bf16, causal

Source: flash-attention/csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu
Operation: Compute dQ, dK, dV given dO, Q, K, V, O, and saved LSE.
"""
import torch
import math

BATCH_SIZE = 8
NUM_HEADS = 16
SEQ_LEN = 8192
HEAD_DIM = 96

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_flash_attn_bwd(
    const void* dO,    // [B, H, S, D] bf16
    const void* Q,     // [B, H, S, D] bf16
    const void* K,     // [B, H, S, D] bf16
    const void* V,     // [B, H, S, D] bf16
    const void* O,     // [B, H, S, D] bf16
    const float* lse,  // [B, H, S]
    void* dQ,          // [B, H, S, D] bf16
    void* dK,          // [B, H, S, D] bf16
    void* dV,          // [B, H, S, D] bf16
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
    "Sq": SEQ_LEN,
    "Sk": SEQ_LEN,
    "D": HEAD_DIM,
    "scale": SCALE,
}

TOLERANCES = {"atol": 0.025, "rtol": 0.05}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/flash_attn",
    "reference_sources/flash-attention/csrc/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = ["-gencode", "arch=compute_80,code=compute_80"]


def get_inputs():
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)

    # Forward O and lse in FP32, a few heads at a time: the full [B, H, S, S] score
    # tensor would need B*H*S*S*4 bytes (64-128 GB for the hdim32/64 shapes).
    O = torch.empty_like(Q)
    lse = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, device=Q.device, dtype=torch.float32)
    causal_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device, dtype=torch.bool), diagonal=1)
    for b in range(BATCH_SIZE):
        for h0 in range(0, NUM_HEADS, 4):
            h1 = min(h0 + 4, NUM_HEADS)
            attn_w = torch.matmul(Q[b, h0:h1].float(), K[b, h0:h1].float().transpose(-2, -1)) * SCALE
            attn_w.masked_fill_(causal_mask, float('-inf'))
            lse[b, h0:h1] = torch.logsumexp(attn_w, dim=-1)
            P = torch.softmax(attn_w, dim=-1)
            O[b, h0:h1] = torch.matmul(P, V[b, h0:h1].float()).to(torch.bfloat16)
            del attn_w, P
    dO = torch.randn_like(O)

    return [("dO", dO), ("Q", Q), ("K", K), ("V", V), ("O", O), ("lse", lse)]


def get_outputs(inputs):
    Q = inputs[1][1]
    K = inputs[2][1]
    B, H, S, D = Q.shape
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    dQ = torch.full_like(Q, float('nan'))
    dK = torch.full_like(K, float('nan'))
    dV = torch.full_like(K, float('nan'))
    return [("dQ", dQ), ("dK", dK), ("dV", dV)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    dO = inputs[0][1].float()
    Q = inputs[1][1].float()
    K = inputs[2][1].float()
    V = inputs[3][1].float()
    O = inputs[4][1].float()
    lse = inputs[5][1]

    B, H, S, D = Q.shape

    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * SCALE

    causal_mask = torch.triu(
        torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1
    )
    attn_weights.masked_fill_(causal_mask, float('-inf'))

    P = torch.exp(attn_weights - lse.unsqueeze(-1))

    dsoftmax_sum = (dO * O).sum(dim=-1, keepdim=True)

    dV = torch.matmul(P.transpose(-2, -1), dO).to(torch.bfloat16)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - dsoftmax_sum)
    dQ = (torch.matmul(dS, K) * SCALE).to(torch.bfloat16)
    dK = (torch.matmul(dS.transpose(-2, -1), Q) * SCALE).to(torch.bfloat16)

    return [("dQ", dQ), ("dK", dK), ("dV", dV)]


DESCRIPTION = """\
Implement CUDA kernels for the backward pass of exact softmax attention with causal masking.
The full S x S attention matrix must not be materialized in global memory.

Forward pass being differentiated (per batch b and head h; Sq = Sk = S):
  S = Q @ K^T * scale
  Causal mask: S[i][j] = -inf (P[i][j] = 0) for j > i   (query row i attends keys 0..i)
  P = exp(S - lse[:, None])        (= softmax(S) along the key axis)
  O = P @ V

Given the upstream gradient dO, compute (per batch b and head h):
  Dsum[i] = sum_d dO[i, d] * O[i, d]
  dV = P^T @ dO
  dP = dO @ V^T
  dS = P * (dP - Dsum[:, None])
  dQ = dS @ K * scale
  dK = dS^T @ Q * scale

Inputs (contiguous row-major [B, H, S, D] layout, D fastest):
  Q, K, V, dO: [%d, %d, %d, %d] bf16, i.i.d. standard normal rounded to bf16
  O:   same shape, bf16 — forward output (computed in FP32 from Q, K, V, rounded to bf16)
  lse: [%d, %d, %d] fp32 — natural-log log-sum-exp of each row of S (from the forward pass)

Outputs (bf16): dQ, dK, dV with the same shapes as Q, K, V.

scale = 1/sqrt(D) = %f

Precision: matrix products must use bf16 (or higher-precision) operands with FP32
accumulation; Dsum and the softmax recomputation must be done in FP32. Quantizing any
operand to FP8, INT8 or any lower precision is not allowed.

Correctness: dQ, dK and dV are compared element-wise with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 0.025, rtol = 0.05
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN,
       SCALE)
