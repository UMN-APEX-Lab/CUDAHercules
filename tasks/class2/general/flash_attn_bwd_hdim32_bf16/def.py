"""
Task: Flash Attention Backward — hdim=32, bf16, non-causal

Source: flash-attention/csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
Operation: Compute dQ, dK, dV given dO, Q, K, V, O, and saved LSE.
"""
import torch
import math

BATCH_SIZE = 16
NUM_HEADS = 32
SEQ_LEN = 8192
HEAD_DIM = 32

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

TOLERANCES = {"atol": 5e-2, "rtol": 5e-2}

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

    attn_w = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * SCALE


    lse = torch.logsumexp(attn_w, dim=-1).float()
    P = torch.softmax(attn_w, dim=-1)
    O = torch.matmul(P, V.float()).to(torch.bfloat16)
    dO = torch.randn_like(O)

    return [("dO", dO), ("Q", Q), ("K", K), ("V", V), ("O", O), ("lse", lse)]


def get_outputs(inputs):
    Q = inputs[1][1]
    K = inputs[2][1]
    B, H, S, D = Q.shape
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(K)
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

    P = torch.exp(attn_weights - lse.unsqueeze(-1))

    dsoftmax_sum = (dO * O).sum(dim=-1, keepdim=True)

    dV = torch.matmul(P.transpose(-2, -1), dO).to(torch.bfloat16)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - dsoftmax_sum)
    dQ = (torch.matmul(dS, K) * SCALE).to(torch.bfloat16)
    dK = (torch.matmul(dS.transpose(-2, -1), Q) * SCALE).to(torch.bfloat16)

    return [("dQ", dQ), ("dK", dK), ("dV", dV)]


DESCRIPTION = """\
Implement CUDA kernels for the Flash Attention backward pass.

Given upstream gradient dO and saved forward tensors (Q, K, V, O, lse), compute:
  dQ, dK, dV — gradients w.r.t. Q, K, V

Algorithm (tiled, avoids materializing full attention matrix):
  1. Preprocessing: dsoftmax_sum[b,h,i] = sum_d(dO[b,h,i,d] * O[b,h,i,d])
  2. For each K/V tile:
     For each Q tile:
       S = Q_tile @ K_tile^T * scale
       P = exp(S - lse)
       dV += P^T @ dO_tile
       dP = dO_tile @ V_tile^T
       dS = P * (dP - dsoftmax_sum)
       dQ += dS @ K_tile * scale
       dK += dS^T @ Q_tile * scale

Inputs (all bf16, contiguous [B, H, S, D]):
  dO: [%d, %d, %d, %d], Q/K/V/O: same shape
  lse: [%d, %d, %d] fp32

Outputs (bf16): dQ, dK, dV same shape as Q, K, V

scale = 1/sqrt(D) = %f
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN,
       SCALE)
