"""
Task: Flash Attention 3 Backward — hdim=64, fp16, non-causal

Source: flash-attention/hopper/ (FA3 Hopper SM90 kernels)
Computes dQ, dK, dV given dO and forward pass outputs.
"""
import torch
import math

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 256
SEQ_LEN_K = 256
HEAD_DIM = 64

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_fa3_bwd(
    const void* Q,          // [B, H, Sq, D] fp16
    const void* K,          // [B, H, Sk, D] fp16
    const void* V,          // [B, H, Sk, D] fp16
    const void* O,          // [B, H, Sq, D] fp16
    const void* dO,         // [B, H, Sq, D] fp16
    void* dQ,               // [B, H, Sq, D] fp16
    void* dK,               // [B, H, Sk, D] fp16
    void* dV,               // [B, H, Sk, D] fp16
    const float* lse,       // [B, H, Sq]
    float* dsoftmax_sum,    // [B, H, Sq]
    float* dq_accum,        // [B, H, Sq, D] fp32 accumulator
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

TOLERANCES = {"atol": 5e-2, "rtol": 5e-2}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/flash_attn_v3",
    "reference_sources/flash-attention/csrc/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = ["-gencode", "arch=compute_90a,code=sm_90a"]


def get_inputs():
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                     device='cuda', dtype=torch.float16)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.float16)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.float16)

    # Forward pass to get O and lse
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    attn = torch.matmul(Qf, Kf.transpose(-2, -1)) * SCALE


    lse = torch.logsumexp(attn, dim=-1).float()
    O = torch.matmul(torch.softmax(attn, dim=-1), Vf).to(torch.float16)
    dO = torch.randn_like(O)

    return [("Q", Q), ("K", K), ("V", V), ("O", O), ("dO", dO), ("lse", lse)]


def get_outputs(inputs):
    Q = inputs[0][1]
    K = inputs[1][1]
    B, H, Sq, D = Q.shape
    Sk = K.shape[2]
    dQ = torch.empty(B, H, Sq, D, device='cuda', dtype=Q.dtype)
    dK = torch.empty(B, H, Sk, D, device='cuda', dtype=K.dtype)
    dV = torch.empty(B, H, Sk, D, device='cuda', dtype=K.dtype)
    dsoftmax_sum = torch.empty(B, H, Sq, device='cuda', dtype=torch.float32)
    dq_accum = torch.zeros(B, H, Sq, D, device='cuda', dtype=torch.float32)
    return [("dQ", dQ), ("dK", dK), ("dV", dV), ("dsoftmax_sum", dsoftmax_sum), ("dq_accum", dq_accum)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    Q = inputs[0][1].float()
    K = inputs[1][1].float()
    V = inputs[2][1].float()
    O = inputs[3][1].float()
    dO = inputs[4][1].float()

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)

    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * SCALE


    attn_probs = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_probs, V)

    out.backward(dO)

    dQ = Q.grad.to(torch.float16)
    dK = K.grad.to(torch.float16)
    dV = V.grad.to(torch.float16)

    # dsoftmax_sum = rowwise dot(dO, O)
    dsoftmax_sum = (dO * O).sum(dim=-1).float()
    dq_accum = torch.zeros_like(Q)

    return [("dQ", dQ), ("dK", dK), ("dV", dV), ("dsoftmax_sum", dsoftmax_sum), ("dq_accum", dq_accum)]


DESCRIPTION = """\
Implement a CUDA kernel for Flash Attention 3 backward pass (Hopper SM90).

Given forward outputs (O, lse) and upstream gradient dO, compute:
  dQ, dK, dV — gradients of Q, K, V

scale = 1/sqrt(D) = 0.125000
Non-causal (full attention).
"""
