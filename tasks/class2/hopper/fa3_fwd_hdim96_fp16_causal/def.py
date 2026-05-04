"""
Task: Flash Attention 3 Forward — hdim=96, fp16, causal

Source: flash-attention/hopper/ (FA3 Hopper SM90 kernels)
Operation: O = softmax(Q @ K^T * scale) @ V
Also outputs log-sum-exp (LSE) per query row.
"""
import torch
import math

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 256
SEQ_LEN_K = 256
HEAD_DIM = 96

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_fa3_fwd(
    const void* Q,     // [B, H, Sq, D] fp16
    const void* K,     // [B, H, Sk, D] fp16
    const void* V,     // [B, H, Sk, D] fp16
    void* O,           // [B, H, Sq, D] fp16
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
    return [("Q", Q), ("K", K), ("V", V)]


def get_outputs(inputs):
    Q = inputs[0][1]
    B, H, Sq, D = Q.shape
    O = torch.empty(B, H, Sq, D, device='cuda', dtype=Q.dtype)
    lse = torch.empty(B, H, Sq, device='cuda', dtype=torch.float32)
    return [("O", O), ("lse", lse)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    Q = inputs[0][1].float()
    K = inputs[1][1].float()
    V = inputs[2][1].float()

    B, H, Sq, D = Q.shape
    Sk = K.shape[2]

    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * SCALE
    # Causal mask
    causal_mask = torch.triu(torch.full((Sq, Sk), float("-inf"), device=Q.device), diagonal=1)
    attn_weights = attn_weights + causal_mask

    lse = torch.logsumexp(attn_weights, dim=-1)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    O = torch.matmul(attn_probs, V).to(torch.float16)

    return [("O", O), ("lse", lse)]


DESCRIPTION = """\
Implement a CUDA kernel for Flash Attention 3 forward pass (Hopper SM90).

Compute scaled dot-product attention WITHOUT materializing the full attention matrix:
  S = Q @ K^T * scale
  Apply causal mask (upper triangular = -inf)
  O = softmax(S) @ V
  lse[i] = log(sum(exp(S[i][:])))

Inputs (all fp16, contiguous, [B, H, S, D] layout):
  Q: [4, 8, 256, 96]
  K: [4, 8, 256, 96]
  V: [4, 8, 256, 96]

Outputs:
  O:   [B, H, Sq, D] fp16 — attention output
  lse: [B, H, Sq] fp32 — log-sum-exp per query row

scale = 1/sqrt(D) = 0.102062
Causal: upper triangular of attention matrix is masked to -inf.
"""
