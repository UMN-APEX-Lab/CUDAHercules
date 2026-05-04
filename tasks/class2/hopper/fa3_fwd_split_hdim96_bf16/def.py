"""
Task: Flash Attention 3 Forward Split-KV — hdim=96, bf16, non-causal

Source: flash-attention/hopper/ (FA3 Hopper SM90 kernels)
Split-KV parallelism with 4 splits.
"""
import torch
import math

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 256
SEQ_LEN_K = 256
HEAD_DIM = 96
NUM_SPLITS = 4

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_fa3_fwd_split(
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
);
"""

SCALAR_ARGS = {
    "B": BATCH_SIZE,
    "H": NUM_HEADS,
    "Sq": SEQ_LEN_Q,
    "Sk": SEQ_LEN_K,
    "D": HEAD_DIM,
    "scale": SCALE,
    "num_splits": NUM_SPLITS,
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
                     device='cuda', dtype=torch.bfloat16)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)
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



    lse = torch.logsumexp(attn_weights, dim=-1)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    O = torch.matmul(attn_probs, V).to(torch.bfloat16)

    return [("O", O), ("lse", lse)]


DESCRIPTION = """\
Implement a CUDA kernel for Flash Attention 3 forward pass with split-KV parallelism.

Same as standard forward but the KV sequence is split across 4 parallel workers.
Each worker computes partial softmax on its chunk, then results are combined.

scale = 1/sqrt(D) = 0.102062
Non-causal (full attention).
"""
