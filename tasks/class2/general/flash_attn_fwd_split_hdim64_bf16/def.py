"""
Task: Flash Attention Forward Split-KV — hdim=64, bf16, non-causal

Source: flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu
Operation: O = softmax(Q @ K^T * scale) @ V, using split-KV parallelism.
Split-KV splits the K/V sequence across multiple thread block groups,
each computes partial softmax results, then combines.
"""
import torch
import math

BATCH_SIZE = 8
NUM_HEADS = 32
SEQ_LEN_Q = 8192
SEQ_LEN_K = 8192
HEAD_DIM = 64
NUM_SPLITS = 4

SCALE = 1.0 / math.sqrt(HEAD_DIM)

FUNCTION_SIGNATURE = """\
void launch_flash_attn_fwd_split(
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
    O = torch.empty(B, H, Sq, D, device='cuda', dtype=torch.bfloat16)
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
Implement a CUDA kernel for the forward pass of Flash Attention using split-KV parallelism.

Split-KV approach: split the K/V sequence into num_splits chunks, compute partial
softmax results independently per chunk, then combine using log-sum-exp correction.

  S = Q @ K^T * scale
  O = softmax(S) @ V
  lse[i] = log(sum(exp(S[i][:])))

Inputs (all bf16, contiguous, [B, H, S, D] layout):
  Q: [%d, %d, %d, %d]
  K: [%d, %d, %d, %d]
  V: [%d, %d, %d, %d]
  num_splits: %d

Outputs:
  O:   [B, H, Sq, D] bf16 — attention output
  lse: [B, H, Sq] fp32 — log-sum-exp per query row

scale = 1/sqrt(D) = %f
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       NUM_SPLITS, SCALE)
