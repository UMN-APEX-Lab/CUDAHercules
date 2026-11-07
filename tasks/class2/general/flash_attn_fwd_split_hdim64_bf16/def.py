"""
Task: Flash Attention Forward Split-KV — hdim=64, bf16, non-causal

Source: flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu
Operation: O = softmax(Q @ K^T * scale) @ V, using split-KV parallelism.
Split-KV splits the K/V sequence across multiple thread block groups,
each computes partial softmax results, then combines.
Shape: Qwen3-32B GQA decode (8 q heads packed into Sq per KV head):
B=16, H=8, Sq=8, Sk=6144.
"""
import torch
import math

BATCH_SIZE = 16
NUM_HEADS = 8
SEQ_LEN_Q = 8
SEQ_LEN_K = 6144
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

# Decode outputs are small (|O| ~ 1/sqrt(Sk) for random inputs) and lse ~ log(Sk),
# so 5e-2 would accept dropping a whole KV split or ignoring the causal mask.
# A correct implementation differs from the reference by <= 1 output ulp; for
# bf16 that is up to ~37% of (1e-3, 5e-3) at |O| ~ 0.07, so bf16 uses rtol 1e-2
# (<= 30%), which still fails a dropped split (~45x), mask errors (~4x) and a
# 1% scale error (~1.7x).
TOLERANCES = {"atol": 1e-3, "rtol": 1e-2}

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
Implement the attention forward pass for one decode step over a long KV cache.

Workload: one batched decode step of a grouped-query-attention LLM
(Qwen3-32B: 64 query heads, 8 KV heads) on one GPU, for 16 sequences with a
6144-token KV cache each. The 8 query heads that share a KV head are packed
into the query-row dimension, so each (b, h) is one KV head with Sq = 8 query rows.
This task uses head dimension D = 64.

For every batch index b and head h (query rows i, key/value rows j):
  S[i][j]    = scale * dot(Q[b,h,i,:], K[b,h,j,:])
  O[b,h,i,:] = sum_j softmax_j(S[i][:]) * V[b,h,j,:]
  lse[b,h,i] = log(sum_j exp(S[i][j]))      (natural logarithm)

Inputs (bf16, contiguous, [B, H, S, D] layout):
  Q: [%d, %d, %d, %d]
  K: [%d, %d, %d, %d]
  V: [%d, %d, %d, %d]
  scale = 1/sqrt(D) = %f
  num_splits = %d: passed by the caller; O and lse must equal attention over the
    full key sequence however the work is partitioned.

Outputs:
  O:   [%d, %d, %d, %d] bf16
  lse: [%d, %d, %d] fp32

Precision: dot products, softmax statistics and the P*V sums must be accumulated
in FP32 (or higher); O is rounded to bf16 once, at the end.

Correctness: on 3 random input sets (Q, K, V ~ N(0, 1)), O and lse are each
compared element-wise with a reference: |out - ref| <= atol + rtol * |ref|,
atol = %g, rtol = %g.
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       SCALE, NUM_SPLITS,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q,
       TOLERANCES["atol"], TOLERANCES["rtol"])
