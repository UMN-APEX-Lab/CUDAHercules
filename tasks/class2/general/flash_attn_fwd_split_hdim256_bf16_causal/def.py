"""
Task: Flash Attention Forward Split-KV — hdim=256, bf16, causal

Source: flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu
Operation: O = softmax(Q @ K^T * scale) @ V, using split-KV parallelism.
Split-KV splits the K/V sequence across multiple thread block groups,
each computes partial softmax results, then combines.
Shape: speculative-decoding verification (MHA, causal):
B=8, H=20, Sq=8, Sk=8192.
"""
import torch
import math

BATCH_SIZE = 8
NUM_HEADS = 20
SEQ_LEN_Q = 8
SEQ_LEN_K = 8192
HEAD_DIM = 256
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

    causal_mask = torch.triu(
        torch.ones(Sq, Sk, device=Q.device, dtype=torch.bool), diagonal=Sk - Sq + 1
    )
    attn_weights.masked_fill_(causal_mask, float('-inf'))

    lse = torch.logsumexp(attn_weights, dim=-1)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    O = torch.matmul(attn_probs, V).to(torch.bfloat16)

    return [("O", O), ("lse", lse)]


DESCRIPTION = """\
Implement the causal attention forward pass for a speculative-decoding verification step.

Workload: speculative-decoding verification on a multi-head-attention LLM
(40 heads of dimension 128, as in Llama-2-13B) on one GPU: each of the
8 sequences checks Sq = 8 draft tokens, which are the last 8
positions of its 8192-token KV cache. This task uses head dimension
D = 256 with H = 20 heads.

For every batch index b and head h (query rows i, key/value rows j):
  S[i][j]    = scale * dot(Q[b,h,i,:], K[b,h,j,:])
  S[i][j]    = -inf where j > i + (Sk - Sq)   (causal, bottom-right aligned)
  O[b,h,i,:] = sum_j softmax_j(S[i][:]) * V[b,h,j,:]
  lse[b,h,i] = log(sum_j exp(S[i][j]))      (natural logarithm)

Causal mask (bottom-right aligned): query row i is the token at absolute
position i + (Sk - Sq) and attends to keys j <= i + (Sk - Sq); the last query
row attends to all Sk keys.

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
