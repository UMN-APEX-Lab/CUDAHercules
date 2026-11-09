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

# Test-input distribution: Q and K are built so that the attention logits
# S = scale * Q K^T look like those of a trained LLM (see DESCRIPTION):
#   ordinary logits ~ N(0, (LOGIT_STD * r_i)^2), r_i = exp(QNORM_LOGSTD * clip(N(0,1), -3, 3))
#   key 0 is an attention sink: +SINK_LOGIT for every query
#   the first NEG_Q query rows x the first NEG_K keys: -NEG_LOGIT (very negative logits)
LOGIT_STD = 3.6
QNORM_LOGSTD = 0.3
SINK_LOGIT = 16.0
NEG_LOGIT = 120.0
NEG_K = 40


def _structured_qk(B, H, Sq, Sk, D, dtype):
    dev = 'cuda'
    d = torch.arange(D, device=dev)
    u = torch.where(d % 2 == 0, 1.0, -1.0) / math.sqrt(D)         # sink direction
    w = torch.where((d // 2) % 2 == 0, 1.0, -1.0) / math.sqrt(D)  # negative-block direction (u . w = 0)

    def rand_perp(*shape):
        # N(0, I) projected orthogonal to u and w, rescaled so the logit std is exactly LOGIT_STD.
        z = torch.randn(*shape, D, device=dev)
        z -= (z @ u)[..., None] * u
        z -= (z @ w)[..., None] * w
        return z.mul_((D / (D - 2)) ** 0.25 * math.sqrt(LOGIT_STD))

    scale = 1.0 / math.sqrt(D)
    m = math.sqrt(SINK_LOGIT / scale)
    n = math.sqrt(NEG_LOGIT / scale)
    neg_q = min(NEG_K, Sq // 2)
    r = torch.exp(QNORM_LOGSTD * torch.randn(B, H, Sq, 1, device=dev).clamp_(-3.0, 3.0))
    q = rand_perp(B, H, Sq).mul_(r).add_(m * u)
    q[:, :, :neg_q] += n * w
    q = q.to(dtype)
    k = rand_perp(B, H, Sk)
    k[:, :, 0] += m * u
    k[:, :, :NEG_K] -= n * w
    return q, k.to(dtype)

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
# Calibrated against FP64 truth on the LLM-like test inputs: atol/rtol are mirrored in
# task.yaml correctness.tolerances; per_output and max_normalized_rms live only here
# (eval.check_correctness reads them from def.py).
TOLERANCES = {"atol": 0.004, "rtol": 0.01, "max_normalized_rms": 0.004,
              "per_output": {"lse": {"atol": 0.0004, "rtol": 0, "max_normalized_rms": 3e-06}}}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/flash_attn",
    "reference_sources/flash-attention/csrc/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = ["-gencode", "arch=compute_80,code=compute_80"]


def get_inputs():
    Q, K = _structured_qk(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_K, HEAD_DIM, torch.bfloat16)
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

Input distribution (each of the 3 input sets is a fresh random draw; it mimics the
attention logits of a trained LLM). V is i.i.d. N(0, 1), rounded to bf16. Q and K are
constructed so that the logits S[i][j] = scale * dot(Q[i], K[j]) have this structure:
  - ordinary logits are ~ N(0, (3.6 * r_i)^2) with a per-query norm factor
    r_i = exp(0.3 * clip(N(0, 1), -3, 3)) (overall logit std ~4);
  - key 0 is an attention sink: its logit is +16 higher, for every query;
  - the first min(40, Sq/2) query rows have logits shifted by -120 against the first
    40 keys (very negative logits).
  Construction: Q[i] = r_i*a*z_i + m*u (+ n*w for the first min(40, Sq/2) rows),
  K[j] = a*y_j (+ m*u for j = 0) (- n*w for j < 40), with z_i, y_j ~ N(0, I) projected
  orthogonal to two fixed orthonormal vectors u, w, and a, m, n set to give the logit
  statistics above; Q and K are then rounded to bf16.

Outputs:
  O:   [%d, %d, %d, %d] bf16
  lse: [%d, %d, %d] fp32

Precision: dot products, softmax statistics and the P*V sums must be accumulated
in FP32 (or higher); O is rounded to bf16 once, at the end.

Correctness: O and lse are compared with a reference implementation over
3 random input sets; each output must pass both checks
  element-wise:   |out - ref| <= atol + rtol * |ref|
  normalized RMS: sqrt(mean((out - ref)^2)) / sqrt(mean(ref^2)) <= max_rms
with
  O:   atol = 0.004, rtol = 0.01, max_rms = 0.004
  lse: atol = 0.0004, rtol = 0, max_rms = 3e-06
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_K, HEAD_DIM,
       SCALE, NUM_SPLITS,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q)
