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

# Calibrated against FP64 truth on the LLM-like test inputs: atol/rtol are mirrored in
# task.yaml correctness.tolerances; per_output and max_normalized_rms live only here
# (eval.check_correctness reads them from def.py).
TOLERANCES = {"atol": 0.15, "rtol": 0.02, "max_normalized_rms": 0.007,
              "per_output": {"dK": {"atol": 0.5, "rtol": 0.02, "max_normalized_rms": 0.007},
                             "dV": {"atol": 0.8, "rtol": 0.02, "max_normalized_rms": 0.006}}}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/flash_attn",
    "reference_sources/flash-attention/csrc/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = ["-gencode", "arch=compute_80,code=compute_80"]


def get_inputs():
    Q, K = _structured_qk(BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN, HEAD_DIM, torch.bfloat16)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                     device='cuda', dtype=torch.bfloat16)

    # Forward O and lse in FP32, a few heads at a time: the full [B, H, S, S] score
    # tensor would need B*H*S*S*4 bytes (64-128 GB for the hdim32/64 shapes).
    O = torch.empty_like(Q)
    lse = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, device=Q.device, dtype=torch.float32)

    for b in range(BATCH_SIZE):
        for h0 in range(0, NUM_HEADS, 4):
            h1 = min(h0 + 4, NUM_HEADS)
            attn_w = torch.matmul(Q[b, h0:h1].float(), K[b, h0:h1].float().transpose(-2, -1)) * SCALE

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

    P = torch.exp(attn_weights - lse.unsqueeze(-1))

    dsoftmax_sum = (dO * O).sum(dim=-1, keepdim=True)

    dV = torch.matmul(P.transpose(-2, -1), dO).to(torch.bfloat16)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - dsoftmax_sum)
    dQ = (torch.matmul(dS, K) * SCALE).to(torch.bfloat16)
    dK = (torch.matmul(dS.transpose(-2, -1), Q) * SCALE).to(torch.bfloat16)

    return [("dQ", dQ), ("dK", dK), ("dV", dV)]


DESCRIPTION = """\
Implement CUDA kernels for the backward pass of exact softmax attention.
The full S x S attention matrix must not be materialized in global memory.

Forward pass being differentiated (per batch b and head h; Sq = Sk = S):
  S = Q @ K^T * scale
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
  Q, K, V, dO: [%d, %d, %d, %d] bf16
  O:   same shape, bf16 — forward output (computed in FP32 from Q, K, V, rounded to bf16)
  lse: [%d, %d, %d] fp32 — natural-log log-sum-exp of each row of S (from the forward pass)
Input distribution (each of the 3 input sets is a fresh random draw; it mimics the
attention logits of a trained LLM). V and dO are i.i.d. N(0, 1), rounded to bf16. Q and K are
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

Outputs (bf16): dQ, dK, dV with the same shapes as Q, K, V.

scale = 1/sqrt(D) = %f

Precision: matrix products must use bf16 (or higher-precision) operands with FP32
accumulation; Dsum and the softmax recomputation must be done in FP32. Quantizing any
operand to FP8, INT8 or any lower precision is not allowed.

Correctness: dQ, dK and dV are compared with a reference implementation over
3 random input sets; each output must pass both checks
  element-wise:   |out - ref| <= atol + rtol * |ref|
  normalized RMS: sqrt(mean((out - ref)^2)) / sqrt(mean(ref^2)) <= max_rms
with
  dQ: atol = 0.15, rtol = 0.02, max_rms = 0.007
  dK: atol = 0.5, rtol = 0.02, max_rms = 0.007
  dV: atol = 0.8, rtol = 0.02, max_rms = 0.006
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
       BATCH_SIZE, NUM_HEADS, SEQ_LEN,
       SCALE)
