"""
Task: LayerNorm Forward — hidden_size=8192, BF16 activations, FP32 compute

Source: flash-attention/csrc/layer_norm/ln_fwd_kernels.cuh
Operation: z = bf16((x - mu) / sqrt(var + eps) * gamma + beta)
"""
import torch

ROWS = 16384
COLS = 8192
EPS = 1e-5

FUNCTION_SIGNATURE = """\
void launch_layernorm_forward(
    const void* x,        // [rows, cols] bf16 input
    const float* gamma,   // [cols] fp32 scale weight
    const float* beta,    // [cols] fp32 bias weight
    void* z,              // [rows, cols] bf16 normalized output
    float* mu,            // [rows] fp32 per-row mean
    float* rs,            // [rows] fp32 per-row reciprocal std
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS, "eps": EPS}

TOLERANCES = {"atol": 4e-05, "rtol": 0.024}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    # Per-row scale 10^U(-2, 1): rows with a small variance make eps matter (a kernel
    # that drops eps, or uses the unbiased variance, fails on them).
    row_scale = torch.pow(10.0, torch.empty(ROWS, 1, device='cuda').uniform_(-2.0, 1.0))
    x = (torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * row_scale).to(torch.bfloat16)
    gamma = torch.randn(COLS, device='cuda', dtype=torch.float32)
    beta = torch.randn(COLS, device='cuda', dtype=torch.float32)
    return [("x", x), ("gamma", gamma), ("beta", beta)]


def get_outputs(inputs):
    x = inputs[0][1]
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    z = torch.full_like(x, float('nan'))
    mu = torch.full((ROWS,), float('nan'), device='cuda', dtype=torch.float32)
    rs = torch.full((ROWS,), float('nan'), device='cuda', dtype=torch.float32)
    return [("z", z), ("mu", mu), ("rs", rs)]


def reference_fn(inputs):
    """PyTorch reference: FP32 compute on the BF16 input, BF16 output."""
    x = inputs[0][1].float()
    gamma = inputs[1][1]
    beta = inputs[2][1]
    mu = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    rs = torch.rsqrt(var + EPS)
    z = ((x - mu.unsqueeze(-1)) * rs.unsqueeze(-1) * gamma + beta).to(torch.bfloat16)
    return [("z", z), ("mu", mu), ("rs", rs)]


DESCRIPTION = """\
Implement a CUDA kernel for the LayerNorm forward pass on BF16 activations (BF16 input
and output; FP32 weights, FP32 statistics, FP32 compute).

Scenario: LayerNorm forward in 70B-parameter LLM training with tensor parallelism
(hidden size 8192; LayerNorm is not sharded and normalizes the full hidden vector of
each token): 16384 tokens per GPU per step (e.g. 4 sequences x 4096 tokens).

Given x of shape (rows, cols) and gamma, beta of shape (cols,), with xf = float(x):
  mu[i]   = (1/cols) * sum_j xf[i, j]
  var[i]  = (1/cols) * sum_j (xf[i, j] - mu[i])^2      (biased variance)
  rs[i]   = 1 / sqrt(var[i] + eps)
  z[i, j] = bf16( (xf[i, j] - mu[i]) * rs[i] * gamma[j] + beta[j] )
where bf16() is the round-to-nearest-even conversion.

Dimensions: rows=%d, cols=%d, eps=%g. All tensors are contiguous and row-major:
  x, z          bfloat16 (__nv_bfloat16), passed as void pointers
  gamma, beta   float32
  mu, rs        float32
Inputs: x[i, :] = bf16(s_i * N(0, 1)) with a per-row scale s_i = 10^U(-2, 1) (so var[i]
ranges over ~1e-4 .. 1e2 and eps matters for the small rows); gamma and beta are
i.i.d. N(0, 1).
Outputs: z [rows, cols] (bf16), mu [rows] (fp32), rs [rows] (fp32).

Precision: BF16 is only the storage format of the activations. All arithmetic
(sums, mean, variance, normalization, scale and shift) must be done in FP32 (or
higher) on values converted from BF16, and each BF16 output must come from a single
round-to-nearest-even conversion of its final FP32 value. FP16/BF16/FP8 arithmetic,
tensor-core MMAs, or holding intermediates in a lower precision are not allowed,
even if the result happens to pass the tolerance check.

Correctness: z, mu and rs are compared element-wise (in FP32) with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 4e-05, rtol = 0.024
rtol allows about 3 BF16 ulps of rounding difference; atol is small, so z values that
nearly cancel to zero (where (xf - mu) * rs * gamma ~ -beta) must be accurate to FP32
level. NaN/Inf fail, and every output element must be written (output buffers are pre-
filled with NaN).
""" % (ROWS, COLS, EPS)
