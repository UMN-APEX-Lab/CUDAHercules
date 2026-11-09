"""
Task: LayerNorm Forward — hidden_size=4096, BF16 activations, FP32 compute

Source: flash-attention/csrc/layer_norm/ln_fwd_kernels.cuh
Operation: z = bf16((x - mu) / sqrt(var + eps) * gamma + beta)
"""
import math

import torch

ROWS = 32768
COLS = 4096
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

TOLERANCES = {"atol": 0.00025, "rtol": 0.024,
              "per_output": {"mu": {"atol": 0.0002, "rtol": 2e-05},
                             "rs": {"atol": 0.0, "rtol": 0.0005}}}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]

# Realistic activation rows (see "Inputs" in DESCRIPTION). Row classes sit at fixed row
# indices, so every input set has the same mix:
#   i % 8 == 3      mean-offset rows c_i + s_i * N(0, 1), |c_i| / s_i = 10^U(1, log10(50)),
#                   random sign: a mean that is large relative to the spread (outlier tokens);
#                   E[x^2] - mu^2 cancels badly on them and BF16 quantizes their spread.
#   i % 512 == 100  constant rows: every element is c_i ~ N(0, 1) (variance 0, rs = 1/sqrt(eps)).
#   i % 512 == 356  near-constant rows c_i + t_i * N(0, 1), c_i ~ N(0, 1), t_i = 10^U(-5, -3)
#                   (variance << eps once rounded to BF16).
#   other rows      zero-mean rows s_i * N(0, 1).
# s_i = 10^U(-2, 1) per row: rows with a small variance make eps matter.
OFFSET_RATIO_MAX = 50.0


def _row_scale_and_classes():
    i = torch.arange(ROWS, device='cuda')
    s = torch.pow(10.0, torch.empty(ROWS, 1, device='cuda').uniform_(-2.0, 1.0))
    return s, i[i % 8 == 3], i[i % 512 == 100], i[i % 512 == 356]


def _add_mean_offset(a, s, off):
    ratio = torch.pow(10.0, torch.empty(off.numel(), 1, device='cuda').uniform_(
        1.0, math.log10(OFFSET_RATIO_MAX)))
    sign = torch.randint(0, 2, (off.numel(), 1), device='cuda').float() * 2.0 - 1.0
    a[off] += sign * ratio * s[off]


def _activation():
    """BF16 activation x [ROWS, COLS] with the row classes above."""
    s, off, const, near = _row_scale_and_classes()
    a = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * s
    _add_mean_offset(a, s, off)
    a[const] = torch.randn(const.numel(), 1, device='cuda').expand(-1, COLS)
    t = torch.pow(10.0, torch.empty(near.numel(), 1, device='cuda').uniform_(-5.0, -3.0))
    a[near] = (torch.randn(near.numel(), 1, device='cuda')
               + t * torch.randn(near.numel(), COLS, device='cuda'))
    return a.to(torch.bfloat16)


def get_inputs():
    x = _activation()
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

Scenario: LayerNorm forward in 7-8B-parameter LLM training / prefill (hidden size 4096):
32768 tokens per GPU per step (e.g. 8 sequences x 4096 tokens).

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
Inputs: Every row i has a scale s_i = 10^U(-2, 1). One row in every 8 is a mean-offset row,
x[i, :] = bf16(c_i + s_i * N(0, 1)) with |c_i| / s_i = 10^U(1, log10(50)) and a random
sign (a mean that is large relative to the spread, as on outlier tokens; BF16 rounding
quantizes the spread of these rows). One row in every 512 is constant,
x[i, :] = bf16(c_i) with c_i ~ N(0, 1) (variance 0, so rs = 1/sqrt(eps)), and one row in
every 512 is near-constant, x[i, :] = bf16(c_i + t_i * N(0, 1)) with c_i ~ N(0, 1) and
t_i = 10^U(-5, -3) (variance << eps). All other rows are zero-mean,
x[i, :] = bf16(s_i * N(0, 1)). The row classes sit at the same row indices in every
input set.
So var[i] ranges from 0 to ~1e2 and eps matters for the small rows. gamma and beta are
i.i.d. N(0, 1).
Outputs: z [rows, cols] (bf16), mu [rows] (fp32), rs [rows] (fp32).

Precision: BF16 is only the storage format of the activations. All arithmetic
(sums, mean, variance, normalization, scale and shift) must be done in FP32 (or
higher) on values converted from BF16, and each BF16 output must come from a single
round-to-nearest-even conversion of its final FP32 value. FP16/BF16/FP8 arithmetic,
tensor-core MMAs, or holding intermediates in a lower precision are not allowed,
even if the result happens to pass the tolerance check.

Correctness: z, mu and rs are compared element-wise (in FP32) with a reference
implementation over 3 random input sets, |out - ref| <= atol + rtol * |ref| with
  z: atol = 0.00025, rtol = 0.024
  mu: atol = 0.0002, rtol = 2e-05
  rs: atol = 0, rtol = 0.0005
mu and rs are FP32 outputs and must be FP32-accurate. For z, rtol allows about 3 BF16 ulps of rounding difference; atol is small compared with |z| ~ 1,
so z values that nearly cancel to zero (where (xf - mu) * rs * gamma ~ -beta) must be
accurate to well below BF16 resolution. NaN/Inf fail, and every output element must be
written (output buffers are pre-filled with NaN).
""" % (ROWS, COLS, EPS)
