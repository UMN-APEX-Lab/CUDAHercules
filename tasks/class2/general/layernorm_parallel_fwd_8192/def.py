"""
Task: LayerNorm Parallel Residual Forward — hidden_size=8192, BF16 activations, FP32 compute

Source: flash-attention/csrc/layer_norm/ln_parallel_residual_fwd_kernels.cuh
Operation: Fused residual addition + layer normalization.
  xf = x0 + residual (FP32), x = bf16(xf)
  z = bf16(LayerNorm(xf, gamma, beta))
"""
import math

import torch

ROWS = 16384
COLS = 8192
EPS = 1e-5

FUNCTION_SIGNATURE = """\
void launch_layernorm_parallel_fwd(
    const void* x0,        // [rows, cols] bf16 new activation
    const void* residual,  // [rows, cols] bf16 residual stream
    const float* gamma,    // [cols] fp32 scale weight
    const float* beta,     // [cols] fp32 bias weight
    void* z,               // [rows, cols] bf16 normalized output
    void* x,               // [rows, cols] bf16 x0 + residual (saved for backward)
    float* mu,             // [rows] fp32 per-row mean
    float* rs,             // [rows] fp32 per-row reciprocal std
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS, "eps": EPS}

TOLERANCES = {"atol": 0.0008, "rtol": 0.024,
              "per_output": {"mu": {"atol": 0.0005, "rtol": 2e-05},
                             "rs": {"atol": 0.0, "rtol": 0.0015}}}  # mirrored in task.yaml correctness.tolerances

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


def _activation_pair():
    """BF16 (x0, residual) [ROWS, COLS]: x0 = s_i * N(0, 1) is the block output, the row
    classes above live in the residual stream; on constant and near-constant rows x0 = 0
    (so x = residual is exactly the BF16 row of the class)."""
    s, off, const, near = _row_scale_and_classes()
    x0 = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * s
    res = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * s
    _add_mean_offset(res, s, off)
    x0[const] = 0.0
    res[const] = torch.randn(const.numel(), 1, device='cuda').expand(-1, COLS)
    t = torch.pow(10.0, torch.empty(near.numel(), 1, device='cuda').uniform_(-5.0, -3.0))
    x0[near] = 0.0
    res[near] = (torch.randn(near.numel(), 1, device='cuda')
                 + t * torch.randn(near.numel(), COLS, device='cuda'))
    return x0.to(torch.bfloat16), res.to(torch.bfloat16)


def get_inputs():
    x0, residual = _activation_pair()
    gamma = torch.randn(COLS, device='cuda', dtype=torch.float32)
    beta = torch.randn(COLS, device='cuda', dtype=torch.float32)
    return [("x0", x0), ("residual", residual), ("gamma", gamma), ("beta", beta)]


def get_outputs(inputs):
    x0 = inputs[0][1]
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    z = torch.full_like(x0, float('nan'))
    x = torch.full_like(x0, float('nan'))
    mu = torch.full((ROWS,), float('nan'), device='cuda', dtype=torch.float32)
    rs = torch.full((ROWS,), float('nan'), device='cuda', dtype=torch.float32)
    return [("z", z), ("x", x), ("mu", mu), ("rs", rs)]


def reference_fn(inputs):
    """PyTorch reference: FP32 compute on the BF16 inputs, BF16 z and x."""
    x0 = inputs[0][1]
    residual = inputs[1][1]
    gamma = inputs[2][1]
    beta = inputs[3][1]

    xf = x0.float() + residual.float()
    mu = xf.mean(dim=-1)
    var = xf.var(dim=-1, unbiased=False)
    rs = torch.rsqrt(var + EPS)
    z = ((xf - mu.unsqueeze(-1)) * rs.unsqueeze(-1) * gamma + beta).to(torch.bfloat16)

    return [("z", z), ("x", xf.to(torch.bfloat16)), ("mu", mu), ("rs", rs)]


DESCRIPTION = """\
Implement a CUDA kernel for the fused residual addition + LayerNorm forward pass on
BF16 activations (BF16 inputs and outputs; FP32 weights, FP32 statistics, FP32 compute).

Scenario: fused residual add + LayerNorm forward of a pre-norm block in 70B-parameter
LLM training with tensor parallelism (hidden size 8192; LayerNorm is not sharded and
normalizes the full hidden vector of each token): 16384 tokens per GPU per step (e.g. 4
sequences x 4096 tokens).

Given x0 and residual of shape (rows, cols) and gamma, beta of shape (cols,):
  xf[i,j] = float(x0[i,j]) + float(residual[i,j])      (FP32 sum)
  x[i,j]  = bf16(xf[i,j])                              (residual stream, saved for backward)
  mu[i]   = (1/cols) * sum_j xf[i, j]
  var[i]  = (1/cols) * sum_j (xf[i, j] - mu[i])^2      (biased variance)
  rs[i]   = 1 / sqrt(var[i] + eps)
  z[i,j]  = bf16( (xf[i,j] - mu[i]) * rs[i] * gamma[j] + beta[j] )
where bf16() is the round-to-nearest-even conversion. mu, var and z are computed from the
FP32 sum xf, not from the rounded x.

Dimensions: rows=%d, cols=%d, eps=%g. All tensors are contiguous and row-major:
  x0, residual, z, x    bfloat16 (__nv_bfloat16), passed as void pointers
  gamma, beta           float32
  mu, rs                float32
Inputs: x0 is the block output and residual the residual stream; every row i has a scale
s_i = 10^U(-2, 1). One row in every 8 is a mean-offset row: x0[i, :] = bf16(s_i * N(0, 1))
and residual[i, :] = bf16(c_i + s_i * N(0, 1)) with |c_i| / s_i = 10^U(1, log10(50)) and
a random sign (a mean that is large relative to the spread, as on outlier tokens; BF16
rounding quantizes the spread of these rows). One row in every 512 is constant:
x0[i, :] = 0 and residual[i, :] = bf16(c_i) with c_i ~ N(0, 1) (variance 0, so
rs = 1/sqrt(eps)). One row in every 512 is near-constant: x0[i, :] = 0 and
residual[i, :] = bf16(c_i + t_i * N(0, 1)) with c_i ~ N(0, 1) and t_i = 10^U(-5, -3)
(variance << eps).
All other rows have x0[i, :] and residual[i, :] = bf16(s_i * N(0, 1)). The row classes sit
at the same row indices in every input set.
So var[i] ranges from ~0 to ~2e2 and eps matters for the small rows. gamma and beta are
i.i.d. N(0, 1).
Outputs: z [rows, cols] (bf16), x [rows, cols] (bf16), mu [rows] (fp32), rs [rows] (fp32).

Precision: BF16 is only the storage format of the activations. All arithmetic
(sums, mean, variance, normalization, scale and shift) must be done in FP32 (or
higher) on values converted from BF16, and each BF16 output must come from a single
round-to-nearest-even conversion of its final FP32 value. FP16/BF16/FP8 arithmetic,
tensor-core MMAs, or holding intermediates in a lower precision are not allowed,
even if the result happens to pass the tolerance check.

Correctness: z, x, mu and rs are compared element-wise (in FP32) with a reference
implementation over 3 random input sets, |out - ref| <= atol + rtol * |ref| with
  z and x: atol = 0.0008, rtol = 0.024
  mu: atol = 0.0005, rtol = 2e-05
  rs: atol = 0, rtol = 0.0015
mu and rs are FP32 outputs and must be FP32-accurate. For z and x, rtol allows about 3 BF16 ulps of rounding difference; atol is small compared with |z| ~ 1,
so z values that nearly cancel to zero (where (xf - mu) * rs * gamma ~ -beta) must be
accurate to well below BF16 resolution. NaN/Inf fail, and every output element must be
written (output buffers are pre-filled with NaN).
""" % (ROWS, COLS, EPS)
