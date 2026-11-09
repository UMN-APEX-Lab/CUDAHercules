"""
Task: LayerNorm Backward — hidden_size=1024, BF16 activations/gradients, FP32 compute

Source: flash-attention/csrc/layer_norm/ln_bwd_kernels.cuh
Operation: Compute dx (bf16), dgamma, dbeta (fp32) given upstream gradient dz (bf16),
saved x (bf16), mu, rs (fp32) and gamma (fp32).
"""
import math

import torch

ROWS = 131072
COLS = 1024

FUNCTION_SIGNATURE = """\
void launch_layernorm_backward(
    const void* dz,       // [rows, cols] bf16 upstream gradient
    const void* x,        // [rows, cols] bf16 saved input from forward
    const float* mu,      // [rows] fp32 saved mean
    const float* rs,      // [rows] fp32 saved reciprocal std
    const float* gamma,   // [cols] fp32 scale weight
    void* dx,             // [rows, cols] bf16 gradient w.r.t. input
    float* dgamma,        // [cols] fp32 gradient w.r.t. gamma
    float* dbeta,         // [cols] fp32 gradient w.r.t. beta
    int rows,
    int cols,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS}

TOLERANCES = {"atol": 0.001, "rtol": 0.024}  # mirrored in task.yaml correctness.tolerances

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
    # rs spans ~0.1 .. 316 (1/sqrt(eps) on the constant rows), so dx (proportional to rs)
    # covers a wide range and a missing ds/db term or BF16 arithmetic shows up on the
    # high-rs rows. mu and rs are what the forward pass saves: the FP32 statistics of x.
    x = _activation()
    xf = x.float()
    mu = xf.mean(dim=-1)
    rs = torch.rsqrt(xf.var(dim=-1, unbiased=False) + 1e-5)
    del xf
    gamma = torch.randn(COLS, device='cuda', dtype=torch.float32)
    dz = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32).to(torch.bfloat16)
    return [("dz", dz), ("x", x), ("mu", mu), ("rs", rs), ("gamma", gamma)]


def get_outputs(inputs):
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    dx = torch.full((ROWS, COLS), float('nan'), device='cuda', dtype=torch.bfloat16)
    dgamma = torch.full((COLS,), float('nan'), device='cuda', dtype=torch.float32)
    dbeta = torch.full((COLS,), float('nan'), device='cuda', dtype=torch.float32)
    return [("dx", dx), ("dgamma", dgamma), ("dbeta", dbeta)]


def reference_fn(inputs):
    """PyTorch reference: FP32 compute on the BF16 inputs, BF16 dx."""
    dz = inputs[0][1].float()
    x = inputs[1][1].float()
    mu = inputs[2][1]
    rs = inputs[3][1]
    gamma = inputs[4][1]

    x_hat = (x - mu.unsqueeze(-1)) * rs.unsqueeze(-1)
    dgamma = (dz * x_hat).sum(dim=0)
    dbeta = dz.sum(dim=0)

    g = dz * gamma
    ds = (g * x_hat).sum(dim=-1, keepdim=True)
    db = g.sum(dim=-1, keepdim=True)
    dx = (rs.unsqueeze(-1) * (g - (ds * x_hat + db) / COLS)).to(torch.bfloat16)

    return [("dx", dx), ("dgamma", dgamma), ("dbeta", dbeta)]


DESCRIPTION = """\
Implement a CUDA kernel for the LayerNorm backward pass on BF16 activations and
gradients (BF16 dz, x and dx; FP32 weights, statistics, weight gradients and compute).

Scenario: LayerNorm backward in BERT-large training (hidden size 1024): 131072 tokens
per step (256 sequences x 512 tokens).

Given the upstream gradient dz [rows, cols], the saved forward input x [rows, cols], its
saved statistics mu [rows] and rs [rows] (rs = 1/sqrt(var + 1e-5), biased variance) and
gamma [cols], with xf = float(x) and dzf = float(dz):
  x_hat[i,j] = (xf[i,j] - mu[i]) * rs[i]
  g[i,j]     = dzf[i,j] * gamma[j]
  ds[i]      = sum_j g[i,j] * x_hat[i,j]
  db[i]      = sum_j g[i,j]
  dx[i,j]    = bf16( rs[i] * (g[i,j] - (ds[i] * x_hat[i,j] + db[i]) / cols) )
  dgamma[j]  = sum_i dzf[i,j] * x_hat[i,j]
  dbeta[j]   = sum_i dzf[i,j]
where bf16() is the round-to-nearest-even conversion.

Dimensions: rows=%d, cols=%d. All tensors are contiguous and row-major:
  dz, x, dx            bfloat16 (__nv_bfloat16), passed as void pointers
  mu, rs, gamma        float32
  dgamma, dbeta        float32
Inputs: x is the forward activation.
Every row i has a scale s_i = 10^U(-2, 1). One row in every 8 is a mean-offset row,
x[i, :] = bf16(c_i + s_i * N(0, 1)) with |c_i| / s_i = 10^U(1, log10(50)) and a random
sign (a mean that is large relative to the spread, as on outlier tokens; BF16 rounding
quantizes the spread of these rows). One row in every 512 is constant,
x[i, :] = bf16(c_i) with c_i ~ N(0, 1) (variance 0, so rs = 1/sqrt(eps)), and one row in
every 512 is near-constant, x[i, :] = bf16(c_i + t_i * N(0, 1)) with c_i ~ N(0, 1) and
t_i = 10^U(-5, -3) (variance << eps). All other rows are zero-mean,
x[i, :] = bf16(s_i * N(0, 1)). The row classes sit at the same row indices in every
input set.
mu and rs are the FP32 statistics of float(x) (so rs[i] ranges over ~0.1 .. 316); gamma
is i.i.d. N(0, 1); dz = bf16(N(0, 1)).
Outputs: dx [rows, cols] (bf16), dgamma [cols] (fp32), dbeta [cols] (fp32).

Precision: BF16 is only the storage format of the activations and gradients. All
arithmetic (x_hat, products, the row sums ds/db, the dx formula, the column sums
dgamma/dbeta) must be done in FP32 (or higher) on values converted from BF16, and
each BF16 output must come from a single round-to-nearest-even conversion of its
final FP32 value. FP16/BF16/FP8 arithmetic, tensor-core MMAs, or holding
intermediates in a lower precision are not allowed, even if the result happens to
pass the tolerance check.

Correctness: dx, dgamma and dbeta are compared element-wise (in FP32) with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 0.001, rtol = 0.024
rtol allows about 3 BF16 ulps of rounding difference; atol is small compared with the
gradient magnitudes (|dx|, |dgamma| and |dbeta| reach ~1e3), so values that nearly
cancel to zero must be accurate to FP32 level. NaN/Inf fail, and every output element
must be written (output buffers are pre-filled with NaN).
""" % (ROWS, COLS)
