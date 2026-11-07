"""
Task: LayerNorm Parallel Residual Forward — hidden_size=1024

Source: flash-attention/csrc/layer_norm/ln_parallel_residual_fwd_kernels.cuh
Operation: Fused residual addition + layer normalization.
  x = x0 + residual
  z = LayerNorm(x, gamma, beta)
"""
import torch

ROWS = 32 * 512
COLS = 1024
EPS = 1e-5

FUNCTION_SIGNATURE = """\
void launch_layernorm_parallel_fwd(
    const float* x0,       // [rows, cols] new activation
    const float* residual, // [rows, cols] residual stream
    const float* gamma,    // [cols] scale weight
    const float* beta,     // [cols] bias weight
    float* z,              // [rows, cols] normalized output
    float* x,              // [rows, cols] x0 + residual (saved for backward)
    float* mu,             // [rows] per-row mean
    float* rs,             // [rows] per-row reciprocal std
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS, "eps": EPS}

TOLERANCES = {"atol": 1e-05, "rtol": 1e-05}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    # Per-row scale 10^U(-2, 1): rows with a small variance make eps matter (a kernel
    # that drops eps, or uses the unbiased variance, fails on them).
    row_scale = torch.pow(10.0, torch.empty(ROWS, 1, device='cuda').uniform_(-2.0, 1.0))
    x0 = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * row_scale
    residual = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * row_scale
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
    """PyTorch reference for correctness validation."""
    x0 = inputs[0][1]
    residual = inputs[1][1]
    gamma = inputs[2][1]
    beta = inputs[3][1]

    x = x0 + residual
    mu = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    rs = torch.rsqrt(var + EPS)
    x_hat = (x - mu.unsqueeze(-1)) * rs.unsqueeze(-1)
    z = x_hat * gamma + beta

    return [("z", z), ("x", x), ("mu", mu), ("rs", rs)]


DESCRIPTION = """\
Implement a CUDA kernel for fused residual addition + LayerNorm forward pass.

  x[i,j] = x0[i,j] + residual[i,j]
  mu[i]  = (1/cols) * sum_j x[i, j]
  var[i] = (1/cols) * sum_j (x[i, j] - mu[i])^2      (biased variance)
  rs[i]  = 1 / sqrt(var[i] + eps)
  z[i,j] = (x[i,j] - mu[i]) * rs[i] * gamma[j] + beta[j]

Dimensions: rows=%d, cols=%d, eps=%g. All tensors are contiguous row-major fp32.
x0[i, :] and residual[i, :] are s_i * N(0, 1) with a per-row scale s_i = 10^U(-2, 1)
(so var[i] ranges over ~2e-4 .. 2e2 and eps matters for the small rows); gamma and beta
are i.i.d. N(0, 1).
Outputs: z [rows, cols], x [rows, cols], mu [rows], rs [rows].

Precision: all arithmetic must be FP32 (or higher). FP16/BF16/FP8 arithmetic or
storing intermediates in a lower precision is not allowed.

Correctness: z, x, mu and rs are compared element-wise with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 1e-05, rtol = 1e-05
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (ROWS, COLS, EPS)
