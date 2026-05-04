"""
Task: LayerNorm Parallel Residual Forward — hidden_size=8192

Source: flash-attention/csrc/layer_norm/ln_parallel_residual_fwd_kernels.cuh
Operation: Fused residual addition + layer normalization.
  x = x0 + residual
  z = LayerNorm(x, gamma, beta)
"""
import torch

ROWS = 32 * 512
COLS = 8192
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

TOLERANCES = {"atol": 1e-4, "rtol": 1e-4}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    x0 = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    residual = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    gamma = torch.ones(COLS, device='cuda', dtype=torch.float32)
    beta = torch.zeros(COLS, device='cuda', dtype=torch.float32)
    return [("x0", x0), ("residual", residual), ("gamma", gamma), ("beta", beta)]


def get_outputs(inputs):
    x0 = inputs[0][1]
    z = torch.empty_like(x0)
    x = torch.empty_like(x0)
    mu = torch.empty(ROWS, device='cuda', dtype=torch.float32)
    rs = torch.empty(ROWS, device='cuda', dtype=torch.float32)
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
  mu[i] = mean(x[i, :])
  rs[i] = rsqrt(var(x[i, :]) + eps)
  z[i,j] = (x[i,j] - mu[i]) * rs[i] * gamma[j] + beta[j]

Dimensions: rows=%d, cols=%d, eps=%g
""" % (ROWS, COLS, EPS)
