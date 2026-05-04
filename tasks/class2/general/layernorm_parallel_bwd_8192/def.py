"""
Task: LayerNorm Parallel Residual Backward — hidden_size=8192

Source: flash-attention/csrc/layer_norm/ln_parallel_residual_bwd_kernels.cuh
Operation: Backward of fused residual addition + layer normalization.
  Compute dx0, dresidual, dgamma, dbeta given dz, x (= x0+residual), mu, rs, gamma.
"""
import torch

ROWS = 32 * 512
COLS = 8192

FUNCTION_SIGNATURE = """\
void launch_layernorm_parallel_bwd(
    const float* dz,         // [rows, cols] upstream gradient
    const float* x,          // [rows, cols] saved x0+residual from forward
    const float* mu,         // [rows] saved mean
    const float* rs,         // [rows] saved reciprocal std
    const float* gamma,      // [cols] scale weight
    float* dx0,              // [rows, cols] gradient w.r.t. x0
    float* dresidual,        // [rows, cols] gradient w.r.t. residual
    float* dgamma,           // [cols] gradient w.r.t. gamma
    float* dbeta,            // [cols] gradient w.r.t. beta
    int rows,
    int cols,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS}

TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    x0 = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    residual = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    x = x0 + residual
    mu = x.mean(dim=-1)
    rs = torch.rsqrt(x.var(dim=-1, unbiased=False) + 1e-5)
    gamma = torch.ones(COLS, device='cuda', dtype=torch.float32)
    dz = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    return [("dz", dz), ("x", x), ("mu", mu), ("rs", rs), ("gamma", gamma)]


def get_outputs(inputs):
    dx0 = torch.empty(ROWS, COLS, device='cuda', dtype=torch.float32)
    dresidual = torch.empty(ROWS, COLS, device='cuda', dtype=torch.float32)
    dgamma = torch.empty(COLS, device='cuda', dtype=torch.float32)
    dbeta = torch.empty(COLS, device='cuda', dtype=torch.float32)
    return [("dx0", dx0), ("dresidual", dresidual), ("dgamma", dgamma), ("dbeta", dbeta)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    dz = inputs[0][1]
    x = inputs[1][1]
    mu = inputs[2][1]
    rs = inputs[3][1]
    gamma = inputs[4][1]

    x_hat = (x - mu.unsqueeze(-1)) * rs.unsqueeze(-1)
    dgamma = (dz * x_hat).sum(dim=0)
    dbeta = dz.sum(dim=0)

    D = COLS
    dz_gamma = dz * gamma
    ds = (dz_gamma * x_hat).sum(dim=-1, keepdim=True)
    db = dz_gamma.sum(dim=-1, keepdim=True)
    dx = rs.unsqueeze(-1) * (dz_gamma - (ds * x_hat + db) / D)

    # For parallel residual: dx0 = dx, dresidual = dx
    dx0 = dx
    dresidual = dx.clone()

    return [("dx0", dx0), ("dresidual", dresidual), ("dgamma", dgamma), ("dbeta", dbeta)]


DESCRIPTION = """\
Implement a CUDA kernel for the backward pass of fused residual + LayerNorm.

Given upstream gradient dz, saved x (= x0 + residual), mean mu, reciprocal std rs, gamma:

  x_hat = (x - mu) * rs
  dgamma = sum_rows(dz * x_hat)
  dbeta = sum_rows(dz)
  dx = rs * (dz*gamma - (ds*x_hat + db) / cols)
  dx0 = dx
  dresidual = dx  (gradient flows through both branches)

Dimensions: rows=%d, cols=%d
""" % (ROWS, COLS)
