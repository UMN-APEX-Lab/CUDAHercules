"""
Task: LayerNorm Backward — hidden_size=4096

Source: flash-attention/csrc/layer_norm/ln_bwd_kernels.cuh
Operation: Compute dx, dgamma, dbeta given upstream gradient dz, saved x, mu, rs, gamma.
"""
import torch

ROWS = 32 * 512
COLS = 4096

FUNCTION_SIGNATURE = """\
void launch_layernorm_backward(
    const float* dz,      // [rows, cols] upstream gradient
    const float* x,       // [rows, cols] saved input from forward
    const float* mu,      // [rows] saved mean
    const float* rs,      // [rows] saved reciprocal std
    const float* gamma,   // [cols] scale weight
    float* dx,            // [rows, cols] gradient w.r.t. input
    float* dgamma,        // [cols] gradient w.r.t. gamma
    float* dbeta,         // [cols] gradient w.r.t. beta
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
    x = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    mu = x.mean(dim=-1)
    rs = torch.rsqrt(x.var(dim=-1, unbiased=False) + 1e-5)
    gamma = torch.ones(COLS, device='cuda', dtype=torch.float32)
    dz = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    return [("dz", dz), ("x", x), ("mu", mu), ("rs", rs), ("gamma", gamma)]


def get_outputs(inputs):
    dx = torch.empty(ROWS, COLS, device='cuda', dtype=torch.float32)
    dgamma = torch.empty(COLS, device='cuda', dtype=torch.float32)
    dbeta = torch.empty(COLS, device='cuda', dtype=torch.float32)
    return [("dx", dx), ("dgamma", dgamma), ("dbeta", dbeta)]


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

    return [("dx", dx), ("dgamma", dgamma), ("dbeta", dbeta)]


DESCRIPTION = """\
Implement a CUDA kernel for LayerNorm backward pass.

Given upstream gradient dz, saved input x, mean mu, reciprocal std rs, and weight gamma:

Per-row (data gradient):
  x_hat[i,j] = (x[i,j] - mu[i]) * rs[i]
  dz_gamma[i,j] = dz[i,j] * gamma[j]
  ds[i] = sum_j(dz_gamma[i,j] * x_hat[i,j])
  db[i] = sum_j(dz_gamma[i,j])
  dx[i,j] = rs[i] * (dz_gamma[i,j] - (ds[i] * x_hat[i,j] + db[i]) / cols)

Per-column (weight gradients):
  dgamma[j] = sum_i(dz[i,j] * x_hat[i,j])
  dbeta[j] = sum_i(dz[i,j])

Dimensions: rows=%d, cols=%d
""" % (ROWS, COLS)
