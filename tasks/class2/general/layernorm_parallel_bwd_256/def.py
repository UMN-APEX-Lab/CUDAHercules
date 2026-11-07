"""
Task: LayerNorm Parallel Residual Backward — hidden_size=256

Source: flash-attention/csrc/layer_norm/ln_parallel_residual_bwd_kernels.cuh
Operation: Backward of fused residual addition + layer normalization.
  Compute dx0, dresidual, dgamma, dbeta given dz, x (= x0+residual), mu, rs, gamma.
"""
import torch

ROWS = 32 * 512
COLS = 256

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

TOLERANCES = {"atol": 0.003, "rtol": 0.0001}  # mirrored in task.yaml correctness.tolerances

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    x0 = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    residual = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    x = x0 + residual
    mu = x.mean(dim=-1)
    rs = torch.rsqrt(x.var(dim=-1, unbiased=False) + 1e-5)
    gamma = torch.randn(COLS, device='cuda', dtype=torch.float32)
    dz = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    return [("dz", dz), ("x", x), ("mu", mu), ("rs", rs), ("gamma", gamma)]


def get_outputs(inputs):
    # NaN-poisoned so that elements the kernel does not write fail the comparison.
    dx0 = torch.full((ROWS, COLS), float('nan'), device='cuda', dtype=torch.float32)
    dresidual = torch.full((ROWS, COLS), float('nan'), device='cuda', dtype=torch.float32)
    dgamma = torch.full((COLS,), float('nan'), device='cuda', dtype=torch.float32)
    dbeta = torch.full((COLS,), float('nan'), device='cuda', dtype=torch.float32)
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

Given upstream gradient dz [rows, cols], saved x = x0 + residual [rows, cols], saved mean
mu [rows], saved reciprocal std rs [rows] (rs = 1/sqrt(var + 1e-5), biased variance) and
weight gamma [cols]:
  x_hat[i,j]    = (x[i,j] - mu[i]) * rs[i]
  dz_gamma[i,j] = dz[i,j] * gamma[j]
  ds[i] = sum_j(dz_gamma[i,j] * x_hat[i,j])
  db[i] = sum_j(dz_gamma[i,j])
  dx[i,j] = rs[i] * (dz_gamma[i,j] - (ds[i] * x_hat[i,j] + db[i]) / cols)
  dx0 = dx
  dresidual = dx      (the gradient flows to both branches; write both outputs)
  dgamma[j] = sum_i(dz[i,j] * x_hat[i,j])
  dbeta[j]  = sum_i(dz[i,j])

Dimensions: rows=%d, cols=%d. All tensors are contiguous row-major fp32.
x0, residual, gamma and dz are i.i.d. standard normal; mu and rs are the statistics of x.
Outputs: dx0 [rows, cols], dresidual [rows, cols], dgamma [cols], dbeta [cols].

Precision: all arithmetic must be FP32 (or higher). FP16/BF16/FP8 arithmetic or
storing intermediates in a lower precision is not allowed.

Correctness: dx0, dresidual, dgamma and dbeta are compared element-wise with a reference
implementation over 3 random input sets:
  |out - ref| <= atol + rtol * |ref|,  atol = 0.003, rtol = 0.0001
NaN/Inf fail, and every output element must be written (output buffers are
pre-filled with NaN).
""" % (ROWS, COLS)
