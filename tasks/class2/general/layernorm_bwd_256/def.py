"""
Task: LayerNorm Backward — hidden_size=256, BF16 activations/gradients, FP32 compute

Source: flash-attention/csrc/layer_norm/ln_bwd_kernels.cuh
Operation: Compute dx (bf16), dgamma, dbeta (fp32) given upstream gradient dz (bf16),
saved x (bf16), mu, rs (fp32) and gamma (fp32).
"""
import torch

ROWS = 262144
COLS = 256

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


def get_inputs():
    # Per-row scale 10^U(-2, 1) on x: rs spans ~0.1 .. 95, so dx (proportional to rs)
    # covers a wide range and a missing ds/db term or BF16 arithmetic shows up on the
    # high-rs rows.
    row_scale = torch.pow(10.0, torch.empty(ROWS, 1, device='cuda').uniform_(-2.0, 1.0))
    x = (torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32) * row_scale).to(torch.bfloat16)
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

Scenario: LayerNorm backward in small-model training (ViT-Small / BERT-mini scale,
hidden size 256): 262144 tokens per step, e.g. 256 images x 1024 patches or 512
sequences x 512 tokens.

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
Inputs: x[i, :] = bf16(s_i * N(0, 1)) with a per-row scale s_i = 10^U(-2, 1); mu and rs
are the FP32 statistics of float(x) (so rs[i] ranges over ~0.1 .. 95); gamma is
i.i.d. N(0, 1); dz = bf16(N(0, 1)).
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
