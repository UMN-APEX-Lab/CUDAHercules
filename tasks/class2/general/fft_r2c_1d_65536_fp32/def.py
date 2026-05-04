"""
Task: 1D Real-to-Complex FFT -- 65536, float32, batch=16

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 65536
BATCH_SIZE = 16

FUNCTION_SIGNATURE = """\
void launch_fft_r2c_1d_65536_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {
    "N": N,
    "batch_size": BATCH_SIZE,
    "inverse": 0,
}

TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(16, 65536, device="cuda", dtype=torch.float32)
    return [("input", x.view(-1))]


def get_outputs(inputs):
    out = torch.empty(16 * 32769 * 2, device="cuda", dtype=torch.float32)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(16, 65536)
    y = torch.fft.rfftn(x, dim=[-1])
    y_real = torch.view_as_real(y)  # [..., 2]
    return [("output", y_real.reshape(-1))]


DESCRIPTION = """\
Implement a CUDA kernel for 1D Real-to-Complex FFT (forward transform).

Parameters:
  input:  [16 * 65536] float32 -- real input (flat)
  output: [16 * 32769 * 2] float32 -- complex output (interleaved real/imag)
  N:          65536 -- total elements per batch (product of dimensions)
  batch_size: 16
  inverse:    0 for forward FFT

Dimensions: 65536
Data type: float32

The forward DFT is defined as:
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)

Each batch element is an independent 1D FFT.
"""
