"""
Task: 1D Complex-to-Complex FFT -- 4096, float64, batch=64

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 4096
BATCH_SIZE = 64

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_1d_4096_fp64(
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
    x = torch.randn(64, 4096, 2, device="cuda", dtype=torch.float64).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(64, 4096, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fft(x_complex)
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a CUDA kernel for 1D Complex-to-Complex FFT (forward transform).

Parameters:
  input:  [64 * 4096 * 2] float64 -- complex input (interleaved real/imag)
  output: [64 * 4096 * 2] float64 -- complex output (interleaved real/imag)
  N:          4096 -- total elements per batch (product of dimensions)
  batch_size: 64
  inverse:    0 for forward FFT

Dimensions: 4096
Data type: float64

The forward DFT is defined as:
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)

Each batch element is an independent 1D FFT.
"""
