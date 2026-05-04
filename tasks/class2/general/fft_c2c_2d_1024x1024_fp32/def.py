"""
Task: 2D Complex-to-Complex FFT -- 1024x1024, float32, batch=1

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 1048576
BATCH_SIZE = 1

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_2d_1024x1024_fp32(
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
    x = torch.randn(1, 1048576, 2, device="cuda", dtype=torch.float32).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(1, 1024, 1024, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fftn(x_complex, dim=[-2, -1])
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a CUDA kernel for 2D Complex-to-Complex FFT (forward transform).

Parameters:
  input:  [1 * 1048576 * 2] float32 -- complex input (interleaved real/imag)
  output: [1 * 1048576 * 2] float32 -- complex output (interleaved real/imag)
  N:          1048576 -- total elements per batch (product of dimensions)
  batch_size: 1
  inverse:    0 for forward FFT

Dimensions: 1024x1024
Data type: float32

The forward DFT is defined as:
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)

Each batch element is an independent 2D FFT.
"""
