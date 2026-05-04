"""
Task: 3D Complex-to-Complex FFT -- 64x64x64, float32, batch=1

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 262144
BATCH_SIZE = 1

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_3d_64x64x64_fp32(
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
    x = torch.randn(1, 262144, 2, device="cuda", dtype=torch.float32).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(1, 64, 64, 64, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fftn(x_complex, dim=[-3, -2, -1])
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a CUDA kernel for 3D Complex-to-Complex FFT (forward transform).

Parameters:
  input:  [1 * 262144 * 2] float32 -- complex input (interleaved real/imag)
  output: [1 * 262144 * 2] float32 -- complex output (interleaved real/imag)
  N:          262144 -- total elements per batch (product of dimensions)
  batch_size: 1
  inverse:    0 for forward FFT

Dimensions: 64x64x64
Data type: float32

The forward DFT is defined as:
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)

Each batch element is an independent 3D FFT.
"""
