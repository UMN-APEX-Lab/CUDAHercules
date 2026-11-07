"""
Task: 1D Complex-to-Complex FFT -- 65536, float32, batch=2048

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 65536
BATCH_SIZE = 2048

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_1d_65536_fp32(
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

TOLERANCES = {"atol": 0.002, "rtol": 0.001}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(2048, 65536, 2, device="cuda", dtype=torch.float32).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(2048, 65536, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fft(x_complex)
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 1D complex-to-complex FFT.

Workload: 2048 independent length-65536 complex signals per call (1 GiB of
complex64 input), e.g. batched spectral analysis of signal streams.

Unnormalized forward transform of every batch element b:
  X[b][k] = sum_{n=0}^{N-1} x[b][n] * exp(-2*pi*i*k*n/N),  k = 0 .. N-1

Parameters:
  input:  [2048 * 65536 * 2] float32 -- complex input, interleaved (re, im), transform b at
          complex offset b*N
  output: [2048 * 65536 * 2] float32 -- complex output, same layout
  N:          65536 -- elements per transform (product of the dimensions)
  batch_size: 2048
  inverse:    always 0 (forward transform)

Precision: FP32 (or higher) arithmetic for twiddle factors and butterflies;
FP16 / BF16 / TF32 or other reduced-precision arithmetic is not allowed.

Correctness: on 3 random inputs (real and imaginary parts ~ N(0, 1)),
every output value must satisfy |out - ref| <= atol + rtol * |ref| against a
reference FFT, with atol = 0.002, rtol = 0.001 (output components have
RMS ~ sqrt(N)).
"""
