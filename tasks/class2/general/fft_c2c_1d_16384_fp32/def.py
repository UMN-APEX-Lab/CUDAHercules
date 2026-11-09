"""
Task: 1D Complex-to-Complex FFT -- 16384, float32, batch=8192

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 16384
BATCH_SIZE = 8192

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_1d_16384_fp32(
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

TOLERANCES = {"atol": 0.001, "rtol": 0.001}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(8192, 16384, 2, device="cuda", dtype=torch.float32).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(8192, 16384, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fft(x_complex)
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 1D complex-to-complex FFT.

Workload: 8192 independent length-16384 complex signals per call (1 GiB of
complex64 input), e.g. batched spectral analysis of signal streams.

Unnormalized forward transform of every batch element b:
  X[b][k] = sum_{n=0}^{N-1} x[b][n] * exp(-2*pi*i*k*n/N),  k = 0 .. N-1

Parameters:
  input:  [8192 * 16384 * 2] float32 -- complex input, interleaved (re, im), transform b at
          complex offset b*N
  output: [8192 * 16384 * 2] float32 -- complex output, same layout
  N:          16384 -- elements per transform (product of the dimensions)
  batch_size: 8192
  inverse:    always 0 (forward transform)

Precision: the result must be FP32-accurate; single-pass lower-precision
(FP16 / BF16 / TF32) computation cannot meet the tolerance below. Native FP32
arithmetic, or emulating FP32 with lower-precision units (e.g. split-precision
tensor-core arithmetic), are both allowed as long as the result meets the
tolerance.

Correctness: on 3 random inputs (real and imaginary parts ~ N(0, 1)),
every output value must satisfy |out - ref| <= atol + rtol * |ref| against a
reference FFT, with atol = 0.001, rtol = 0.001 (output components have
RMS ~ sqrt(N)).
"""
