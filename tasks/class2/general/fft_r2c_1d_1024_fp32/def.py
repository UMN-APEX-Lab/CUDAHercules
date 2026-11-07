"""
Task: 1D Real-to-Complex FFT -- 1024, float32, batch=262144

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 1024
BATCH_SIZE = 262144

FUNCTION_SIGNATURE = """\
void launch_fft_r2c_1d_1024_fp32(
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
    x = torch.randn(262144, 1024, device="cuda", dtype=torch.float32)
    return [("input", x.view(-1))]


def get_outputs(inputs):
    out = torch.empty(262144 * 513 * 2, device="cuda", dtype=torch.float32)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(262144, 1024)
    y = torch.fft.rfftn(x, dim=[-1])
    y_real = torch.view_as_real(y)  # [..., 2]
    return [("output", y_real.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 1D real-to-complex FFT.

Workload: 262144 independent real length-1024 signals per call (1 GiB of float32
input), e.g. spectra of batched audio or sensor streams.

Unnormalized forward transform of every batch element b:
  X[b][k] = sum_{n=0}^{N-1} x[b][n] * exp(-2*pi*i*k*n/N),  k = 0 .. N/2 (the non-redundant half)

Parameters:
  input:  [262144 * 1024] float32 -- real input, signal b at offset b*N
  output: [262144 * 513 * 2] float32 -- complex output X[b][0..N/2], interleaved (re, im)
  N:          1024 -- elements per transform (product of the dimensions)
  batch_size: 262144
  inverse:    always 0 (forward transform)

Precision: FP32 (or higher) arithmetic for twiddle factors and butterflies;
FP16 / BF16 / TF32 or other reduced-precision arithmetic is not allowed.

Correctness: on 3 random inputs (every input value ~ N(0, 1)),
every output value must satisfy |out - ref| <= atol + rtol * |ref| against a
reference FFT, with atol = 0.001, rtol = 0.001 (output components have
RMS ~ sqrt(N)).
"""
