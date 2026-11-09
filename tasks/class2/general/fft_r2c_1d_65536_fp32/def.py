"""
Task: 1D Real-to-Complex FFT -- 65536, float32, batch=4096

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 65536
BATCH_SIZE = 4096

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

TOLERANCES = {"atol": 0.002, "rtol": 0.001}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(4096, 65536, device="cuda", dtype=torch.float32)
    return [("input", x.view(-1))]


def get_outputs(inputs):
    out = torch.empty(4096 * 32769 * 2, device="cuda", dtype=torch.float32)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(4096, 65536)
    y = torch.fft.rfftn(x, dim=[-1])
    y_real = torch.view_as_real(y)  # [..., 2]
    return [("output", y_real.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 1D real-to-complex FFT.

Workload: 4096 independent real length-65536 signals per call (1 GiB of float32
input), e.g. spectra of batched audio or sensor streams.

Unnormalized forward transform of every batch element b:
  X[b][k] = sum_{n=0}^{N-1} x[b][n] * exp(-2*pi*i*k*n/N),  k = 0 .. N/2 (the non-redundant half)

Parameters:
  input:  [4096 * 65536] float32 -- real input, signal b at offset b*N
  output: [4096 * 32769 * 2] float32 -- complex output X[b][0..N/2], interleaved (re, im)
  N:          65536 -- elements per transform (product of the dimensions)
  batch_size: 4096
  inverse:    always 0 (forward transform)

Precision: the result must be FP32-accurate; single-pass lower-precision
(FP16 / BF16 / TF32) computation cannot meet the tolerance below. Native FP32
arithmetic, or emulating FP32 with lower-precision units (e.g. split-precision
tensor-core arithmetic), are both allowed as long as the result meets the
tolerance.

Correctness: on 3 random inputs (every input value ~ N(0, 1)),
every output value must satisfy |out - ref| <= atol + rtol * |ref| against a
reference FFT, with atol = 0.002, rtol = 0.001 (output components have
RMS ~ sqrt(N)).
"""
