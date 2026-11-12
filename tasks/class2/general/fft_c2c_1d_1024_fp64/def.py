"""
Task: 1D Complex-to-Complex FFT -- 1024, float64, batch=65536

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 1024
BATCH_SIZE = 65536

FUNCTION_SIGNATURE = """\
void launch_fft_c2c_1d_1024_fp64(
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

TOLERANCES = {"atol": 4e-13, "rtol": 0.0, "max_normalized_rms": 1.7e-15}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(65536, 1024, 2, device="cuda", dtype=torch.float64).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(65536, 1024, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fft(x_complex)
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 1D complex-to-complex FFT.

Workload: 65536 independent length-1024 double-precision transforms per call
(1 GiB of complex128 input), e.g. the batched 1D transforms of a spectral
PDE solver.

Unnormalized forward transform of every batch element b:
  X[b][k] = sum_{n=0}^{N-1} x[b][n] * exp(-2*pi*i*k*n/N),  k = 0 .. N-1

Parameters:
  input:  [65536 * 1024 * 2] float64 -- complex input, interleaved (re, im), transform b at
          complex offset b*N
  output: [65536 * 1024 * 2] float64 -- complex output, same layout
  N:          1024 -- elements per transform (product of the dimensions)
  batch_size: 65536
  inverse:    always 0 (forward transform)

Precision: the result must be FP64-accurate, i.e. as accurate as a native FP64
FFT; the tolerance below is set at that level. Native FP64 arithmetic, or
emulating FP64 with lower-precision units, are both allowed as long as the
result meets the tolerance (emulation with less than FP64 precision, such as
double-float arithmetic, does not).

Correctness: on 3 random inputs (real and imaginary parts ~ N(0, 1)), compared
in float64 against a reference FFT: every output value must satisfy
|out - ref| <= atol = 4e-13 (rtol = 0; about 12 * 2^-53 * log2(N) * sqrt(N), the output
components have RMS ~ sqrt(N) = 32), and RMS(out - ref) / RMS(ref) <= 1.7e-15
(about 1.5 * 2^-53 * log2(N)). A correct native FP64 FFT stays >= 3x inside both.
"""
