"""
Task: 3D Complex-to-Complex FFT -- 64x64x64, float32, batch=512

Source: cuFFT (NVIDIA CUDA Library Samples)
"""
import torch

N = 262144
BATCH_SIZE = 512

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

TOLERANCES = {"atol": 0.004, "rtol": 0.001}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = []
REFERENCE_EXTRA_CUDA_FLAGS = []
REFERENCE_EXTRA_LDFLAGS = ["-lcufft"]


def get_inputs():
    x = torch.randn(512, 262144, 2, device="cuda", dtype=torch.float32).contiguous()
    return [("input", x.view(-1))]


def get_outputs(inputs):
    x_flat = inputs[0][1]
    out = torch.empty_like(x_flat)
    return [("output", out)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x_flat = inputs[0][1]
    x = x_flat.view(512, 64, 64, 64, 2)
    x_complex = torch.view_as_complex(x)
    y_complex = torch.fft.fftn(x_complex, dim=[-3, -2, -1])
    y = torch.view_as_real(y_complex)
    return [("output", y.reshape(-1))]


DESCRIPTION = """\
Implement a batched forward 3D complex-to-complex FFT.

Workload: a batch of 512 complex 64x64x64 volumes per call (1 GiB of complex64
input), e.g. batched volume transforms in cryo-EM or spectral simulations.

Unnormalized forward transform of every batch element b:
  X[b][k0][k1][k2] = sum_{n0,n1,n2} x[b][n0][n1][n2]
                     * exp(-2*pi*i*(k0*n0/N0 + k1*n1/N1 + k2*n2/N2))
  with N0 = N1 = N2 = 64; each volume is row-major [N0][N1][N2]

Parameters:
  input:  [512 * 262144 * 2] float32 -- complex input, interleaved (re, im), transform b at
          complex offset b*N
  output: [512 * 262144 * 2] float32 -- complex output, same layout
  N:          262144 -- elements per transform (product of the dimensions)
  batch_size: 512
  inverse:    always 0 (forward transform)

Precision: the result must be FP32-accurate; single-pass lower-precision
(FP16 / BF16 / TF32) computation cannot meet the tolerance below. Native FP32
arithmetic, or emulating FP32 with lower-precision units (e.g. split-precision
tensor-core arithmetic), are both allowed as long as the result meets the
tolerance.

Correctness: on 3 random inputs (real and imaginary parts ~ N(0, 1)),
every output value must satisfy |out - ref| <= atol + rtol * |ref| against a
reference FFT, with atol = 0.004, rtol = 0.001 (output components have
RMS ~ sqrt(N)).
"""
