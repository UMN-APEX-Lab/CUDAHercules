"""
Task: SageAttention v3 FP4 Transposed Quantization -- head_dim=64

Source: SageAttention v3 (Blackwell NVFP4 transposed quantization for V)
"""
import torch
import math

HEAD_DIM = 64
BATCH_SIZE = 2
NUM_HEADS = 16
SEQ_LEN = 1024
BLOCK_SIZE = 128
PADDED_SEQ = 1024  # round up to BLOCK_SIZE

FUNCTION_SIGNATURE = """\
void sageattn3_fp4_quant_trans_hdim64(
    const void* input,
    void* output,
    void* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {
    "batch_size": BATCH_SIZE,
    "num_heads": NUM_HEADS,
    "num_tokens": SEQ_LEN,
}

TOLERANCES = {"atol": 0.0, "rtol": 0.0}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/sage_attn_v3/quantization",
]
REFERENCE_EXTRA_CUDA_FLAGS = [
    "-gencode", "arch=compute_120a,code=sm_120a",
    "-include", "cassert",
    "-std=c++17",
]
REFERENCE_EXTRA_LDFLAGS = []


def get_inputs():
    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                     device="cuda", dtype=torch.float16)
    return [("input", x.contiguous().view(-1))]


def get_outputs(inputs):
    # Transposed output: [B, H, D, PADDED_SEQ/2] packed FP4
    out = torch.zeros(BATCH_SIZE * NUM_HEADS * HEAD_DIM * PADDED_SEQ // 2,
                      device="cuda", dtype=torch.uint8)
    # Transposed scales: [B, H, D, PADDED_SEQ/16]
    out_sf = torch.zeros(BATCH_SIZE * NUM_HEADS * HEAD_DIM * PADDED_SEQ // 16,
                         device="cuda", dtype=torch.uint8)
    return [("output", out), ("output_sf", out_sf)]


def reference_fn(inputs):
    # Python reference is a stub for transposed quantization
    out = torch.zeros(BATCH_SIZE * NUM_HEADS * HEAD_DIM * PADDED_SEQ // 2,
                      device="cuda", dtype=torch.uint8)
    out_sf = torch.zeros(BATCH_SIZE * NUM_HEADS * HEAD_DIM * PADDED_SEQ // 16,
                         device="cuda", dtype=torch.uint8)
    return [("output", out), ("output_sf", out_sf)]


DESCRIPTION = """\
Implement a CUDA kernel for FP4 transposed quantization (for attention V matrix).

This kernel converts FP16 input [B, H, S, D] to transposed packed FP4 [B, H, D, S/2]
with FP8 E4M3 scale factors. The transpose is done in shared memory for efficiency.
This layout allows the attention kernel to read V column-wise for the P @ V matmul.

Algorithm:
  1. Load [BLOCK_SIZE, HEAD_DIM] tile of FP16 values per thread block
  2. Store to shared memory
  3. Read back transposed: each thread now handles [D_chunk, BLOCK_SIZE_chunk]
  4. Compute per-16-element max and FP8 scale factor
  5. Convert to FP4 E2M1 using PTX cvt.rn.satfinite.e2m1x2.f32
  6. Write transposed packed output and block-scaled scale factors

Parameters:
  input:     [2 * 16 * 1024 * 64] float16 -- input V (BHSD layout)
  output:    [2 * 16 * 64 * 512] uint8 -- transposed packed FP4
  output_sf: [2 * 16 * 64 * 64] uint8 -- transposed FP8 scales
  batch_size:  2
  num_heads:   16
  num_tokens:  1024

Head dimension: 64
Sequence length padded to: 1024 (multiple of 128)
Output layout: transposed [B, H, D, S] for column-wise V access
Requires: SM120+ (Blackwell)
"""
