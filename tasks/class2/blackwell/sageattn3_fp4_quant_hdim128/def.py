"""
Task: SageAttention v3 FP4 Quantization -- head_dim=128

Source: SageAttention v3 (Blackwell NVFP4 quantization)
"""
import torch
import math

HEAD_DIM = 128
BATCH_SIZE = 2
NUM_HEADS = 16
SEQ_LEN = 1024
BLOCK_SIZE = 128
SF_PER_TOKEN = 8  # one FP8 scale per 16 elements

FUNCTION_SIGNATURE = """\
void sageattn3_fp4_quant_hdim128(
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

TOLERANCES = {"atol": 0.0, "rtol": 0.0}  # exact match for quantization

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
    # Packed FP4: each byte holds 2 FP4 values
    out = torch.zeros(BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM // 2,
                      device="cuda", dtype=torch.uint8)
    # Block-scaled layout for scale factors
    out_sf = torch.zeros(BATCH_SIZE * NUM_HEADS * SEQ_LEN * SF_PER_TOKEN,
                         device="cuda", dtype=torch.uint8)
    return [("output", out), ("output_sf", out_sf)]


def _fp4_quantize_ref(x):
    """Python reference: quantize FP16 to NVFP4 E2M1 with per-16-element FP8 scales.

    This is a simplified reference that returns dequantized values since
    we cannot easily replicate the exact bit-level NVFP4 packing in Python.
    The CUDA reference is authoritative for bit-exact output.
    """
    B, H, S, D = x.shape
    # Reshape to blocks of 16 elements
    x_blocks = x.view(B, H, S, D // 16, 16).float()
    # Per-block max absolute value
    amax = x_blocks.abs().amax(dim=-1)  # [B, H, S, D//16]
    # FP8 E4M3 scale = max / 6.0
    scale = amax / 6.0
    # Quantize (simplified: round to nearest representable FP4)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    x_scaled = x_blocks * inv_scale.unsqueeze(-1)
    # FP4 E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # Clamp to [-6, 6] and round
    x_clamped = x_scaled.clamp(-6.0, 6.0)
    # Dequantize back
    x_deq = x_clamped * scale.unsqueeze(-1)
    return x_deq.view(B, H, S, D).half()


def reference_fn(inputs):
    x = inputs[0][1].view(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    # For quantization tasks, the CUDA reference is authoritative.
    # This Python reference is only used by test_reference.py for sanity checking.
    q = _fp4_quantize_ref(x)
    # Return dummy outputs matching expected shapes
    out = torch.zeros(BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM // 2,
                      device="cuda", dtype=torch.uint8)
    out_sf = torch.zeros(BATCH_SIZE * NUM_HEADS * SEQ_LEN * SF_PER_TOKEN,
                         device="cuda", dtype=torch.uint8)
    return [("output", out), ("output_sf", out_sf)]


DESCRIPTION = """\
Implement a CUDA kernel for FP4 (NVFP4 E2M1) quantization with block-scaled scale factors.

This is a SageAttention v3 quantization kernel that converts FP16 input to packed FP4
(E2M1 format) with per-16-element FP8 (E4M3) scale factors.

Algorithm:
  1. Load 16 consecutive FP16 elements per thread group
  2. Compute max absolute value across the 16 elements
  3. Compute FP8 E4M3 scale factor: scale = max_abs / 6.0
  4. Scale input: x_scaled = x * (1.0 / scale)
  5. Convert to FP4 E2M1 using PTX: cvt.rn.satfinite.e2m1x2.f32
  6. Pack two FP4 values per byte (uint8 output)
  7. Write scale factors in block-scaled layout for SM120 MMA consumption

Block-scaled layout for scale factors (per 64-token super-block):
  offset = (col_id / 4) * 256 + (col_id % 4) + (token_id_local / 16) * 4 + (token_id_local % 16) * 16
  where token_id_local = token_id % 64, col_id = head_dim_offset / 16

Parameters:
  input:     [2 * 16 * 1024 * 128] float16 -- input tensor (BHSD layout)
  output:    [2 * 16 * 1024 * 64] uint8 -- packed FP4 output
  output_sf: [2 * 16 * 1024 * 8] uint8 -- FP8 E4M3 scale factors (block-scaled layout)
  batch_size:  2
  num_heads:   16
  num_tokens:  1024

Head dimension: 128
Quantization: per-16-element blocks, NVFP4 E2M1 format
Scale factors: FP8 E4M3 in SM120 block-scaled layout
Requires: SM120+ (Blackwell) for cvt.rn.satfinite.e2m1x2.f32 PTX instruction
"""
