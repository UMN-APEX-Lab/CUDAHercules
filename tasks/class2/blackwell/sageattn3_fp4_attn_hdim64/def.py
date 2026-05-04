"""
Task: SageAttention v3 FP4 Attention -- head_dim=64

Source: SageAttention v3 (Blackwell BlockScaled FP4 MMA attention)
The reference internally quantizes FP16 inputs to FP4 then runs BlockScaled attention.
"""
import torch
import math

HEAD_DIM = 64
BATCH_SIZE = 2
NUM_HEADS = 16
QO_LEN = 1024
KV_LEN = 1024
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
IS_CAUSAL = False

FUNCTION_SIGNATURE = """\
void sageattn3_fp4_attn_hdim64(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_size,
    int num_heads,
    int qo_len,
    int kv_len,
    float sm_scale,
    int is_causal,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {
    "batch_size": BATCH_SIZE,
    "num_heads": NUM_HEADS,
    "qo_len": QO_LEN,
    "kv_len": KV_LEN,
    "sm_scale": SM_SCALE,
    "is_causal": 0,
}

TOLERANCES = {"atol": 5e-2, "rtol": 5e-2}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = [
    "reference_sources/sage_attn_v3/blackwell",
    "reference_sources/sage_attn_v3/quantization",
    "reference_sources/cutlass/include",
]
REFERENCE_EXTRA_CUDA_FLAGS = [
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-include", "cassert",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-DNDEBUG",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
]
REFERENCE_EXTRA_LDFLAGS = []


def get_inputs():
    torch.manual_seed(42)
    q = torch.randn(BATCH_SIZE, NUM_HEADS, QO_LEN, HEAD_DIM,
                     device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, KV_LEN, HEAD_DIM,
                     device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, KV_LEN, HEAD_DIM,
                     device="cuda", dtype=torch.float16)
    return [
        ("Q", q.contiguous().view(-1)),
        ("K", k.contiguous().view(-1)),
        ("V", v.contiguous().view(-1)),
    ]


def get_outputs(inputs):
    out = torch.empty(BATCH_SIZE * NUM_HEADS * QO_LEN * HEAD_DIM,
                      device="cuda", dtype=torch.bfloat16)
    return [("O", out)]


def reference_fn(inputs):
    """PyTorch reference: standard FP16 attention (approximate, no FP4 quantization)."""
    q = inputs[0][1].view(BATCH_SIZE, NUM_HEADS, QO_LEN, HEAD_DIM).float()
    k = inputs[1][1].view(BATCH_SIZE, NUM_HEADS, KV_LEN, HEAD_DIM).float()
    v = inputs[2][1].view(BATCH_SIZE, NUM_HEADS, KV_LEN, HEAD_DIM).float()

    attn = torch.matmul(q, k.transpose(-2, -1)) * SM_SCALE
    
    
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return [("O", out.to(torch.bfloat16).reshape(-1))]


DESCRIPTION = """\
Implement a CUDA kernel for FP4 quantized attention on Blackwell GPUs (SM120).

This is a SageAttention v3 pipeline: quantize FP16 Q/K/V to NVFP4 (E2M1) with
per-16-element FP8 block scales, then compute attention using BlockScaled FP4
MMA instructions (mma.sync.aligned.kind::mxf4nvf4.block_scale).

Full pipeline:
  1. Quantize Q [B,H,S,D] FP16 -> FP4 packed [B,H,S,D/2] + FP8 scales [B,H,S,D/16]
  2. Quantize K with permutation for MMA-friendly access pattern
  3. Quantize V with transpose: [B,H,S,D] -> [B,H,D,S] (column-wise for P@V)
  4. Run BlockScaled FP4 attention:
     a. S = Q_fp4 @ K_fp4^T using BlockScaled MMA (16x32x64)
     b. P = softmax(S * sm_scale) 
     c. O = P @ V_fp4 using BlockScaled MMA
  5. Output O in BF16

Parameters:
  Q:  [2 * 16 * 1024 * 64] float16 -- queries
  K:  [2 * 16 * 1024 * 64] float16 -- keys
  V:  [2 * 16 * 1024 * 64] float16 -- values
  O:  [2 * 16 * 1024 * 64] bfloat16 -- output
  batch_size:   2
  num_heads:    16
  qo_len:       1024
  kv_len:       1024
  sm_scale:     1/sqrt(64) = 0.125000
  is_causal:    0


Head dimension: 64
MMA instruction: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
Block tile: 128x128 (M x N)
Requires: SM120+ (Blackwell) for BlockScaled FP4 MMA
"""
