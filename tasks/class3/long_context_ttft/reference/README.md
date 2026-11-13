# Long-context prefill reference (block-sparse FP8 attention for SM120)

Reference implementation for the long-context time-to-first-token task. It speeds up the prefill of long prompts
(64K to 256K tokens) for a vLLM 0.30.0 server on one SM120 GPU (RTX PRO 6000 Blackwell, compute capability 12.0).

What it contains:

- `lcp_ref/kernels/`: FP8 (e4m3) FlashAttention prefill kernels written in the CUTLASS CuTe DSL. `fa_fp8_paged.py` is the
  kernel that runs in production; it reads K and V in place from vLLM's paged FP8 KV cache (TMA + mbarrier pipelines,
  8 MMA warps and a producer warp group, in-kernel V transpose) and supports per-(head, query block) block lists.
  `fa_base.py`, `fa_fp8.py` and `fa_fp8_tma.py` are the parent classes it inherits from. Kernels JIT-compile on first
  use (no C++ build).
- `lcp_ref/sparse_prefill.py`: dynamic block selection. For each head and 64-row query block it scores cached 64-key
  K block means and keeps the smallest block set carrying a fraction `tau` of the softmax mass (the diagonal and the
  first block are always kept). It also holds the per-request K-mean and Q-scale caches and the hook entry point.
- `shim/sitecustomize.py`: a runtime hook. Python imports it at start-up in every process (including vLLM's spawned
  workers); once `vllm.v1.attention.backends.flashinfer` is imported it wraps `FlashInferMetadataBuilder.build` and
  `FlashInferImpl.forward`, so qualifying prefill chunks run the sparse FP8 kernel while decode tokens and short chunks
  keep vLLM's own kernels. Nothing in the vLLM installation is modified.

Configuration: vLLM's stock FP8 W8A8 GEMMs (`--quantization fp8`) and FP8 KV cache (`--kv-cache-dtype fp8`). With an FP8
KV cache on SM120 vLLM selects its FlashInfer backend, which is the one this reference hooks.

## Environment variables

Read once per process by `lcp_ref.sparse_prefill.config()`; a JSON file named by `LCP_SPARSE_CONFIG` can supply the same
keys (`enabled`, `tau`, `min_kv`, `min_q`, `kernel`) as defaults.

| variable | reference value | default | meaning |
|---|---|---|---|
| `LCP_SPARSE_ATTN` | `1` | `1` | enable the hook (`0` runs vLLM's stock attention) |
| `LCP_SPARSE_KERNEL` | `fp8` | `fp8` | kernel selector; only `fp8` is implemented |
| `LCP_SPARSE_TAU` | `0.95` | `0.95` | attention mass kept per head and query block |
| `LCP_SPARSE_MIN_Q` | (default) | `1024` | minimum chunk length for the sparse path |
| `LCP_SPARSE_MIN_KV` | (default) | `8192` | minimum context length for the sparse path |
| `LCP_SPARSE_DEBUG` | (unset) | `0` | number of diagnostic lines to print |

## Launch

Requirements are in `requirements.txt` (vLLM 0.30.0 environment with CUDA 13).

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>
export HF_HOME=<dir containing the Qwen/Qwen3.8-27B snapshot>
REF=<path to this reference directory>
PYTHONPATH=$REF:$REF/shim LCP_SPARSE_ATTN=1 LCP_SPARSE_KERNEL=fp8 LCP_SPARSE_TAU=0.95 \
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8 \
    --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only \
    --gpu-memory-utilization 0.85 --port <port>
```

On start-up each process prints `[lcp-sparse] FlashInfer backend hooked (runtime shim)` and, once the first prefill batch is seen,
`[lcp-sparse] config {...}`. Without the shim on `PYTHONPATH` the server runs vLLM's stock path.

## Notes

- The hook applies only to causal, non-windowed attention, never during CUDA-graph capture, and only to prefill requests
  with a chunk of at least `LCP_SPARSE_MIN_Q` tokens and a context of at least `LCP_SPARSE_MIN_KV` tokens. Shorter
  prefill requests in the same batch run the same kernel densely; decode tokens use vLLM's own kernels.
- K and V are used as stored in the cache with the layer's static scales (1.0 for a checkpoint without KV scales); Q is
  quantised to e4m3 per request and layer from the first sparse chunk.
- The kernels are specialised to head_dim 256 and 64-key blocks with 128-row query blocks.
- Third-party licence notices are in `THIRD_PARTY_NOTICES`.
