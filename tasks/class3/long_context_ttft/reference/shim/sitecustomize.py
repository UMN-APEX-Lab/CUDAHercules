"""Runtime hook for vLLM's FlashInfer backend (FP8 KV cache) without touching site-packages.

Activate with  PYTHONPATH=<reference>:<reference>/shim  on the `vllm serve` command line; Python imports this file at
interpreter start in every process (including vLLM's spawned workers). It registers an import hook that, once
vllm.v1.attention.backends.flashinfer is loaded, wraps
  - FlashInferMetadataBuilder.build   -> stash_common_metadata (per-request host facts, once per step)
  - FlashInferImpl.forward            -> try_sparse_forward_fi for the prefill requests; the original forward then runs
                                         for the decode tokens only (metadata copy with num_prefill_tokens = 0)
LCP_SPARSE_ATTN=0 disables the hook at runtime.
"""
import copy
import importlib.abc
import importlib.util
import sys

_TARGET = 'vllm.v1.attention.backends.flashinfer'


def _install(module):
    import torch
    from lcp_ref.sparse_prefill import stash_common_metadata, try_sparse_forward_fi

    Builder, Impl = module.FlashInferMetadataBuilder, module.FlashInferImpl
    if getattr(Impl, '_lcp_hooked', False):
        return
    orig_build, orig_forward = Builder.build, Impl.forward

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        return stash_common_metadata(md, common_attn_metadata, page_size=getattr(self, 'page_size', None))

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        md = attn_metadata
        if (md is None or md.use_cascade or md.num_prefill_tokens == 0 or not hasattr(md, 'lcp_seq_lens')
                or output_scale is not None or getattr(self, 'is_kvcache_nvfp4', False)):
            return orig_forward(self, layer, query, key, value, kv_cache, md, output, output_scale, output_block_scale)
        n_act = md.num_actual_tokens
        kv = kv_cache.view(torch.float8_e4m3fn) if kv_cache.dtype == torch.uint8 else kv_cache
        k4, v4 = kv.permute(*self.kv_cache_layout.layer_view_order).split(self.head_size, dim=-1)
        if try_sparse_forward_fi(self, layer, query[:n_act], key[:n_act] if key is not None else None, k4, v4,
                                 output[:n_act], md):
            if md.num_decode_tokens == 0:
                return output
            md = copy.copy(md)
            md.num_prefills, md.num_prefill_tokens, md.prefill = 0, 0, None
        return orig_forward(self, layer, query, key, value, kv_cache, md, output, output_scale, output_block_scale)

    Builder.build, Impl.forward, Impl._lcp_hooked = build, forward, True
    print('[lcp-sparse] FlashInfer backend hooked (runtime shim)', flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    _busy = False

    def find_spec(self, name, path, target=None):
        if name != _TARGET or _Finder._busy:
            return None
        _Finder._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            _Finder._busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return None
        loader, orig_exec = spec.loader, spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _install(module)
        loader.exec_module = exec_module
        return spec


if _TARGET in sys.modules:
    _install(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _Finder())
