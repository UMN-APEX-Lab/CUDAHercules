"""Dynamic block-sparse causal prefill attention for long contexts (SM120).

Per full-attention layer and prefill chunk: mean-pooled q/k block scores, per-head top-mass (tau) block selection
from cached K block means, then the paged FP8 FlashAttention kernel over the selected 64-key blocks (the diagonal
block is always kept and is placed last in each list).
"""
from __future__ import annotations

import json
import math
import os

import torch

BLOCK = 64


class SparsePrefill:
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, device,
                 tau: float = 0.95, min_kv: int = 8192, min_q: int = 1024,
                 workspace_bytes: int = 256 << 20, kernel: str = 'fp8'):
        self.kernel = kernel
        self.H, self.Hk, self.D = num_heads, num_kv_heads, head_dim
        self.g = num_heads // num_kv_heads
        self.tau, self.min_kv, self.min_q = tau, min_kv, min_q
        self.scale = head_dim ** -0.5
        self.stats = {'calls': 0, 'blocks_total': 0, 'blocks_kept': 0, 'plan_ms': 0.0, 'run_ms': 0.0, 'select_ms': 0.0}
        self._kept_acc = None
        self.timing = False

    def eligible(self, q_len: int, kv_len: int) -> bool:
        return q_len >= self.min_q and kv_len >= self.min_kv

    @torch.no_grad()
    def block_masks(self, q: torch.Tensor, k, q_start: int, valid=None, union: bool = True, km=None):
        """q [n, H, D]; k [Hk, N, D] (head-major) or [N, Hk, D] (NHD) with N a multiple of BLOCK;
        q_start = absolute position of q[0]. Returns bool masks [Hk, MB, NB] (union over the query
        heads of each KV head) or [H, MB, NB] per query head when union=False.
        """
        n = q.shape[0]
        if km is not None:                                   # cached block means [Hk, NB, D] (fp32)
            NB = km.shape[1]
            head_major = None
        else:
            head_major = k.shape[0] == self.Hk and k.shape[1] != self.Hk
            N = k.shape[1] if head_major else k.shape[0]
            NB = N // BLOCK
        MB = -(-n // BLOCK)
        dev = q.device
        assert n % BLOCK == 0 and q_start % BLOCK == 0, 'callers pad q to whole blocks'
        if valid is None:
            valid = (0, n)
        rows = torch.arange(n, device=dev)
        real = ((rows >= valid[0]) & (rows < valid[0] + valid[1])).float()
        counts = real.view(MB, BLOCK).sum(1).clamp_min(1.0)
        qm = (q.float() * real[:, None, None]).view(MB, BLOCK, self.H, self.D).sum(1) / counts[:, None, None]
        if km is None and head_major:
            km = k.view(self.Hk, NB, BLOCK, self.D).float().mean(2)                  # [Hk, NB, D]
        elif km is None:
            km = k.view(NB, BLOCK, self.Hk, self.D).float().mean(1).transpose(0, 1)  # [Hk, NB, D]
        gi = (q_start // BLOCK) + torch.arange(MB, device=dev)                       # diagonal block per q block
        j = torch.arange(NB, device=dev)
        future = j[None, :] > gi[:, None]                                            # [MB, NB]
        qg = qm.view(MB, self.Hk, self.g, self.D)
        logits = torch.einsum('mhgd,hnd->hmgn', qg, km) * self.scale                 # [Hk, MB, g, NB]
        logits.masked_fill_(future[None, :, None, :], float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        srt, order = torch.sort(probs, dim=-1, descending=True)
        keep = (srt.cumsum(-1) < self.tau).sum(-1, keepdim=True) + 1                 # blocks per (h, m, head)
        sel = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, order, j[None, None, None, :] < keep)
        if union:
            masks = sel.any(2)                                                       # [Hk, MB, NB]
        else:
            masks = sel.permute(0, 2, 1, 3).reshape(self.H, MB, NB)                  # [H, MB, NB], head = kvh * g + i
        masks[:, torch.arange(MB, device=dev), gi] = True                            # diagonal
        masks[:, :, 0] = True                                                        # sink
        masks &= ~future[None]
        return masks

    @torch.no_grad()
    def forward_fp8_paged(self, q, k_view, v_view, page_table, kv_len, q_start, out=None, km=None, q_scale=None,
                               k_scale=1.0, v_scale=1.0, dense=False):
        """FP8 kernel reading K/V straight from a paged e4m3 cache (no gather, quantisation or transpose).

        q [n, H, D] bf16 at absolute positions q_start.. with kv_len == q_start + n; k_view/v_view: (P, D, Hk, NB) e4m3
        views of the cache (any strides, D contiguous) with the cache's static scales k_scale/v_scale; page_table int32
        [npages]: this request's physical pages; km: cached K block means [Hk, NB, D] (required unless dense);
        q_scale: python float (cached per request by the caller; None = per-call amax, one device sync).
        """
        from .kernels.fa_fp8_paged import attention_fp8_paged
        BM = 128
        n = q.shape[0]
        assert kv_len == q_start + n, (kv_len, q_start, n)
        front = q_start % BM
        end = (-(front + n)) % BM
        if front or end:
            q = torch.cat([q.new_zeros(front, self.H, self.D), q, q.new_zeros(end, self.H, self.D)])
        M = q.shape[0]
        MB = M // BLOCK
        qs = q_start - front
        t0 = self._tick()
        counts = lists = None
        if not dense:
            if km is None:
                raise ValueError('the paged path needs cached K block means (km)')
            nb_need = -(-(qs + M) // BLOCK)                  # padded query range may reach beyond the real keys
            if km.shape[1] < nb_need:
                km = torch.nn.functional.pad(km, (0, 0, 0, nb_need - km.shape[1]))
            masks = self.block_masks(q, None, qs, valid=(front, n), union=False, km=km)   # [H, MB(64), NB]
            masks = masks.view(self.H, MB // 2, 2, -1).any(2)                            # 128-row blocks
            counts = masks.sum(2).to(torch.int32).flatten()
            lists = torch.argsort((~masks).to(torch.int8), dim=-1, stable=True).to(torch.int32).reshape(self.H * (MB // 2), -1)
        if q_scale is None:
            q_scale = float(q.float().abs().amax().clamp_min(1e-6).item()) / 448.0
        q8 = (q * (1.0 / q_scale)).to(torch.float8_e4m3fn)
        t1 = self._tick()
        o = attention_fp8_paged(q8, q_scale, k_view, v_view, k_scale, v_scale, page_table, kv_len, qs,
                                      counts=counts, lists=lists, m_block=BM)[front:front + n]
        t2 = self._tick()
        if self.timing:
            self.stats['select_ms'] += t1 - t0
            self.stats['run_ms'] += t2 - t1
        self.stats['calls'] += 1
        if not dense:
            self.stats['blocks_total'] += int(self.H * ((qs // BLOCK) * MB + MB * (MB + 1) // 2))
            self._kept_acc = counts.sum() * 2 if self._kept_acc is None else self._kept_acc + counts.sum() * 2
        if out is None:
            return o
        out.copy_(o)
        return out

    def kept_fraction(self):
        """Blocks kept / causal blocks over all calls so far (one device sync)."""
        if self._kept_acc is None or not self.stats['blocks_total']:
            return None
        return int(self._kept_acc.item()) / self.stats['blocks_total']

    def _tick(self):
        if not self.timing:
            return 0.0
        torch.cuda.synchronize()
        import time
        return time.time() * 1000.0

# ---------------------------------------------------------------------------
# vLLM glue: dynamic block-sparse path for long prefill chunks in the FlashInfer
# backend (FP8 KV cache). Enabled by LCP_SPARSE_ATTN (default on);
# LCP_SPARSE_TAU, LCP_SPARSE_MIN_KV, LCP_SPARSE_MIN_Q tune it.
# ---------------------------------------------------------------------------
_CONFIG = None
_STATE: dict = {}


_DEBUG_LEFT = [int(os.environ.get('LCP_SPARSE_DEBUG', '0'))]


def _debug(msg):
    if _DEBUG_LEFT[0] > 0:
        _DEBUG_LEFT[0] -= 1
        print('[lcp-sparse] ' + msg, flush=True)


def config():
    """LCP_SPARSE_* environment variables, with optional defaults from the JSON file named by
    LCP_SPARSE_CONFIG (useful when a launcher does not forward environment variables)."""
    global _CONFIG
    if _CONFIG is None:
        file_cfg = {}
        path = os.environ.get('LCP_SPARSE_CONFIG')
        if path:
            with open(path) as f:
                file_cfg = json.load(f)
        def pick(env, key, default, cast):
            if env in os.environ:
                return cast(os.environ[env])
            return cast(file_cfg.get(key, default))
        _CONFIG = {
            'enabled': pick('LCP_SPARSE_ATTN', 'enabled', 1, lambda v: str(v) not in ('0', 'false', 'False')),
            'tau': pick('LCP_SPARSE_TAU', 'tau', 0.95, float),
            'min_kv': pick('LCP_SPARSE_MIN_KV', 'min_kv', 8192, int),
            'min_q': pick('LCP_SPARSE_MIN_Q', 'min_q', 1024, int),
            'kernel': pick('LCP_SPARSE_KERNEL', 'kernel', 'fp8', str),
        }
        print('[lcp-sparse] config %r' % (_CONFIG,), flush=True)
    return _CONFIG


def _state(num_heads, num_kv_heads, head_dim, device):
    key = (num_heads, num_kv_heads, head_dim, str(device))
    if key not in _STATE:
        cfg = config()
        _STATE[key] = SparsePrefill(num_heads, num_kv_heads, head_dim, device, tau=cfg['tau'],
                                    min_kv=cfg['min_kv'], min_q=cfg['min_q'], kernel=cfg['kernel'])
    return _STATE[key]


_KMEAN_CACHE: dict = {}
_KMEAN_MAX_ENTRIES = 128


def _cached_block_means(layer_key, first_page, key_chunk, q_start, kv_len, key_all=None):
    """Maintain per-(layer, request) K block means [Hk, NB, D] (fp32), updated from the chunk's keys.

    A request is identified by its first physical page. An entry is only trusted when it was built
    contiguously by this request (its last chunk ended exactly at q_start); otherwise - new request,
    or a request whose first chunks were too short for the sparse path and whose first page was
    recycled from an earlier request - the entry is rebuilt from the full key history `key_all`
    [kv_len, Hk, D] (always available: the glue gathers the whole KV history of the request).
    Chunk boundaries unaligned to BLOCK make the boundary block's mean use only the rows seen so far.
    """
    n, Hk, D = key_chunk.shape
    cache_key = (layer_key, first_page)
    entry = _KMEAN_CACHE.get(cache_key)
    b0, b1 = q_start // BLOCK, -(-(q_start + n) // BLOCK)
    if entry is None or q_start == 0 or entry['nb'] < b0 or entry.get('last_kv') != q_start:
        if len(_KMEAN_CACHE) >= _KMEAN_MAX_ENTRIES:
            _KMEAN_CACHE.pop(next(iter(_KMEAN_CACHE)))
        cap = max(4096, b1)
        entry = {'km': torch.zeros(Hk, cap, D, dtype=torch.float32, device=key_chunk.device), 'nb': 0, 'last_kv': 0}
        _KMEAN_CACHE[cache_key] = entry
        if q_start > 0:
            if callable(key_all):
                key_all = key_all()
            if key_all is None:
                raise RuntimeError('stale block-mean cache entry and no key history to rebuild it')
            # rebuild the means of the blocks before this chunk from the request's own history
            hist = key_all[:q_start]
            full, rem = q_start // BLOCK, q_start % BLOCK
            if full:
                entry['km'][:, :full] = hist[:full * BLOCK].float().view(full, BLOCK, Hk, D).mean(1).transpose(0, 1)
            if rem:   # partial block: the chunk below completes it from `front` rows of its own data
                entry['km'][:, full] = hist[full * BLOCK:].float().sum(0) / rem
            entry['nb'] = full + (1 if rem else 0)
            key_chunk = key_chunk if rem == 0 else torch.cat([hist[full * BLOCK:], key_chunk], 0)
            if rem:
                q_start, n = full * BLOCK, n + rem
                b0 = full
    if entry['km'].shape[1] < b1:
        grown = torch.zeros(Hk, max(b1, 2 * entry['km'].shape[1]), D, dtype=torch.float32, device=key_chunk.device)
        grown[:, :entry['km'].shape[1]] = entry['km']
        entry['km'] = grown
    front = q_start - b0 * BLOCK
    rows = (b1 - b0) * BLOCK
    padded = key_chunk.new_zeros(rows, Hk, D, dtype=torch.float32)
    padded[front:front + n] = key_chunk.float()
    valid = torch.zeros(rows, device=key_chunk.device, dtype=torch.float32)
    valid[front:front + n] = 1.0
    cnt = valid.view(b1 - b0, BLOCK).sum(1).clamp_min(1.0)
    entry['km'][:, b0:b1] = (padded.view(b1 - b0, BLOCK, Hk, D).sum(1) / cnt[:, None, None]).transpose(0, 1)
    entry['nb'] = b1
    entry['last_kv'] = kv_len
    return entry['km'][:, :b1]


_SCALE_CACHE: dict = {}
SCALE_MARGIN = 1.5


_QSCALE_CACHE: dict = {}


def _cached_q_scale(layer_key, first_page, q, q_start):
    """Per-(layer, request) e4m3 scale for Q (amax/448 x margin) from the request's first sparse chunk; K and V come
    pre-quantised from the FP8 KV cache with the layer's static scales. Same continuity rule as _cached_scales."""
    key = (layer_key, first_page)
    entry = _QSCALE_CACHE.get(key)
    if q_start == 0 or entry is None or entry['last_kv'] != q_start:
        if len(_QSCALE_CACHE) >= _KMEAN_MAX_ENTRIES:
            _QSCALE_CACHE.pop(next(iter(_QSCALE_CACHE)))
        amax = float(q.float().abs().amax().clamp_min(1e-6).item())
        entry = {'scale': amax * SCALE_MARGIN / 448.0, 'last_kv': 0}
        _QSCALE_CACHE[key] = entry
    entry['last_kv'] = q_start + q.shape[0]
    return entry['scale']


def _gather_keys_bf16(kview, page_table, k_scale, rows):
    """Dequantised K history [rows, Hk, D] bf16 from a (P, D, Hk, NB) e4m3 cache view (rebuild path only)."""
    P, D, Hk, NB = kview.shape
    npages = -(-rows // P)
    hist = kview[:, :, :, page_table[:npages].long()].permute(3, 0, 2, 1).reshape(npages * P, Hk, D)[:rows]
    return (hist.float() * k_scale).to(torch.bfloat16)


def _cache_views(k4, v4, num_kv_heads, page_size, layout_name=''):
    """(P, D, Hk, NB) views of the per-layer K and V cache tensors (NB, a, b, D) whatever the backend's layout:
    head-major (NB, Hk, P, D) ("HND"/"LBHNC") or page-major (NB, P, Hk, D) ("NHD"/"LBNHC")."""
    NB, a, b, D = k4.shape
    if page_size is not None and (a, b) == (num_kv_heads, page_size) and a != b:
        hnd = True
    elif page_size is not None and (a, b) == (page_size, num_kv_heads) and a != b:
        hnd = False
    elif 'HN' in str(layout_name).upper():
        hnd = True
    elif 'NH' in str(layout_name).upper():
        hnd = False
    else:
        hnd = a == num_kv_heads
    if hnd:
        return k4.permute(2, 3, 1, 0), v4.permute(2, 3, 1, 0)
    return k4.permute(1, 3, 2, 0), v4.permute(1, 3, 2, 0)


def stash_common_metadata(md, common, page_size=None):
    """Record on a FlashInferMetadata the per-request facts the hook needs (host copies, once per step).

    Called from FlashInferMetadataBuilder.build (runtime shim) with the builder's kernel page size.
    Skipped for decode-only batches and during CUDA graph capture, where it would be a device sync for nothing.
    """
    if getattr(md, 'num_prefill_tokens', 0) == 0 or torch.cuda.is_current_stream_capturing():
        return md
    md.lcp_page_size = page_size
    md.lcp_num_reqs = int(common.num_reqs)
    md.lcp_max_q = int(common.max_query_len)
    md.lcp_max_seq = int(common.max_seq_len)
    md.lcp_qsl = common.query_start_loc_cpu[:common.num_reqs + 1].tolist()
    md.lcp_block_table = common.block_table_tensor
    cfg = config()
    if cfg['enabled'] and md.lcp_max_q >= cfg['min_q'] and md.lcp_max_seq >= cfg['min_kv']:
        try:
            from vllm.utils.gpu_sync_debug import gpu_sync_allowed   # async-scheduling guard, if this vLLM has it
        except Exception:                                           # pragma: no cover
            import contextlib
            gpu_sync_allowed = contextlib.nullcontext
        with gpu_sync_allowed():
            md.lcp_seq_lens = common.seq_lens[:common.num_reqs].tolist()
            md.lcp_first_pages = common.block_table_tensor[:common.num_reqs, 0].tolist()
    return md


@torch.no_grad()
def try_sparse_forward_fi(impl, layer, query, key, k_cache4, v_cache4, output, md) -> bool:
    """FlashInfer-backend variant of the hook, for an FP8 (e4m3) KV cache read directly by the kernel.

    query/key/output: this batch's [num_actual_tokens, ...] tensors; k_cache4/v_cache4: (NB, Hk, P, D) e4m3 views of
    the layer's cache (as FlashInferImpl.forward builds them); md: FlashInferMetadata with the fields stashed by
    stash_common_metadata. FlashInfer orders decode requests first. When at least one prefill request qualifies,
    every prefill request of the batch is computed here (short chunks run the paged kernel densely) and True is
    returned; the caller then runs its own path for the decode tokens only. Returns False (nothing written) otherwise.
    """
    cfg = config()
    if not cfg['enabled'] or cfg['kernel'] != 'fp8' or not hasattr(md, 'lcp_seq_lens'):
        return False
    window = getattr(impl, 'sliding_window', None)
    windowed = window is not None and any(int(w) >= 0 for w in window)
    if md.causal is not True or windowed:
        _debug('fallback: causal=%r window=%r' % (md.causal, window))
        return False
    if md.lcp_max_q < cfg['min_q'] or md.lcp_max_seq < cfg['min_kv']:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if k_cache4.dtype != torch.float8_e4m3fn or v_cache4.dtype != torch.float8_e4m3fn:
        _debug('fallback: cache dtype %s' % k_cache4.dtype)
        return False
    qsl, seq_lens, nreq, nd = md.lcp_qsl, md.lcp_seq_lens, md.lcp_num_reqs, md.num_decodes
    eligible = [r for r in range(nd, nreq) if qsl[r + 1] - qsl[r] >= cfg['min_q'] and seq_lens[r] >= cfg['min_kv']]
    if not eligible:
        return False
    _debug('sparse path (fp8 cache): %d/%d prefill requests, q_lens=%s kv_lens=%s' % (
        len(eligible), nreq - nd, [qsl[i + 1] - qsl[i] for i in range(nd, nreq)], seq_lens[nd:nreq]))
    H, D = query.shape[1], query.shape[2]
    Hk = int(getattr(impl, 'num_kv_heads', 0)) or (k_cache4.shape[1] if k_cache4.shape[1] < k_cache4.shape[2] else k_cache4.shape[2])
    kview, vview = _cache_views(k_cache4, v_cache4, Hk, getattr(md, 'lcp_page_size', None),
                                getattr(impl, 'kv_cache_layout', ''))                 # (P, D, Hk, NB)
    P = kview.shape[0]
    sp = _state(H, Hk, D, query.device)
    bt = md.lcp_block_table
    k_scale, v_scale = float(layer._k_scale_float), float(layer._v_scale_float)
    for r in range(nd, nreq):
        kv_len = seq_lens[r]
        n = qsl[r + 1] - qsl[r]
        q_r = query[qsl[r]:qsl[r + 1]]
        o_r = output[qsl[r]:qsl[r + 1]]
        q_start = kv_len - n
        pt = bt[r, :-(-kv_len // P)]
        q_scale = _cached_q_scale(id(impl), md.lcp_first_pages[r], q_r, q_start)
        if r not in eligible:
            # short chunk of another request in the same batch: dense attention with the same paged kernel
            sp.forward_fp8_paged(q_r, kview, vview, pt, kv_len, q_start, out=o_r, dense=True,
                                      q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
            continue
        km = _cached_block_means(id(impl), md.lcp_first_pages[r], key[qsl[r]:qsl[r + 1]], q_start, kv_len,
                                 key_all=lambda: _gather_keys_bf16(kview, pt, k_scale, q_start))
        sp.forward_fp8_paged(q_r, kview, vview, pt, kv_len, q_start, out=o_r, km=km, q_scale=q_scale,
                                  k_scale=k_scale, v_scale=v_scale)
    return True

