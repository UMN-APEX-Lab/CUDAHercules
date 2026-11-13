"""SM120 FP8 FlashAttention that reads K and V straight from a paged e4m3 KV cache (TMA + mbarrier, warp-specialised).

Differences from fa_fp8_tma.py:
- K and V come from the paged cache: global views (P, D, Hk, NB) with P the page size (a multiple of 16), D contiguous,
  NB the number of physical pages; a per-request page table maps logical 16-key sub-blocks to physical pages. Every
  64-key tile is 4 TMA boxes of 16 rows per 128-column half (8 TMA ops per operand per tile), so any P works and the
  tile may straddle pages.
- V is consumed in its natural [key][d] layout: the B fragment of the PV MMA is loaded with the SM120 8-bit
  transposing ldmatrix (ldmatrix.m16n16.trans.b8). The key order that P's in-register A fragment expects (VT_KEY_PERM)
  is produced by composing the smem tile with a permutation layout, so the host neither transposes nor permutes V.
- seqlen_k is an explicit argument (the cache view's row count is the page size). Keys beyond seqlen_k inside the last
  tile are masked; the cache rows they occupy may belong to other requests and must hold finite e4m3 values.
- Q is quantised per chunk by the host (q8, s_q) as before; K and V use the cache's static scales s_k, s_v.
smem is unchanged: Q 32 KB + K 2x16 KB + V 2x16 KB.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline

from .fa_fp8 import FP8, P_SCALE
from .fa_fp8_tma import FlashAttentionFP8Sm120Tma, STAGES, MMA_REGS, PRODUCER_REGS

SUB = 16            # rows per TMA box: 16-key sub-blocks, the finest page granularity vLLM uses


class FlashAttentionFP8Sm120Paged(FlashAttentionFP8Sm120Tma):
    def __init__(self, head_dim: int, m_block_size: int = 128, n_block_size: int = 64, is_causal: bool = True,
                 qhead_per_kvhead: int = 1, block_sparse: bool = False, page_size: int = 16, vview: str = 'perm'):
        super().__init__(head_dim, m_block_size, n_block_size, is_causal, qhead_per_kvhead, block_sparse)
        if page_size % SUB != 0:
            raise ValueError('page size must be a multiple of 16')
        self._page_size = page_size
        # vview: how the (d, key) B-operand view of a V stage is built (experiment switch)
        #   'perm'    composition with the VT_KEY_PERM layout (production intent)
        #   'compose' composition with a plain transpose layout, no key permutation (needs host-permuted V)
        #   'select'  mode swap of the swizzled layout, no key permutation (needs host-permuted V)
        self._vview = vview

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mPages: cute.Tensor,
                 seqlen_k: cutlass.Int32, softmax_scale: cutlass.Float32, out_scale: cutlass.Float32,
                 q_offset: cutlass.Int32, mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32,
                 stream: cuda.CUstream):
        """mQ (Sq, D, H) e4m3; mK, mV (P, D, Hk, NB) e4m3 cache views; mO (Sq, H, D) bf16; mPages int32 [npages]."""
        self._out_dtype = mO.element_type
        m, n, d = self._m_block_size, self._n_block_size, self._head_dim_padded
        row_major = cutlass.utils.LayoutEnum.ROW_MAJOR
        sQ_layout = sm90_utils.make_smem_layout_a(row_major, (m, n, d), FP8, 1)          # (m, d, 1)
        sK_layout = sm90_utils.make_smem_layout_b(row_major, (m, n, d), FP8, STAGES)     # (n, d, S); V uses the same
        sK16_layout = cute.slice_(sm90_utils.make_smem_layout_b(row_major, (m, SUB, d), FP8, 1), (None, None, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_Q, tQ = cpasync.make_tiled_tma_atom(op, mQ, cute.slice_(sQ_layout, (None, None, 0)), (m, d))
        tma_K, tK = cpasync.make_tiled_tma_atom(op, mK, sK16_layout, (SUB, d))
        tma_V, tV = cpasync.make_tiled_tma_atom(op, mV, sK16_layout, (SUB, d))
        q_bytes = cute.size_in_bytes(FP8, cute.slice_(sQ_layout, (None, None, 0)))
        k_bytes = cute.size_in_bytes(FP8, cute.slice_(sK_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_k: cute.struct.MemRange[cutlass.Int64, 2 * STAGES]
            mbar_v: cute.struct.MemRange[cutlass.Int64, 2 * STAGES]
            sQ: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sK_layout)], 1024]

        tiled_mma = cute.make_tiled_mma(warp.MmaFP8Op(FP8, cutlass.Float32, (16, 8, 32)),
                                        (self._mma_warps, 1, 1), permutation_mnk=(self._mma_warps * 16, 16, 32))
        grid_dim = (cute.ceil_div(mQ.shape[0], m), cute.size(mQ.shape[2]), 1)
        LOG2_E = 1.4426950408889634074
        self.kernel(tma_Q, tQ, tma_K, tK, tma_V, tV, mO, mPages, seqlen_k, softmax_scale * LOG2_E, out_scale, q_offset,
                    mCounts, mLists, list_stride, sQ_layout, sK_layout, tiled_mma, q_bytes, k_bytes,
                    SharedStorage).launch(grid=grid_dim, block=[self._cta_threads, 1, 1], stream=stream,
                                          min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tma_Q: cute.CopyAtom, mQ: cute.Tensor, tma_K: cute.CopyAtom, mK: cute.Tensor,
               tma_V: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor, mPages: cute.Tensor, seqlen_k: cutlass.Int32,
               softmax_scale_log2: cutlass.Float32, out_scale: cutlass.Float32, q_offset: cutlass.Int32,
               mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout,
               tiled_mma: cute.TiledMma, q_bytes: cutlass.Constexpr, k_bytes: cutlass.Constexpr,
               SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, num_head, _ = cute.arch.block_idx()
        kv_head = num_head // self._qhead_per_kvhead
        seqlen_q = mQ.shape[0]
        PAGE = self._page_size
        n_block_max = cute.ceil_div(seqlen_k, self._n_block_size)
        if cutlass.const_expr(self._is_causal):
            n_block_max = min(cute.ceil_div((m_block + 1) * self._m_block_size + q_offset, self._n_block_size), n_block_max)
        list_begin = cutlass.Int32(0)
        list_count = n_block_max
        if cutlass.const_expr(self._block_sparse):
            m_blocks = cute.ceil_div(seqlen_q, self._m_block_size)
            list_row = num_head * m_blocks + m_block
            list_begin = list_row * list_stride
            list_count = mCounts[list_row]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sK_layout)
        mbar_q = storage.mbar_q.data_ptr()
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_q, 1)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._mma_warps)
        pipe_k = pipeline.PipelineTmaAsync.create(num_stages=STAGES, producer_group=producer_group,
                                                  consumer_group=consumer_group, tx_count=k_bytes,
                                                  barrier_storage=storage.mbar_k.data_ptr(), defer_sync=True)
        pipe_v = pipeline.PipelineTmaAsync.create(num_stages=STAGES, producer_group=producer_group,
                                                  consumer_group=consumer_group, tx_count=k_bytes,
                                                  barrier_storage=storage.mbar_v.data_ptr(), defer_sync=True)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # TMA partitions. Q: (m, d, m_blocks, 1, H). K/V cache: 16-row boxes, (16, d, P/16, 1, Hk, NB);
        # smem K/V stages are split into the same 16-row sub-tiles: (16, d, 4, 1, S).
        gQ = cute.local_tile(mQ, (self._m_block_size, self._head_dim_padded), (None, None, None))
        gK = cute.local_tile(mK, (SUB, self._head_dim_padded), (None, None, None, None))
        gV = cute.local_tile(mV, (SUB, self._head_dim_padded), (None, None, None, None))
        sK_sub = cute.local_tile(sK, (SUB, self._head_dim_padded), (None, None, None))
        sV_sub = cute.local_tile(sV, (SUB, self._head_dim_padded), (None, None, None))
        one = cute.make_layout(1)
        tQsQ, tQgQ = cpasync.tma_partition(tma_Q, 0, one, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cpasync.tma_partition(tma_K, 0, one, cute.group_modes(sK_sub, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cpasync.tma_partition(tma_V, 0, one, cute.group_modes(sV_sub, 0, 2), cute.group_modes(gV, 0, 2))

        if warp_idx >= self._mma_warps:
            # ---------------- producer warp group ----------------
            cute.arch.setmaxregister_decrease(PRODUCER_REGS)
            if warp_idx == self._mma_warps:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_q, q_bytes)
                cute.copy(tma_Q, tQgQ[(None, m_block, 0, num_head)], tQsQ[(None, 0)], tma_bar_ptr=mbar_q)
                pk = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                pv = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                npages = mPages.shape[0]
                for n_tile in range(list_count):
                    n_block = n_block_max - n_tile - 1
                    if cutlass.const_expr(self._block_sparse):
                        n_block = mLists[list_begin + list_count - 1 - n_tile]
                    pipe_k.producer_acquire(pk)
                    for i in cutlass.range_constexpr(self._n_block_size // SUB):
                        g = n_block * self._n_block_size + i * SUB                 # first logical key of the box
                        page = mPages[cutlass.min(g // PAGE, npages - 1)]
                        sub = (g % PAGE) // SUB
                        cute.copy(tma_K, tKgK[(None, sub, 0, kv_head, page)], tKsK[(None, i, 0, pk.index)],
                                  tma_bar_ptr=pipe_k.producer_get_barrier(pk))
                    pipe_k.producer_commit(pk)
                    pk.advance()
                    pipe_v.producer_acquire(pv)
                    for i in cutlass.range_constexpr(self._n_block_size // SUB):
                        g = n_block * self._n_block_size + i * SUB
                        page = mPages[cutlass.min(g // PAGE, npages - 1)]
                        sub = (g % PAGE) // SUB
                        cute.copy(tma_V, tVgV[(None, sub, 0, kv_head, page)], tVsV[(None, i, 0, pv.index)],
                                  tma_bar_ptr=pipe_v.producer_get_barrier(pv))
                    pipe_v.producer_commit(pv)
                    pv.advance()
                pipe_k.producer_tail(pk)
                pipe_v.producer_tail(pv)
        else:
            # ---------------- MMA warps ----------------
            cute.arch.setmaxregister_increase(MMA_REGS)
            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ[None, None, 0]))
            half_d = self._head_dim_padded // 2
            acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)), cutlass.Float32)
            acc_O0 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, half_d)), cutlass.Float32)
            acc_O1 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, half_d)), cutlass.Float32)
            acc_O0.fill(0.0)
            acc_O1.fill(0.0)
            tOrP = cute.make_rmem_tensor(thr_mma.partition_shape_A((self._m_block_size, self._n_block_size)), FP8)
            ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), FP8)
            ldsm_t = cute.make_copy_atom(warp.LdMatrix16x16x8bOp(transpose=True, num_matrices=2), FP8)
            copy_sQ = cute.make_tiled_copy_A(ldsm, tiled_mma)
            copy_sK = cute.make_tiled_copy_B(ldsm, tiled_mma)
            # The transposing atom is tiled by its own thread/value layout (one atom = 16 d x 32 keys): going through
            # make_tiled_copy_B would re-derive lane addresses through the MMA's register order, which differs from the
            # atom's (n-half before k-half) and yields half-row sources. Instead the MMA's B fragment registers get a
            # second view in the atom's register order (vfrag_layout): e + 8*n8 + 4*khalf + 16*n16 + half_d*k32.
            copy_sV = cute.make_tiled_copy(ldsm_t, ldsm_t.layout_dst_tv, (16, 32))
            vfrag_layout = cute.make_layout((((4, 2, 2), 1), half_d // 16, self._n_block_size // 32),
                                            stride=(((1, 8, 4), 0), 16, half_d))
            tSsQ, tSrQ_view = copy_sQ.get_slice(tidx).partition_S(sQ[None, None, 0]), copy_sQ.get_slice(tidx).retile(tSrQ)
            # (d, key) view of a V stage with keys permuted inside every 32: logical key i + 2b + 4t + 16h (P's A-fragment
            # slot order, see VT_KEY_PERM) reads physical key i + 8b + 2t + 16h.
            vperm = cute.make_layout((self._head_dim_padded, (2, 2, 4, 2, self._n_block_size // 32)),
                                     stride=(self._n_block_size, (1, 8, 2, 16, 32)))
            mcS = cute.make_identity_tensor((1, seqlen_q, mQ.shape[2], seqlen_k))
            row_max = cute.make_rmem_tensor(2, cutlass.Float32)
            row_sum = cute.make_rmem_tensor(2, cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)
            ck = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
            cv = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
            cute.arch.mbarrier_wait(mbar_q, 0)

            from types import SimpleNamespace
            P = SimpleNamespace(num_head=num_head, m_block=m_block, n_block_max=n_block_max, list_begin=list_begin,
                                list_count=list_count, mLists=mLists, q_offset=q_offset, seqlen_k=seqlen_k,
                                softmax_scale_log2=softmax_scale_log2, tidx=tidx, sK=sK, sV=sV, vperm=vperm,
                                vfrag_layout=vfrag_layout, half_d=half_d,
                                pipe_k=pipe_k, pipe_v=pipe_v,
                                tiled_mma=tiled_mma, thr_mma=thr_mma, copy_sQ=copy_sQ, copy_sK=copy_sK, copy_sV=copy_sV,
                                tSsQ=tSsQ, tSrQ_view=tSrQ_view, tSrQ=tSrQ, tOrP=tOrP, mcS=mcS,
                                acc_S=acc_S, acc_O0=acc_O0, acc_O1=acc_O1, row_max=row_max, row_sum=row_sum)
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size) + 1
            for n_tile in cutlass.range_constexpr(mask_steps):
                if n_tile < list_count:
                    ck = self._qk_tile(P, n_tile, acc_S, ck)
                    self._softmax_tile(P, n_tile, True)
                    cv = self._pv_tile_paged(P, cv)
            for n_tile in range(mask_steps, list_count, 1):
                ck = self._qk_tile(P, n_tile, acc_S, ck)
                self._softmax_tile(P, n_tile, False)
                cv = self._pv_tile_paged(P, cv)

            # epilogue: O * out_scale / row_sum -> bf16, direct global stores
            acc_O0_mn = self._make_acc_tensor_mn_view(acc_O0)
            acc_O1_mn = self._make_acc_tensor_mn_view(acc_O1)
            for r in cutlass.range_constexpr(cute.size(row_sum)):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                bad = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
                inv = out_scale if bad else out_scale * cute.arch.rcp_approx(row_sum[r])
                acc_O0_mn[r, None] = acc_O0_mn[r, None].load() * inv
                acc_O1_mn[r, None] = acc_O1_mn[r, None].load() * inv
            mcO = cute.make_identity_tensor((self._m_block_size, half_d))
            tOcO = thr_mma.partition_C(mcO)
            for i in cutlass.range_constexpr(cute.size(acc_O0)):
                coord = tOcO[i]
                orow = m_block * self._m_block_size + coord[0]
                if orow < seqlen_q:
                    mO[orow, num_head, coord[1]] = acc_O0[i].to(self._out_dtype)
                    mO[orow, num_head, half_d + coord[1]] = acc_O1[i].to(self._out_dtype)

    @cute.jit
    def _pv_tile_paged(self, P, cv):
        """O += P V from the [key][d] V stage cv.index via the 8-bit transposing ldmatrix; releases the stage."""
        P.pipe_v.consumer_wait(cv)
        sV_s = P.sV[None, None, cv.index]
        if cutlass.const_expr(self._vview == 'perm'):
            sV_T = cute.composition(sV_s, P.vperm)                              # (d, keys) view, keys permuted
        elif cutlass.const_expr(self._vview == 'compose'):
            sV_T = cute.composition(sV_s, cute.make_layout((self._head_dim_padded, self._n_block_size),
                                                           stride=(self._n_block_size, 1)))
        else:
            sV_T = cute.make_tensor(sV_s.iterator, cute.select(sV_s.layout, [1, 0]))
        sV0 = cute.local_tile(sV_T, (P.half_d, self._n_block_size), (0, 0))
        sV1 = cute.local_tile(sV_T, (P.half_d, self._n_block_size), (1, 0))
        tOrV0 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sV0))         # ((4,2),16,2) MMA register order
        tOrV1 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sV1))
        thr_cV = P.copy_sV.get_slice(P.tidx % 32)                             # the copy spans one warp; every warp loads all of B
        tOsV0, tOsV1 = thr_cV.partition_S(sV0), thr_cV.partition_S(sV1)       # ((16,1),8,2): one 16-B row per lane
        tOrV0_cp = cute.make_tensor(tOrV0.iterator, P.vfrag_layout)            # same registers, atom order
        tOrV1_cp = cute.make_tensor(tOrV1.iterator, P.vfrag_layout)
        for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            cute.copy(P.copy_sV, tOsV0[None, None, k], tOrV0_cp[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], tOrV0[None, None, k], P.acc_O0)
            cute.copy(P.copy_sV, tOsV1[None, None, k], tOrV1_cp[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], tOrV1[None, None, k], P.acc_O1)
        P.pipe_v.consumer_release(cv)
        cv.advance()
        return cv


# ---------------------------------------------------------------------------
# Torch driver
# ---------------------------------------------------------------------------
import torch  # noqa: E402

_COMPILED_PAGED = {}


def attention_fp8_paged(q8, sq, k_cache, v_cache, sk, sv, page_table, seqlen_k, q_offset, *, counts=None,
                              lists=None, m_block=128, out=None, vview='perm'):
    """Block-sparse / dense causal attention over a paged e4m3 KV cache.

    q8 [n, H, D] e4m3 with scale sq (q = q8 * sq); k_cache, v_cache: e4m3 views [P, D, Hk, NB] of the cache (page rows,
    head_dim contiguous, kv head, physical page; any strides) with static scales sk, sv; page_table int32 [npages]:
    this request's physical pages in order; seqlen_k: keys of the request; q_offset: absolute position of q[0].
    counts/lists as in the TMA kernel driver (128-row query blocks). Returns [n, H, D] bf16.
    Keys beyond seqlen_k are masked but their cache rows are still loaded, so they must hold finite e4m3 values.
    """
    n, H, D = q8.shape
    P, Dc, Hk, NB = k_cache.shape
    assert Dc == D and P % SUB == 0 and k_cache.stride(1) == 1 and v_cache.stride(1) == 1 and tuple(v_cache.shape) == tuple(k_cache.shape)
    assert q8.dtype == torch.float8_e4m3fn and k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    assert page_table.dtype == torch.int32 and page_table.dim() == 1
    sparse = counts is not None
    if not sparse:
        counts = torch.zeros(1, dtype=torch.int32, device=q8.device)
        lists = torch.zeros(1, 1, dtype=torch.int32, device=q8.device)
    list_stride = lists.shape[1]
    if out is None:
        out = torch.empty(n, H, D, dtype=torch.bfloat16, device=q8.device)
    key = (D, H // Hk, sparse, m_block, P, vview)
    fa = _COMPILED_PAGED.get(key)
    qv = q8.permute(0, 2, 1)                                                    # (n, d, heads), d contiguous
    tensors = [from_dlpack(qv, assumed_align=16).mark_layout_dynamic(leading_dim=1)
               .mark_compact_shape_dynamic(mode=1, stride_order=qv.dim_order(), divisibility=16),
               from_dlpack(k_cache, assumed_align=16).mark_layout_dynamic(leading_dim=1),
               from_dlpack(v_cache, assumed_align=16).mark_layout_dynamic(leading_dim=1),
               from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=2)
               .mark_compact_shape_dynamic(mode=2, stride_order=out.dim_order(), divisibility=8)]
    pages = from_dlpack(page_table, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    bsr = [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) for t in (counts, lists.reshape(-1))]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = cutlass.Float32(D ** -0.5 * sq * sk)
    out_scale = cutlass.Float32(sv / P_SCALE)
    args = (*tensors, pages, cutlass.Int32(seqlen_k), softmax_scale, out_scale, cutlass.Int32(q_offset), *bsr,
            cutlass.Int32(list_stride), stream)
    if fa is None:
        obj = FlashAttentionFP8Sm120Paged(D, m_block, 64, True, H // Hk, sparse, P, vview)
        fa = cute.compile(obj, *args)
        _COMPILED_PAGED[key] = fa
    fa(*args)
    return out
