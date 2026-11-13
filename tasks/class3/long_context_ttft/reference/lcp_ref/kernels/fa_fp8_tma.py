"""SM120 FP8 FlashAttention with a TMA + mbarrier pipeline (warp-specialised producer).

Differences from fa_fp8.py (cp.async, CTA-wide barriers):
- 8 MMA warps + a 4-warp producer group (384 threads). Launch allocation is 168 regs/thread; the producer
  warps drop to 24 and the MMA warps rise to 240 (balanced: 4*(168-24) == 8*(240-168)); only the first
  producer warp issues TMA, the other three exit after releasing their registers.
- Q, K and V^T tiles are loaded by TMA (cp.async.bulk.tensor); K and V^T each have a 2-stage
  mbarrier pipeline (PipelineTmaAsync), so the only CTA-wide barrier is the pipeline init.
  Consumer warps wait on "full" barriers and release stages per warp; they never synchronise
  with each other, which lets the two warps of each SM sub-partition drift out of phase.
- OOB rows/keys are zero-filled by TMA (no per-copy predicates).
- P is converted in registers (permuted V^T keys, see fa_fp8.VT_KEY_PERM).
smem: Q 32 KB + K 2x16 KB + V^T 2x16 KB = 96 KB (+ barriers).
Global tensors are given as (rows, d, heads) views: Q (Sq, D, H), K (Sk, D, Hk), V^T (D, Sk, Hk), O (Sq, H, D).
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

from .fa_fp8 import FlashAttentionFP8Sm120, FP8, P_SCALE

STAGES = 2
PRODUCER_WARPS = 4
MMA_REGS = 240
PRODUCER_REGS = 24


class FlashAttentionFP8Sm120Tma(FlashAttentionFP8Sm120):
    def __init__(self, head_dim: int, m_block_size: int = 128, n_block_size: int = 64, is_causal: bool = True,
                 qhead_per_kvhead: int = 1, block_sparse: bool = False, s_pipeline: bool = False,
                 pingpong: bool = False, prefetch: bool = False):
        super().__init__(head_dim, m_block_size, n_block_size, 2 * m_block_size, is_causal, qhead_per_kvhead, block_sparse)
        self._mma_warps = self._num_threads // 32
        self._cta_threads = self._num_threads + 32 * PRODUCER_WARPS
        # s_pipeline: issue S(n+1) = Q K(n+1)^T before softmax(n) (two S accumulators) so the tensor pipe has
        # queued work while this warp runs the softmax ALU chain.
        self._s_pipeline = s_pipeline
        # pingpong: FA3-style scheduling between the two MMA warp groups (warps 0-3 / 4-7, one warp of each per SM
        # sub-partition): named barriers force group g to issue its QK^T(n) + PV(n-1) MMAs while the other group runs
        # softmax, so the tensor pipe is never idle while both warps of a sub-partition are in the ALU phase.
        self._pingpong = pingpong
        # prefetch: issue the ldmatrix loads of k-slice k+1 before the MMAs of slice k (register double buffer)
        self._prefetch = prefetch

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mVt: cute.Tensor, mO: cute.Tensor,
                 softmax_scale: cutlass.Float32, out_scale: cutlass.Float32, q_offset: cutlass.Int32,
                 mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32, stream: cuda.CUstream):
        self._out_dtype = mO.element_type
        m, n, d = self._m_block_size, self._n_block_size, self._head_dim_padded
        row_major = cutlass.utils.LayoutEnum.ROW_MAJOR
        sQ_layout = sm90_utils.make_smem_layout_a(row_major, (m, n, d), FP8, 1)          # (m, d, 1)
        sK_layout = sm90_utils.make_smem_layout_b(row_major, (m, n, d), FP8, STAGES)     # (n, d, S)
        sVt_layout = sm90_utils.make_smem_layout_b(row_major, (m, d, n), FP8, STAGES)    # (d, n, S)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_Q, tQ = cpasync.make_tiled_tma_atom(op, mQ, cute.slice_(sQ_layout, (None, None, 0)), (m, d))
        tma_K, tK = cpasync.make_tiled_tma_atom(op, mK, cute.slice_(sK_layout, (None, None, 0)), (n, d))
        tma_V, tV = cpasync.make_tiled_tma_atom(op, mVt, cute.slice_(sVt_layout, (None, None, 0)), (d, n))
        q_bytes = cute.size_in_bytes(FP8, cute.slice_(sQ_layout, (None, None, 0)))
        k_bytes = cute.size_in_bytes(FP8, cute.slice_(sK_layout, (None, None, 0)))
        v_bytes = cute.size_in_bytes(FP8, cute.slice_(sVt_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_k: cute.struct.MemRange[cutlass.Int64, 2 * STAGES]
            mbar_v: cute.struct.MemRange[cutlass.Int64, 2 * STAGES]
            sQ: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sK_layout)], 1024]
            sVt: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sVt_layout)], 1024]

        tiled_mma = cute.make_tiled_mma(warp.MmaFP8Op(FP8, cutlass.Float32, (16, 8, 32)),
                                        (self._mma_warps, 1, 1), permutation_mnk=(self._mma_warps * 16, 16, 32))
        grid_dim = (cute.ceil_div(mQ.shape[0], m), cute.size(mQ.shape[2]), 1)
        LOG2_E = 1.4426950408889634074
        self.kernel(tma_Q, tQ, tma_K, tK, tma_V, tV, mO, softmax_scale * LOG2_E, out_scale, q_offset, mCounts, mLists,
                    list_stride, sQ_layout, sK_layout, sVt_layout, tiled_mma, q_bytes, k_bytes, v_bytes,
                    SharedStorage).launch(grid=grid_dim, block=[self._cta_threads, 1, 1], stream=stream,
                                          min_blocks_per_mp=1)   # pins the launch register cap (65536/384 -> 168) so setmaxnreg can re-balance

    @cute.kernel
    def kernel(self, tma_Q: cute.CopyAtom, mQ: cute.Tensor, tma_K: cute.CopyAtom, mK: cute.Tensor,
               tma_V: cute.CopyAtom, mVt: cute.Tensor, mO: cute.Tensor,
               softmax_scale_log2: cutlass.Float32, out_scale: cutlass.Float32, q_offset: cutlass.Int32,
               mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sVt_layout: cute.ComposedLayout,
               tiled_mma: cute.TiledMma, q_bytes: cutlass.Constexpr, k_bytes: cutlass.Constexpr,
               v_bytes: cutlass.Constexpr, SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, num_head, _ = cute.arch.block_idx()
        kv_head = num_head // self._qhead_per_kvhead
        seqlen_q, seqlen_k = mQ.shape[0], mK.shape[0]
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
        sVt = storage.sVt.get_tensor(sVt_layout)
        mbar_q = storage.mbar_q.data_ptr()
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_q, 1)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._mma_warps)
        pipe_k = pipeline.PipelineTmaAsync.create(num_stages=STAGES, producer_group=producer_group,
                                                  consumer_group=consumer_group, tx_count=k_bytes,
                                                  barrier_storage=storage.mbar_k.data_ptr(), defer_sync=True)
        pipe_v = pipeline.PipelineTmaAsync.create(num_stages=STAGES, producer_group=producer_group,
                                                  consumer_group=consumer_group, tx_count=v_bytes,
                                                  barrier_storage=storage.mbar_v.data_ptr(), defer_sync=True)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # TMA partitions (coordinate tensors): Q (m, d, m_blocks, 1, H); K (n, d, n_blocks, 1, Hk); V^T (d, n, 1, n_blocks, Hk)
        gQ = cute.local_tile(mQ, (self._m_block_size, self._head_dim_padded), (None, None, None))
        gK = cute.local_tile(mK, (self._n_block_size, self._head_dim_padded), (None, None, None))
        gV = cute.local_tile(mVt, (self._head_dim_padded, self._n_block_size), (None, None, None))
        one = cute.make_layout(1)
        tQsQ, tQgQ = cpasync.tma_partition(tma_Q, 0, one, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cpasync.tma_partition(tma_K, 0, one, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cpasync.tma_partition(tma_V, 0, one, cute.group_modes(sVt, 0, 2), cute.group_modes(gV, 0, 2))

        if warp_idx >= self._mma_warps:
            # ---------------- producer warp group ----------------
            cute.arch.setmaxregister_decrease(PRODUCER_REGS)
            if warp_idx == self._mma_warps:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_q, q_bytes)
                cute.copy(tma_Q, tQgQ[(None, m_block, 0, num_head)], tQsQ[(None, 0)], tma_bar_ptr=mbar_q)
                pk = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                pv = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                for n_tile in range(list_count):
                    n_block = n_block_max - n_tile - 1
                    if cutlass.const_expr(self._block_sparse):
                        n_block = mLists[list_begin + list_count - 1 - n_tile]
                    pipe_k.producer_acquire(pk)
                    cute.copy(tma_K, tKgK[(None, n_block, 0, kv_head)], tKsK[(None, pk.index)],
                              tma_bar_ptr=pipe_k.producer_get_barrier(pk))
                    pipe_k.producer_commit(pk)
                    pk.advance()
                    pipe_v.producer_acquire(pv)
                    cute.copy(tma_V, tVgV[(None, 0, n_block, kv_head)], tVsV[(None, pv.index)],
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
            copy_sQ = cute.make_tiled_copy_A(ldsm, tiled_mma)
            copy_sK = cute.make_tiled_copy_B(ldsm, tiled_mma)
            copy_sVt = cute.make_tiled_copy_B(ldsm, tiled_mma)
            tSsQ, tSrQ_view = copy_sQ.get_slice(tidx).partition_S(sQ[None, None, 0]), copy_sQ.get_slice(tidx).retile(tSrQ)
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
                                softmax_scale_log2=softmax_scale_log2, tidx=tidx, sK=sK, sVt=sVt, half_d=half_d,
                                pipe_k=pipe_k, pipe_v=pipe_v,
                                tiled_mma=tiled_mma, thr_mma=thr_mma, copy_sQ=copy_sQ, copy_sK=copy_sK, copy_sVt=copy_sVt,
                                tSsQ=tSsQ, tSrQ_view=tSrQ_view, tSrQ=tSrQ, tOrP=tOrP, mcS=mcS,
                                acc_S=acc_S, acc_O0=acc_O0, acc_O1=acc_O1, row_max=row_max, row_sum=row_sum)
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size) + 1
            # pipeline states are explicit loop-carried values (mutations inside P would not be tracked)
            if cutlass.const_expr(self._pingpong):
                group = warp_idx // 4                      # warp-uniform: 0 for warps 0-3, 1 for warps 4-7
                my_bar = 2 + group
                other_bar = 3 - group
                if group == 1:                             # group 0 issues first
                    cute.arch.barrier_arrive(barrier_id=2, number_of_threads=256, aligned=False)
                if list_count > 0:                         # tile 0: QK^T only (no previous PV)
                    cute.arch.barrier(barrier_id=my_bar, number_of_threads=256)
                    ck = self._qk_tile(P, 0, acc_S, ck)
                    cute.arch.barrier_arrive(barrier_id=other_bar, number_of_threads=256, aligned=False)
                    self._softmax_tile(P, 0, True)
                for n_tile in cutlass.range_constexpr(mask_steps):
                    if cutlass.const_expr(n_tile >= 1):
                        if n_tile < list_count:
                            cute.arch.barrier(barrier_id=my_bar, number_of_threads=256)
                            ck = self._qk_tile(P, n_tile, acc_S, ck)
                            cv = self._pv_tile(P, cv)
                            cute.arch.barrier_arrive(barrier_id=other_bar, number_of_threads=256, aligned=False)
                            self._softmax_tile(P, n_tile, True)
                for n_tile in range(mask_steps, list_count, 1):
                    cute.arch.barrier(barrier_id=my_bar, number_of_threads=256)
                    ck = self._qk_tile(P, n_tile, acc_S, ck)
                    cv = self._pv_tile(P, cv)
                    cute.arch.barrier_arrive(barrier_id=other_bar, number_of_threads=256, aligned=False)
                    self._softmax_tile(P, n_tile, False)
                if list_count > 0:                         # trailing PV(last)
                    cute.arch.barrier(barrier_id=my_bar, number_of_threads=256)
                    cv = self._pv_tile(P, cv)
                    cute.arch.barrier_arrive(barrier_id=other_bar, number_of_threads=256, aligned=False)
            elif cutlass.const_expr(not self._s_pipeline):
                for n_tile in cutlass.range_constexpr(mask_steps):
                    if n_tile < list_count:
                        ck = self._qk_tile(P, n_tile, acc_S, ck)
                        self._softmax_tile(P, n_tile, True)
                        cv = self._pv_tile(P, cv)
                for n_tile in range(mask_steps, list_count, 1):
                    ck = self._qk_tile(P, n_tile, acc_S, ck)
                    self._softmax_tile(P, n_tile, False)
                    cv = self._pv_tile(P, cv)
            else:
                acc_S1 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)), cutlass.Float32)
                S = (acc_S, acc_S1)
                # prologue: S(0)
                if list_count > 0:
                    ck = self._qk_tile(P, 0, acc_S, ck)
                for n_tile in cutlass.range_constexpr(mask_steps):
                    if n_tile < list_count:
                        ck, cv = self._pipe_step(P, n_tile, True, S[n_tile % 2], S[(n_tile + 1) % 2], ck, cv)
                pa, pb = mask_steps % 2, (mask_steps + 1) % 2
                for n_tile in range(mask_steps, list_count - 1, 2):
                    ck, cv = self._pipe_step(P, n_tile, False, S[pa], S[pb], ck, cv)
                    ck, cv = self._pipe_step(P, n_tile + 1, False, S[pb], S[pa], ck, cv)
                if list_count > mask_steps and (list_count - mask_steps) % 2 == 1:
                    ck, cv = self._pipe_step(P, list_count - 1, False, S[pa], S[pb], ck, cv)

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
    def _qk_tile(self, P, n_tile, acc, ck):
        """acc = Q K(n_tile)^T from the K stage ck.index; releases the stage and advances ck."""
        P.pipe_k.consumer_wait(ck)
        sK_s = P.sK[None, None, ck.index]
        tSrK = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sK_s))
        thr_cK = P.copy_sK.get_slice(P.tidx)
        tSsK, tSrK_view = thr_cK.partition_S(sK_s), thr_cK.retile(tSrK)
        acc.fill(0.0)
        nk = cute.size(P.tSsQ.shape[2])
        if cutlass.const_expr(self._prefetch):
            cute.copy(P.copy_sQ, P.tSsQ[None, None, 0], P.tSrQ_view[None, None, 0])
            cute.copy(P.copy_sK, tSsK[None, None, 0], tSrK_view[None, None, 0])
            for k in cutlass.range_constexpr(nk):
                if cutlass.const_expr(k + 1 < nk):
                    cute.copy(P.copy_sQ, P.tSsQ[None, None, k + 1], P.tSrQ_view[None, None, k + 1])
                    cute.copy(P.copy_sK, tSsK[None, None, k + 1], tSrK_view[None, None, k + 1])
                cute.gemm(P.tiled_mma, acc, P.tSrQ[None, None, k], tSrK[None, None, k], acc)
        else:
            for k in cutlass.range_constexpr(nk):
                cute.copy(P.copy_sQ, P.tSsQ[None, None, k], P.tSrQ_view[None, None, k])
                cute.copy(P.copy_sK, tSsK[None, None, k], tSrK_view[None, None, k])
                cute.gemm(P.tiled_mma, acc, P.tSrQ[None, None, k], tSrK[None, None, k], acc)
        P.pipe_k.consumer_release(ck)
        ck.advance()
        return ck

    @cute.jit
    def _softmax_tile(self, P, n_tile, masked: cutlass.Constexpr):
        """Online softmax of P.acc_S (tile n_tile) and conversion into the e4m3 A fragment P.tOrP."""
        n_block = P.n_block_max - n_tile - 1
        if cutlass.const_expr(self._block_sparse):
            n_block = P.mLists[P.list_begin + P.list_count - 1 - n_tile]
        if cutlass.const_expr(masked):
            cS = cute.local_tile(P.mcS[0, None, P.num_head, None], (self._m_block_size, self._n_block_size), (P.m_block, n_block))
            tScS = P.thr_mma.partition_C(cS)
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, tScS, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, True)
        else:
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, None, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, False)
        for kb in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        j = 4 * kb + 2 * h + (e >> 1)
                        P.tOrP[e + 4 * r + 8 * h + 16 * kb] = (P.acc_S[(e & 1) + 2 * r + 4 * j] * P_SCALE).to(FP8)

    @cute.jit
    def _pv_tile(self, P, cv):
        """O += P V from the V^T stage cv.index (P.tOrP holds the current P); releases the stage, advances cv."""
        P.pipe_v.consumer_wait(cv)
        sVt_s = P.sVt[None, None, cv.index]
        sVt0 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (0, 0))
        sVt1 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (1, 0))
        tOrVt0 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt0))
        tOrVt1 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt1))
        thr_cV = P.copy_sVt.get_slice(P.tidx)
        tOsVt0, tOrVt0_view = thr_cV.partition_S(sVt0), thr_cV.retile(tOrVt0)
        tOsVt1, tOrVt1_view = thr_cV.partition_S(sVt1), thr_cV.retile(tOrVt1)
        if cutlass.const_expr(self._prefetch):
            cute.copy(P.copy_sVt, tOsVt0[None, None, 0], tOrVt0_view[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
                cute.copy(P.copy_sVt, tOsVt1[None, None, k], tOrVt1_view[None, None, k])
                cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], tOrVt0[None, None, k], P.acc_O0)
                if cutlass.const_expr(k + 1 < cute.size(P.tOrP.shape[2])):
                    cute.copy(P.copy_sVt, tOsVt0[None, None, k + 1], tOrVt0_view[None, None, k + 1])
                cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], tOrVt1[None, None, k], P.acc_O1)
        else:
            for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
                cute.copy(P.copy_sVt, tOsVt0[None, None, k], tOrVt0_view[None, None, k])
                cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], tOrVt0[None, None, k], P.acc_O0)
                cute.copy(P.copy_sVt, tOsVt1[None, None, k], tOrVt1_view[None, None, k])
                cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], tOrVt1[None, None, k], P.acc_O1)
        P.pipe_v.consumer_release(cv)
        cv.advance()
        return cv

    @cute.jit
    def _pipe_step(self, P, n_tile, masked: cutlass.Constexpr, cur, nxt, ck, cv):
        """S(n+1) -> nxt (if any), then softmax(cur = S(n)), P V(n)."""
        n_block = P.n_block_max - n_tile - 1
        if cutlass.const_expr(self._block_sparse):
            n_block = P.mLists[P.list_begin + P.list_count - 1 - n_tile]
        if n_tile + 1 < P.list_count:
            ck = self._qk_tile(P, n_tile + 1, nxt, ck)
        if cutlass.const_expr(masked):
            cS = cute.local_tile(P.mcS[0, None, P.num_head, None], (self._m_block_size, self._n_block_size), (P.m_block, n_block))
            tScS = P.thr_mma.partition_C(cS)
            self._softmax_step(cur, P.acc_O0, P.acc_O1, tScS, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, True)
        else:
            self._softmax_step(cur, P.acc_O0, P.acc_O1, None, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, False)
        for kb in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        j = 4 * kb + 2 * h + (e >> 1)
                        P.tOrP[e + 4 * r + 8 * h + 16 * kb] = (cur[(e & 1) + 2 * r + 4 * j] * P_SCALE).to(FP8)
        P.pipe_v.consumer_wait(cv)
        sVt_s = P.sVt[None, None, cv.index]
        sVt0 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (0, 0))
        sVt1 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (1, 0))
        tOrVt0 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt0))
        tOrVt1 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt1))
        thr_cV = P.copy_sVt.get_slice(P.tidx)
        tOsVt0, tOrVt0_view = thr_cV.partition_S(sVt0), thr_cV.retile(tOrVt0)
        tOsVt1, tOrVt1_view = thr_cV.partition_S(sVt1), thr_cV.retile(tOrVt1)
        for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            cute.copy(P.copy_sVt, tOsVt0[None, None, k], tOrVt0_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], tOrVt0[None, None, k], P.acc_O0)
            cute.copy(P.copy_sVt, tOsVt1[None, None, k], tOrVt1_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], tOrVt1[None, None, k], P.acc_O1)
        P.pipe_v.consumer_release(cv)
        cv.advance()
        return ck, cv

    @cute.jit
    def _block_step(self, P, n_tile, masked: cutlass.Constexpr, ck, cv):
        n_block = P.n_block_max - n_tile - 1
        if cutlass.const_expr(self._block_sparse):
            n_block = P.mLists[P.list_begin + P.list_count - 1 - n_tile]
        # ---- S = Q K^T from K stage ck.index
        P.pipe_k.consumer_wait(ck)
        sK_s = P.sK[None, None, ck.index]
        tSrK = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sK_s))
        thr_cK = P.copy_sK.get_slice(P.tidx)
        tSsK, tSrK_view = thr_cK.partition_S(sK_s), thr_cK.retile(tSrK)
        P.acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(P.tSsQ.shape[2])):
            cute.copy(P.copy_sQ, P.tSsQ[None, None, k], P.tSrQ_view[None, None, k])
            cute.copy(P.copy_sK, tSsK[None, None, k], tSrK_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_S, P.tSrQ[None, None, k], tSrK[None, None, k], P.acc_S)
        P.pipe_k.consumer_release(ck)
        ck.advance()
        # ---- online softmax, P -> e4m3 A fragment (registers)
        if cutlass.const_expr(masked):
            cS = cute.local_tile(P.mcS[0, None, P.num_head, None], (self._m_block_size, self._n_block_size), (P.m_block, n_block))
            tScS = P.thr_mma.partition_C(cS)
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, tScS, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, True)
        else:
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, None, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.seqlen_k, False)
        for kb in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        j = 4 * kb + 2 * h + (e >> 1)
                        P.tOrP[e + 4 * r + 8 * h + 16 * kb] = (P.acc_S[(e & 1) + 2 * r + 4 * j] * P_SCALE).to(FP8)
        # ---- O += P V from V^T stage cv.index (two d-halves)
        P.pipe_v.consumer_wait(cv)
        sVt_s = P.sVt[None, None, cv.index]
        sVt0 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (0, 0))
        sVt1 = cute.local_tile(sVt_s, (P.half_d, self._n_block_size), (1, 0))
        tOrVt0 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt0))
        tOrVt1 = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sVt1))
        thr_cV = P.copy_sVt.get_slice(P.tidx)
        tOsVt0, tOrVt0_view = thr_cV.partition_S(sVt0), thr_cV.retile(tOrVt0)
        tOsVt1, tOrVt1_view = thr_cV.partition_S(sVt1), thr_cV.retile(tOrVt1)
        for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            cute.copy(P.copy_sVt, tOsVt0[None, None, k], tOrVt0_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], tOrVt0[None, None, k], P.acc_O0)
            cute.copy(P.copy_sVt, tOsVt1[None, None, k], tOrVt1_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], tOrVt1[None, None, k], P.acc_O1)
        P.pipe_v.consumer_release(cv)
        cv.advance()
        return ck, cv
