"""SM120 FlashAttention with FP8 (e4m3) operands: Q, K, V^T in float8_e4m3fn, fp32 accumulation.

Structure follows the BF16 v2 kernel (4 warps, 64x64 tiles, single-buffered K/V with the
FA2 load overlap), with:
- mma.sync m16n8k32 e4m3 (`warp.MmaFP8Op`), 2x the BF16 MMA rate; K/V tiles are half the bytes.
- V is consumed as V^T [d, key] with keys contiguous (K-major B operand): the caller supplies
  V^T per KV head; no in-kernel transpose (ldmatrix.trans does not exist for 8-bit elements).
- P (after online softmax, scaled by 448 to use the e4m3 range) is written as e4m3 to a
  CTA-private shared tile and re-read with ldmatrix as the A operand of the PV MMA, which avoids
  the cross-lane re-packing that the fp8 A-fragment layout would otherwise need.
- All quantisation scales are folded into two scalars: softmax_scale * s_q * s_k into the
  softmax log2 scale, and s_v / 448 into the output scale.
Same GQA / causal q_offset / per-(head, m_block) block-list support as v2.
"""
from __future__ import annotations

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline

from .fa_base import FlashAttentionForwardSm120

FP8 = cutlass.Float8E4M3FN
P_SCALE = 448.0


class FlashAttentionFP8Sm120(FlashAttentionForwardSm120):
    def __init__(self, head_dim: int, m_block_size: int = 64, n_block_size: int = 64, num_threads: int = 128,
                 is_causal: bool = True, qhead_per_kvhead: int = 1, block_sparse: bool = False):
        if head_dim != 256 or n_block_size != 64 or num_threads != 2 * m_block_size or m_block_size not in (64, 128):
            raise ValueError('fp8 kernel supports head_dim 256, n_block 64, m_block 64 (128 threads) or 128 (256 threads)')
        super().__init__(head_dim, m_block_size, n_block_size, num_threads, is_causal, qhead_per_kvhead, block_sparse)
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mVt: cute.Tensor, mO: cute.Tensor,
                 softmax_scale: cutlass.Float32, out_scale: cutlass.Float32, q_offset: cutlass.Int32,
                 mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32, stream: cuda.CUstream):
        """mQ [B, Sq, H, D] e4m3; mK [B, Sk, Hk, D] e4m3; mVt [B, Hk, D, Sk] e4m3; mO [B, Sq, H, D] bf16."""
        self._dtype: Type[cutlass.Numeric] = FP8
        self._out_dtype = mO.element_type
        m, n, d = self._m_block_size, self._n_block_size, self._head_dim_padded
        row_major = cutlass.utils.LayoutEnum.ROW_MAJOR
        sQ_layout = cute.slice_(sm90_utils.make_smem_layout_a(row_major, (m, n, d), FP8, 1), (None, None, 0))
        sK_layout = cute.slice_(sm90_utils.make_smem_layout_b(row_major, (m, n, d), FP8, 1), (None, None, 0))
        sVt_layout = cute.slice_(sm90_utils.make_smem_layout_b(row_major, (m, d, n), FP8, 1), (None, None, 0))
        sP_layout = cute.slice_(sm90_utils.make_smem_layout_a(row_major, (m, d, n), FP8, 1), (None, None, 0))

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sQ_layout)], 1024]
            sK0: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sK_layout)], 1024]
            sK1: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sK_layout)], 1024]
            sVt: cute.struct.Align[cute.struct.MemRange[FP8, cute.cosize(sVt_layout)], 1024]

        copy_bits = 128
        elems = copy_bits // 8
        atom_async = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), FP8,
                                         num_bits_per_copy=copy_bits)
        # Q/K rows have 256 elements (16 copies of 16 B); V^T rows have 64 keys (4 copies).
        copy_QK = cute.make_tiled_copy_tv(atom_async, cute.make_layout((self._num_threads // 16, 16), stride=(16, 1)),
                                          cute.make_layout((1, elems)))
        copy_Vt = cute.make_tiled_copy_tv(atom_async, cute.make_layout((self._num_threads // 4, 4), stride=(4, 1)),
                                          cute.make_layout((1, elems)))
        tiled_mma = cute.make_tiled_mma(warp.MmaFP8Op(FP8, cutlass.Float32, (16, 8, 32)),
                                        (self._num_threads // 32, 1, 1),
                                        permutation_mnk=(self._num_threads // 32 * 16, 16, 32))
        grid_dim = (cute.ceil_div(mQ.shape[1], m), cute.size(mQ.shape[0]), cute.size(mQ.shape[2]))
        LOG2_E = 1.4426950408889634074
        self.kernel(mQ, mK, mVt, mO, softmax_scale * LOG2_E, out_scale, q_offset, mCounts, mLists, list_stride,
                    sQ_layout, sK_layout, sVt_layout, sP_layout, copy_QK, copy_Vt, tiled_mma, SharedStorage).launch(
            grid=grid_dim, block=[self._num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mVt: cute.Tensor, mO: cute.Tensor,
               softmax_scale_log2: cutlass.Float32, out_scale: cutlass.Float32, q_offset: cutlass.Int32,
               mCounts: cute.Tensor, mLists: cute.Tensor, list_stride: cutlass.Int32,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sVt_layout: cute.ComposedLayout,
               sP_layout: cute.ComposedLayout, copy_QK: cute.TiledCopy, copy_Vt: cute.TiledCopy,
               tiled_mma: cute.TiledMma, SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, num_head = cute.arch.block_idx()
        kv_head = num_head // self._qhead_per_kvhead
        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if cutlass.const_expr(self._is_causal):
            n_block_max = min(cute.ceil_div((m_block + 1) * self._m_block_size + q_offset, self._n_block_size), n_block_max)
        list_begin = cutlass.Int32(0)
        list_count = n_block_max
        if cutlass.const_expr(self._block_sparse):
            m_blocks = cute.ceil_div(mQ.shape[1], self._m_block_size)
            row = num_head * m_blocks + m_block
            list_begin = row * list_stride
            list_count = mCounts[row]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK0.get_tensor(sK_layout)          # buffer 0; buffer 1 follows at +k_elems
        k_elems = cute.cosize(sK_layout)
        sVt = storage.sVt.get_tensor(sVt_layout)

        gQ = cute.local_tile(mQ[batch, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gK = cute.local_tile(mK[batch, None, kv_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        gVt = cute.local_tile(mVt[batch, kv_head, None, None], (self._head_dim_padded, self._n_block_size), (0, None))
        thr_QK = copy_QK.get_slice(tidx)
        thr_Vt = copy_Vt.get_slice(tidx)
        tQgQ, tQsQ = thr_QK.partition_S(gQ), thr_QK.partition_D(sQ)
        tKgK, tKsK = thr_QK.partition_S(gK), thr_QK.partition_D(sK)
        tVgVt, tVsVt = thr_Vt.partition_S(gVt), thr_Vt.partition_D(sVt)
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcK = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(mcQ[batch, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        cK = cute.local_tile(mcK[batch, None, kv_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        tQcQ = thr_QK.partition_S(cQ)
        tKcK = thr_QK.partition_S(cK)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        half_d = self._head_dim_padded // 2
        sVt0 = cute.local_tile(sVt, (half_d, self._n_block_size), (0, 0))
        sVt1 = cute.local_tile(sVt, (half_d, self._n_block_size), (1, 0))
        tOrVt0 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        tOrVt1 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt1))
        # P lives in registers: the S accumulator is converted straight into the e4m3 A fragment (see _block_step)
        tOrP = cute.make_rmem_tensor(thr_mma.partition_shape_A((self._m_block_size, self._n_block_size)), FP8)
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)), cutlass.Float32)
        acc_O0 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, half_d)), cutlass.Float32)
        acc_O1 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, half_d)), cutlass.Float32)
        acc_O0.fill(0.0)
        acc_O1.fill(0.0)
        ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), FP8)
        copy_sQ = cute.make_tiled_copy_A(ldsm, tiled_mma)
        copy_sK = cute.make_tiled_copy_B(ldsm, tiled_mma)
        copy_sVt = cute.make_tiled_copy_B(ldsm, tiled_mma)
        tSsQ, tSrQ_view = copy_sQ.get_slice(tidx).partition_S(sQ), copy_sQ.get_slice(tidx).retile(tSrQ)
        tSsK, tSrK_view = copy_sK.get_slice(tidx).partition_S(sK), copy_sK.get_slice(tidx).retile(tSrK)
        tOsVt0, tOrVt0_view = copy_sVt.get_slice(tidx).partition_S(sVt0), copy_sVt.get_slice(tidx).retile(tOrVt0)
        tOsVt1, tOrVt1_view = copy_sVt.get_slice(tidx).partition_S(sVt1), copy_sVt.get_slice(tidx).retile(tOrVt1)
        mcS = cute.make_identity_tensor((mQ.shape[0], mQ.shape[1], mQ.shape[2], mK.shape[1]))

        row_max = cute.make_rmem_tensor((acc_O0.shape[0][0] * acc_O0.shape[1]), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((acc_O0.shape[0][0] * acc_O0.shape[1]), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # prologue: Q and first K
        for mm in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, mm, 0][1], mQ.layout.shape[1]):
                cute.copy(copy_QK, tQgQ[None, mm, None], tQsQ[None, mm, None])
            else:
                tQsQ[None, mm, None].fill(0)
        first_n = n_block_max - 1
        if cutlass.const_expr(self._block_sparse):
            first_n = mLists[list_begin + list_count - 1]
        for nn in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKcK[0, nn, 0, first_n][1], mK.layout.shape[1]):
                cute.copy(copy_QK, tKgK[None, nn, None, first_n], tKsK[None, nn, None])
            else:
                tKsK[None, nn, None].fill(0)
        cute.arch.cp_async_commit_group()

        from types import SimpleNamespace
        P = SimpleNamespace(mQ=mQ, mK=mK, mVt=mVt, batch=batch, num_head=num_head, m_block=m_block,
                            n_block_max=n_block_max, list_begin=list_begin, list_count=list_count,
                            mLists=mLists, q_offset=q_offset, softmax_scale_log2=softmax_scale_log2,
                            copy_QK=copy_QK, copy_Vt=copy_Vt, tKgK=tKgK, tKsK=tKsK, tKcK=tKcK, tVgVt=tVgVt, tVsVt=tVsVt,
                            sK=sK, sK_layout=sK_layout, k_elems=k_elems, thr_QK=thr_QK, tidx=tidx,
                            tiled_mma=tiled_mma, thr_mma=thr_mma, copy_sQ=copy_sQ, copy_sK=copy_sK, copy_sVt=copy_sVt,
                            tSsQ=tSsQ, tSrQ_view=tSrQ_view, tSrQ=tSrQ, tSsK=tSsK, tSrK_view=tSrK_view,
                            tSrK=tSrK, tOsVt0=tOsVt0, tOrVt0_view=tOrVt0_view, tOrVt0=tOrVt0, tOsVt1=tOsVt1,
                            tOrVt1_view=tOrVt1_view, tOrVt1=tOrVt1, tOrP=tOrP,
                            mcS=mcS, acc_S=acc_S, acc_O0=acc_O0, acc_O1=acc_O1,
                            row_max=row_max, row_sum=row_sum)
        mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size) + 1
        for n_tile in cutlass.range_constexpr(mask_steps):
            if n_tile < list_count:
                self._block_step(P, n_tile, True)
        for n_tile in range(mask_steps, list_count, 1):
            self._block_step(P, n_tile, False)

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
            row = m_block * self._m_block_size + coord[0]
            if row < mQ.shape[1]:
                mO[batch, row, num_head, coord[1]] = acc_O0[i].to(self._out_dtype)
                mO[batch, row, num_head, half_d + coord[1]] = acc_O1[i].to(self._out_dtype)

    @cute.jit
    def _block_step(self, P, n_tile, masked: cutlass.Constexpr):
        n_block = P.n_block_max - n_tile - 1
        has_next = n_block > 0
        next_n_block = n_block - 1
        if cutlass.const_expr(self._block_sparse):
            n_block = P.mLists[P.list_begin + P.list_count - 1 - n_tile]
            has_next = n_tile + 1 < P.list_count
            next_n_block = cutlass.Int32(0)
            if has_next:
                next_n_block = P.mLists[P.list_begin + P.list_count - 2 - n_tile]
        # K(n) (and Q) ready; everyone finished PV(n-1): start V^T(n), then prefetch K(next) into the other buffer
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(P.copy_Vt, P.tVgVt[None, None, None, n_block], P.tVsVt)
        cute.arch.cp_async_commit_group()
        kb = n_tile % 2
        sK_cur = cute.make_tensor((P.sK.iterator + kb * P.k_elems).align(min_align=1024), P.sK_layout)
        sK_nxt = cute.make_tensor((P.sK.iterator + (1 - kb) * P.k_elems).align(min_align=1024), P.sK_layout)
        tKsK_n = P.thr_QK.partition_D(sK_nxt)
        if has_next:
            for nn in cutlass.range_constexpr(cute.size(tKsK_n.shape[1])):
                if cute.elem_less(P.tKcK[0, nn, 0, next_n_block][1], P.mK.layout.shape[1]):
                    cute.copy(P.copy_QK, P.tKgK[None, nn, None, next_n_block], tKsK_n[None, nn, None])
                else:
                    tKsK_n[None, nn, None].fill(0)
        cute.arch.cp_async_commit_group()                      # possibly empty group: keeps the group count static
        tSrK = P.thr_mma.make_fragment_B(P.thr_mma.partition_B(sK_cur))
        thr_cK = P.copy_sK.get_slice(P.tidx)
        tSsK, tSrK_view = thr_cK.partition_S(sK_cur), thr_cK.retile(tSrK)
        P.acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(P.tSsQ.shape[2])):
            cute.copy(P.copy_sQ, P.tSsQ[None, None, k], P.tSrQ_view[None, None, k])
            cute.copy(P.copy_sK, tSsK[None, None, k], tSrK_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_S, P.tSrQ[None, None, k], tSrK[None, None, k], P.acc_S)
        if cutlass.const_expr(masked):
            cS = cute.local_tile(P.mcS[P.batch, None, P.num_head, None], (self._m_block_size, self._n_block_size), (P.m_block, n_block))
            tScS = P.thr_mma.partition_C(cS)
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, tScS, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.mK.shape[1], True)
        else:
            self._softmax_step(P.acc_S, P.acc_O0, P.acc_O1, None, P.row_max, P.row_sum, P.softmax_scale_log2, P.q_offset, P.mK.shape[1], False)
        for kb in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        j = 4 * kb + 2 * h + (e >> 1)
                        P.tOrP[e + 4 * r + 8 * h + 16 * kb] = (P.acc_S[(e & 1) + 2 * r + 4 * j] * P_SCALE).to(FP8)
        # V^T(n) landed (the K(next) group may still be in flight); all warps are past their K(n)/P reads
        cute.arch.cp_async_wait_group(1)
        self.cta_sync_barrier.arrive_and_wait()
        for k in cutlass.range_constexpr(cute.size(P.tOrP.shape[2])):
            cute.copy(P.copy_sVt, P.tOsVt0[None, None, k], P.tOrVt0_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O0, P.tOrP[None, None, k], P.tOrVt0[None, None, k], P.acc_O0)
            cute.copy(P.copy_sVt, P.tOsVt1[None, None, k], P.tOrVt1_view[None, None, k])
            cute.gemm(P.tiled_mma, P.acc_O1, P.tOrP[None, None, k], P.tOrVt1[None, None, k], P.acc_O1)

    @cute.jit
    def _softmax_step(self, acc_S, acc_O0, acc_O1, tScS, row_max, row_sum, scale_log2, q_offset, seqlen_k, masked: cutlass.Constexpr):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O0_mn = self._make_acc_tensor_mn_view(acc_O0)
        acc_O1_mn = self._make_acc_tensor_mn_view(acc_O1)
        tScS_mn = None
        if cutlass.const_expr(masked):
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            if cutlass.const_expr(masked):
                col_limit = cutlass.min(tScS_mn[r, 0][1] + 1 + q_offset, seqlen_k)
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(col_limit, tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            row_max_cur = self._threadquad_reduce_max(row_max_cur)
            row_max_prev = row_max[r]
            row_max_cur = cute.arch.fmax(row_max_prev, row_max_cur)
            row_max_safe = 0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur
            acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - row_max_safe * scale_log2, fastmath=True)
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            prev_minus_cur = cute.math.exp2(row_max_prev * scale_log2 - row_max_safe * scale_log2, fastmath=True)
            prev_minus_cur = 0.0 if row_max_prev == -cutlass.Float32.inf else prev_minus_cur
            acc_S_row_sum = acc_S_row_sum + row_sum[r] * prev_minus_cur
            acc_O0_mn[r, None] = acc_O0_mn[r, None].load() * prev_minus_cur
            acc_O1_mn[r, None] = acc_O1_mn[r, None].load() * prev_minus_cur
            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp




# A-fragment slot s of mma.m16n8k32 (e4m3) holds key 16h + 8(e>>1) + 2t + (e&1) for s = 16h + 4t + e; V^T keys are stored
# in this order inside every 32-key block so P's A fragment can be built from the S accumulator in registers.
VT_KEY_PERM = [16 * (s >> 4) + 8 * ((s & 3) >> 1) + 2 * ((s >> 2) & 3) + ((s & 3) & 1) for s in range(32)]
