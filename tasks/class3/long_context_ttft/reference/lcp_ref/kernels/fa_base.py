# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from types import SimpleNamespace
from typing import Type, Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import cutlass.utils as utils

"""Base SM120 FlashAttention forward kernel (CuTe DSL), derived from the CUTLASS Ampere FlashAttention-2 example.

Q [B, Sq, N, H], K/V [B, Sk, N, H]; online softmax, optional causal mask and per-(head, query block) block lists.
"""


class FlashAttentionForwardSm120:
    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
        block_sparse: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self._head_dim = head_dim
        self._qhead_per_kvhead = qhead_per_kvhead
        self._block_sparse = block_sparse
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        # Check if data type is fp16 or bf16
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False

        # Check if head dimension is a multiple of 8
        if head_dim % 8 != 0:
            return False

        # Check if number of threads is a multiple of 32
        if num_threads % 32 != 0:
            return False

        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False

        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        q_offset: cutlass.Int32,
        mCounts: cute.Tensor,
        mLists: cute.Tensor,
        list_stride: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention v2 kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, 1)

        Prepares the shared memory layout, tiled copy atoms, tiled mma and shared memory storage.
        Then launches the kernel function with the prepared parameters.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale: softmax scale
        :type softmax_scale: cutlass.Float32
        """
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid_dim: (m_block, batch_size, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale_log2,
            q_offset,
            mCounts,
            mLists,
            list_stride,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        q_offset: cutlass.Int32,
        mCounts: cute.Tensor,
        mLists: cute.Tensor,
        list_stride: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Kernel function for flash attention v2.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale_log2: softmax scale log2
        :type softmax_scale_log2: cutlass.Float32
        :param sQ_layout: query layout
        :type sQ_layout: cute.ComposedLayout
        :param sKV_layout: key/value layout
        :type sKV_layout: cute.ComposedLayout
        :param sO_layout: output layout
        :type sO_layout: cute.ComposedLayout
        :param gmem_tiled_copy_QKV: tiled copy for QKV load
        :type gmem_tiled_copy_QKV: cute.TiledCopy
        :param gmem_tiled_copy_O: tiled copy for O store
        :type gmem_tiled_copy_O: cute.TiledCopy
        :param tiled_mma: tiled mma
        :type tiled_mma: cute.TiledMma
        :param SharedStorage: shared storage
        :type SharedStorage: cutlass.Constexpr
        """
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div(
                    (m_block + 1) * self._m_block_size + q_offset,
                    self._n_block_size,
                ),
                n_block_max,
            )
        # Block list for this (head, m_block): dense walks n_block_max-1 .. 0; sparse walks the BSR row backwards.
        list_begin = cutlass.Int32(0)
        list_count = n_block_max
        if cutlass.const_expr(self._block_sparse):
            m_blocks = cute.ceil_div(mQ.shape[1], self._m_block_size)
            row = num_head * m_blocks + m_block
            list_begin = row * list_stride
            list_count = mCounts[row]
        n_block = n_block_max - 1
        if cutlass.const_expr(self._block_sparse):
            n_block = mLists[list_begin + list_count - 1]

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        # (m_block_size, head_dim)
        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_size, None, num_head // self._qhead_per_kvhead, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(
            mV[batch_size, None, num_head // self._qhead_per_kvhead, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transpose view of V to tensor with layout (head_dim, n_block_size) for tiled mma
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for Q and KV
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head // self._qhead_per_kvhead, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )

        # Repeat the partitioning with identity layouts
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and do special process for mn.
        # This is to reduce register pressure and gets 2-3% performance gain compared with allocating the whole tile.
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        # Set predicates for head_dim bounds, seqlen_q/k bounds is processed at the first tile.
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block],
                    tKsK[None, n, None],
                    pred=tKVpKV[None, n, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # Softmax intermediate result: row_max and row_sum
        # ///////////////////////////////////////////////////////////////////////////////
        # shape: (atom_v_m * rest_m)
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        # shape: (atom_v_m * rest_m)
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # group parameters for compute_one_n_block
        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            batch_size=batch_size,
            num_head=num_head,
            q_offset=q_offset,
            next_n_block=n_block - 1,
            has_next=n_block > 0,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last ceil_div(m_block_size, n_block_size) blocks.
        # We will have at least 1 "masking" iteration.
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            # q_offset may be unaligned to n_block_size, so the masked band spans one extra key block
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size) + 1

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            if cutlass.const_expr(self._block_sparse):
                n_block = cutlass.Int32(-1)
                if n_tile < list_count:
                    n_block = mLists[list_begin + list_count - 1 - n_tile]
                basic_params.has_next = n_tile + 1 < list_count
                basic_params.next_n_block = cutlass.Int32(0)
                if basic_params.has_next:
                    basic_params.next_n_block = mLists[list_begin + list_count - 2 - n_tile]
            else:
                basic_params.next_n_block = n_block - 1
                basic_params.has_next = n_block > 0
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=(n_tile == 0),
                        in_mask_steps=True,
                    )
            else:
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    is_first_n_block=True,
                    in_mask_steps=True,
                )

        # Start async loads of rest k-tiles in reverse order, no k-residue handling needed
        for n_tile in range(mask_steps, list_count, 1):
            n_block = n_block_max - n_tile - 1
            if cutlass.const_expr(self._block_sparse):
                n_block = mLists[list_begin + list_count - 1 - n_tile]
                basic_params.has_next = n_tile + 1 < list_count
                basic_params.next_n_block = cutlass.Int32(0)
                if basic_params.has_next:
                    basic_params.next_n_block = mLists[list_begin + list_count - 2 - n_tile]
            else:
                basic_params.next_n_block = n_block - 1
                basic_params.has_next = n_block > 0
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                is_first_n_block=False,
                in_mask_steps=False,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # normalize acc_O by row_sum and calculate the lse
        self.normalize_softmax(acc_O, row_sum)
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        # smem copy atom for O
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        # tiled copy atom for O
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        # sync before all smem stores are done.
        self.cta_sync_barrier.arrive_and_wait()
        # load acc O from smem to rmem for wider vectorization
        cute.copy(
            gmem_tiled_copy_O,
            tOsO,
            tOrO,
        )
        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param gmem_copy_params: gmem copy parameters
        :type gmem_copy_params: SimpleNamespace
        :param smem_copy_params: smem copy parameters
        :type smem_copy_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param is_first_n_block: is first n block
        :type is_first_n_block: cutlass.Constexpr
        """
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # wait for smem tile QK before mma calculation for S
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # load smem tile V for O, special process for the first tile to avoid loading nan.
        # The `if` here is a constexpr, won't be generated in the IR.
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.mK.layout.shape[1],
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # S gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # load first QK k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        # mma for S
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            # load next QK k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # wait for smem tile V for O
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.has_next:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.next_n_block],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # online softmax
        # ///////////////////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////////////////
        # O gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # Convert layout of acc_S to gemm O accept layout.
        # Due to the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # load first V k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        # mma for O
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            # load next V k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax and rescale acc_O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        :param in_mask_steps: in mask steps
        :type in_mask_steps: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        # if it is the first tile, create a mask for residual of S to -inf for softmax.
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # Each iteration processes one row of acc_S
        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # mask residual of S with -inf
            if cutlass.const_expr(in_mask_steps):
                if cutlass.const_expr(not self._is_causal):
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        if cute.elem_less(
                            basic_params.mK.shape[1], tScS_mn[0, c][3] + 1
                        ):
                            acc_S_mn[r, c] = -cutlass.Float32.inf
                else:
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    col_idx_limit = cutlass.min(
                        tScS_mn[r, 0][1] + 1 + basic_params.q_offset, basic_params.mK.shape[1]
                    )
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        # only consider the column index, so the row index sets to 0.
                        if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            # (n_block_size)
            acc_S_row = acc_S_mn[r, None].load()
            # row_max_cur_row => f32
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            # quad reduction for row_max
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            # compute exp(x - max) using exp2(x * log_2(e) - max * log_2(e))
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            # acc_S_row_sum => f32
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            # if it is not the first tile, load the row r of previous row_max and minus row_max_cur_row to update row_sum.
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            # update row_max, row_sum and acc_S
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Normalize acc_O by row_sum.

        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # do quad reduction for row_sum.
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """make acc tensor as mn layout view

        :param acc: input tensor
        :type acc: cute.Tensor
        :return: acc tensor mn layout view
        :rtype: cute.Tensor
        """
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),  # MMA_N
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),  # MMA_N
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """thread quad reduction

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: x + y)
