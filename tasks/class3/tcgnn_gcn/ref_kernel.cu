/*
 * TC-GNN Reference SpMM Kernel
 * WMMA 16x16x8 TF32 Tensor Core accelerated SpMM (ATC'23)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <vector>

#define BLK_H 16
#define BLK_W 8
#define WPB 8

using namespace nvcuda;

__global__ void spmm_forward_cuda_kernel(
    const int * __restrict__ nodePointer,
    const int *__restrict__ edgeList,
    const int *__restrict__ blockPartition,
    const int *__restrict__ edgeToColumn,
    const int *__restrict__ edgeToRow,
    const int numNodes,
    const int numEdges,
    const int embedding_dim,
    const float *__restrict__ input,
    float *output
) {
    const unsigned bid = blockIdx.x;
    const unsigned wid = threadIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned tid = threadIdx.y * blockDim.x + laneid;
    const unsigned warpSize = blockDim.x;
    const unsigned threadPerBlock = blockDim.x * blockDim.y;

    const unsigned dimTileNum = embedding_dim / BLK_H;
    const unsigned nIdx_start = bid * BLK_H;
    const unsigned nIdx_end = min((bid + 1) * BLK_H, (unsigned)numNodes);

    const unsigned eIdx_start = nodePointer[nIdx_start];
    const unsigned eIdx_end = nodePointer[nIdx_end];
    const unsigned num_TC_blocks = blockPartition[bid];
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W];
    __shared__ int sparse_AToX_index[BLK_W];
    extern __shared__ float dense_X[];

    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (unsigned i = 0; i < num_TC_blocks; i++) {

        if (tid < BLK_W) {
            sparse_AToX_index[tid] = numNodes + 1;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

        #pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            if (i * BLK_W <= col && col < (i + 1) * BLK_W) {
                unsigned row_local = edgeToRow[eIdx] % BLK_H;
                unsigned col_local = col % BLK_W;
                sparse_A[row_local * BLK_W + col_local] = 1;
                sparse_AToX_index[col_local] = edgeList[eIdx];
            }
        }
        __syncthreads();

        if (wid < dimTileNum)
            #pragma unroll
            for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
                unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];
                unsigned dense_dimIdx = idx / BLK_W;
                unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
                unsigned target_idx = wid * BLK_W * BLK_H + idx;
                if (source_idx >= dense_bound)
                    dense_X[target_idx] = 0;
                else
                    dense_X[target_idx] = input[source_idx];
            }
        __syncthreads();

        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
            wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (wid < dimTileNum)
        wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H,
                                acc_frag, embedding_dim, wmma::mem_row_major);
}

std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor input
) {
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    const int WARPperBlock = WPB;

    dim3 grid(num_row_windows, 1, 1);
    dim3 block(32, WARPperBlock, 1);

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float);

    spmm_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
        nodePointer.data_ptr<int>(),
        edgeList.data_ptr<int>(),
        blockPartition.data_ptr<int>(),
        edgeToColumn.data_ptr<int>(),
        edgeToRow.data_ptr<int>(),
        num_nodes, num_edges, embedding_dim,
        input.data_ptr<float>(),
        output.data_ptr<float>()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {output};
}
