/*
 * Naive SpMM solution (for testing the harness).
 * LLM replaces this file with optimized CUDA kernels.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void naive_spmm_kernel(
    const int* __restrict__ nodePointer,
    const int* __restrict__ edgeList,
    int numNodes,
    int embedding_dim,
    const float* __restrict__ input,
    float* output)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int dim  = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= numNodes || dim >= embedding_dim) return;

    float sum = 0.0f;
    for (int e = nodePointer[node]; e < nodePointer[node + 1]; e++) {
        int neighbor = edgeList[e];
        sum += input[neighbor * embedding_dim + dim];
    }
    output[node * embedding_dim + dim] = sum;
}

std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,   // unused by naive
    torch::Tensor edgeToColumn,     // unused by naive
    torch::Tensor edgeToRow,        // unused by naive
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor input)
{
    auto output = torch::zeros_like(input);

    dim3 block(16, 16);
    dim3 grid((num_nodes + 15) / 16, (embedding_dim + 15) / 16);

    naive_spmm_kernel<<<grid, block>>>(
        nodePointer.data_ptr<int>(),
        edgeList.data_ptr<int>(),
        num_nodes, embedding_dim,
        input.data_ptr<float>(),
        output.data_ptr<float>());

    return {output};
}
