#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ inline void atomicAdd_F(float* address, float value) {
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
}

__global__ void agnn_forward_kernel(
    float* output,
    float* edge_attention,  // [E] output: normalized attention weights
    const float* input,
    const int* row_pointers,
    const int* column_index,
    const int num_nodes,
    const int dim
) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;

    if (warpId < num_nodes) {
        const int nb_begin = row_pointers[warpId];
        const int nb_end = row_pointers[warpId + 1];

        // Compute |h_src|^2
        float src_norm2 = 0.0f;
        for (int d = laneid; d < dim; d += WARP_SIZE) {
            float val = input[warpId * dim + d];
            src_norm2 += val * val;
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            src_norm2 += __shfl_down_sync(FULL_MASK, src_norm2, offset);
        src_norm2 = __shfl_sync(FULL_MASK, src_norm2, 0);  // broadcast
        float src_norm = sqrtf(src_norm2 + 1e-8f);

        // Phase 1: compute exp(cos_sim) for each neighbor, accumulate sum
        float attn_sum = 0.0f;
        for (int nidx = nb_begin; nidx < nb_end; nidx++) {
            int nid = column_index[nidx];

            float dot_prod = 0.0f;
            float dst_norm2 = 0.0f;
            for (int d = laneid; d < dim; d += WARP_SIZE) {
                float src_val = input[warpId * dim + d];
                float dst_val = input[nid * dim + d];
                dot_prod += src_val * dst_val;
                dst_norm2 += dst_val * dst_val;
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                dot_prod += __shfl_down_sync(FULL_MASK, dot_prod, offset);
                dst_norm2 += __shfl_down_sync(FULL_MASK, dst_norm2, offset);
            }

            if (laneid == 0) {
                float dst_norm = sqrtf(dst_norm2 + 1e-8f);
                float cos_sim = dot_prod / (src_norm * dst_norm);
                float attn = expf(cos_sim);
                edge_attention[nidx] = attn;
                attn_sum += attn;
            }
        }
        attn_sum = __shfl_sync(FULL_MASK, attn_sum, 0);  // broadcast

        // Phase 2: normalize and weighted aggregation
        for (int nidx = nb_begin; nidx < nb_end; nidx++) {
            int nid = column_index[nidx];
            float attn = edge_attention[nidx] / (attn_sum + 1e-8f);
            if (laneid == 0) edge_attention[nidx] = attn;  // store normalized
            attn = __shfl_sync(FULL_MASK, attn, 0);  // broadcast normalized attn

            for (int d = laneid; d < dim; d += WARP_SIZE) {
                output[warpId * dim + d] += attn * input[nid * dim + d];
            }
        }
    }
}

__global__ void agnn_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* edge_attention,
    const int* row_pointers,
    const int* column_index,
    const int num_nodes,
    const int dim
) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;

    if (warpId < num_nodes) {
        const int nb_begin = row_pointers[warpId];
        const int nb_end = row_pointers[warpId + 1];

        for (int nidx = nb_begin; nidx < nb_end; nidx++) {
            int nid = column_index[nidx];
            float attn = edge_attention[nidx];
            for (int d = laneid; d < dim; d += WARP_SIZE) {
                atomicAdd(&grad_input[warpId * dim + d], attn * grad_output[nid * dim + d]);
            }
        }
    }
}

static std::vector<torch::Tensor> agnn_core_forward_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    int dim
) {
    auto output = torch::zeros_like(input);
    auto edge_attention = torch::zeros({num_edges}, input.options());

    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_blocks = (num_nodes + warps_per_block - 1) / warps_per_block;

    agnn_forward_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        edge_attention.data_ptr<float>(),
        input.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        column_index.data_ptr<int>(),
        num_nodes, dim);

    return {output, edge_attention};
}

static std::vector<torch::Tensor> agnn_core_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor edge_attention,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    int dim
) {
    auto grad_input = torch::zeros_like(grad_output);

    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_blocks = (num_nodes + warps_per_block - 1) / warps_per_block;

    agnn_backward_kernel<<<num_blocks, threads_per_block>>>(
        grad_input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        edge_attention.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        column_index.data_ptr<int>(),
        num_nodes, dim);

    return {grad_input};
}

std::vector<torch::Tensor> agnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    bool apply_relu
) {
    auto projected = torch::mm(input, weights).contiguous();
    auto result = agnn_core_forward_cuda(
        projected,
        row_pointers,
        column_index,
        num_nodes,
        num_edges,
        projected.size(1));

    auto output = result[0];
    if (apply_relu) {
        output = torch::relu(output);
    }
    return {output, result[1]};
}

std::vector<torch::Tensor> agnn_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor edge_attention,
    torch::Tensor layer_output,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    bool apply_relu
) {
    auto grad = grad_output.contiguous();
    if (apply_relu) {
        auto relu_mask = layer_output.gt(0).to(grad.scalar_type());
        grad = grad * relu_mask;
    }

    auto grad_input_prime = agnn_core_backward_cuda(
        grad,
        edge_attention,
        row_pointers,
        column_index,
        num_nodes,
        num_edges,
        weights.size(1))[0];
    auto grad_input = torch::mm(grad_input_prime, weights.transpose(0, 1));
    auto grad_weights = torch::mm(input.transpose(0, 1), grad_input_prime);

    return {grad_input, grad_weights};
}
