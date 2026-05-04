/*
 * TC-GNN C++ wrapper -- shared between reference and solution.
 * Provides pybind11 bindings for SpMM forward/backward.
 * The actual CUDA kernel is in a separate .cu file.
 */

#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <algorithm>

#define MIN(x, y) (((x) < (y))? (x) : (y))

// Forward declaration: implemented in the .cu file (ref_kernel.cu or solution.cu)
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
);

// --- CPU Preprocessing ---
// Builds TC-GNN metadata: edgeToRow, edgeToColumn, blockPartition.
// Called once before training. Not performance-critical.

std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length) {
    int loc = 0, cur = 1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < (int)length) {
        if (array[cur] != array[cur - 1]) {
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;
        }
        cur++;
    }
    return nb2col;
}

void preprocess(torch::Tensor edgeList_tensor,
                torch::Tensor nodePointer_tensor,
                int num_nodes,
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor,
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor) {
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

    // Fill edgeToRow
    for (int nid = 0; nid < num_nodes; nid++) {
        for (int eid = nodePointer[nid]; eid < nodePointer[nid + 1]; eid++)
            edgeToRow[eid] = nid;
    }

    // Process each row window
    for (int iter = 0; iter < num_nodes; iter += blockSize_h) {
        int windowId = iter / blockSize_h;
        int block_start = nodePointer[iter];
        int block_end = nodePointer[MIN(iter + blockSize_h, num_nodes)];
        int num_window_edges = block_end - block_start;

        if (num_window_edges == 0) {
            blockPartition[windowId] = 0;
            continue;
        }

        unsigned *neighbor_window = (unsigned *)malloc(num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        std::sort(neighbor_window, neighbor_window + num_window_edges);

        std::map<unsigned, unsigned> clean_edges2col =
            inplace_deduplication(neighbor_window, num_window_edges);

        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) / blockSize_w;

        for (int e_index = block_start; e_index < block_end; e_index++) {
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }

        free(neighbor_window);
    }
}

// --- Python-facing wrappers ---

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow) {
    CHECK_INPUT(input);
    CHECK_INPUT(nodePointer);
    CHECK_INPUT(edgeList);
    CHECK_INPUT(blockPartition);
    CHECK_INPUT(edgeToColumn);
    CHECK_INPUT(edgeToRow);

    int num_nodes = nodePointer.size(0) - 1;
    int num_edges = edgeList.size(0);
    int embedding_dim = input.size(1);

    return spmm_forward_cuda(nodePointer, edgeList,
                             blockPartition, edgeToColumn, edgeToRow,
                             num_nodes, num_edges, embedding_dim, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &preprocess, "TC-GNN Preprocess (CPU)");
    m.def("forward", &spmm_forward, "TC-GNN SpMM forward (CUDA)");
    m.def("backward", &spmm_forward, "TC-GNN SpMM backward (CUDA)");
}
