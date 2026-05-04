// Minimal Liberator black-box backend template.
// The benchmark compiles this file into a shared library and calls the symbols
// declared in custom_backend_api.h.

#include <cuda_runtime.h>
#include <cstdio>

#include "custom_backend_api.h"

namespace {

char g_last_error[256] = "backend not initialized";
uint64_t g_memory_budget_bytes = 0;

__global__ void kh_liberator_warmup() {}

void set_last_error(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "unknown error");
}

}  // namespace

extern "C" int kh_liberator_backend_init(uint64_t memory_budget_bytes) {
    g_memory_budget_bytes = memory_budget_bytes;
    kh_liberator_warmup<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return 1;
    }
    set_last_error("ok");
    return 0;
}

extern "C" void kh_liberator_backend_shutdown() {
    g_memory_budget_bytes = 0;
}

extern "C" const char* kh_liberator_backend_last_error() {
    return g_last_error;
}

extern "C" int kh_liberator_bfs(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const uint32_t* col_indices,
    uint32_t source_node,
    uint32_t* host_out_levels) {
    (void)num_vertices;
    (void)num_edges;
    (void)row_offsets;
    (void)col_indices;
    (void)source_node;
    (void)host_out_levels;
    (void)g_memory_budget_bytes;
    set_last_error("kh_liberator_bfs is not implemented");
    return 1;
}

extern "C" int kh_liberator_cc(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const uint32_t* col_indices,
    uint32_t* host_out_labels) {
    (void)num_vertices;
    (void)num_edges;
    (void)row_offsets;
    (void)col_indices;
    (void)host_out_labels;
    (void)g_memory_budget_bytes;
    set_last_error("kh_liberator_cc is not implemented");
    return 1;
}

extern "C" int kh_liberator_sssp(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const KhWeightedEdge* edges,
    uint32_t source_node,
    uint32_t* host_out_distances) {
    (void)num_vertices;
    (void)num_edges;
    (void)row_offsets;
    (void)edges;
    (void)source_node;
    (void)host_out_distances;
    (void)g_memory_budget_bytes;
    set_last_error("kh_liberator_sssp is not implemented");
    return 1;
}

extern "C" int kh_liberator_pr(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint32_t* out_degree,
    const uint64_t* col_offsets,
    const uint32_t* row_indices,
    uint32_t max_iterations,
    double damping,
    double tolerance,
    double* host_out_ranks) {
    (void)num_vertices;
    (void)num_edges;
    (void)out_degree;
    (void)col_offsets;
    (void)row_indices;
    (void)max_iterations;
    (void)damping;
    (void)tolerance;
    (void)host_out_ranks;
    (void)g_memory_budget_bytes;
    set_last_error("kh_liberator_pr is not implemented");
    return 1;
}
