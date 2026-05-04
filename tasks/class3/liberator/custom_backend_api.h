#pragma once

#include <stdint.h>

extern "C" {

typedef struct {
    uint32_t to_node;
    uint32_t weight;
} KhWeightedEdge;

int kh_liberator_backend_init(uint64_t memory_budget_bytes);
void kh_liberator_backend_shutdown();
const char* kh_liberator_backend_last_error();

int kh_liberator_bfs(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const uint32_t* col_indices,
    uint32_t source_node,
    uint32_t* host_out_levels);

int kh_liberator_cc(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const uint32_t* col_indices,
    uint32_t* host_out_labels);

int kh_liberator_sssp(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint64_t* row_offsets,
    const KhWeightedEdge* edges,
    uint32_t source_node,
    uint32_t* host_out_distances);

int kh_liberator_pr(
    uint64_t num_vertices,
    uint64_t num_edges,
    const uint32_t* out_degree,
    const uint64_t* col_offsets,
    const uint32_t* row_indices,
    uint32_t max_iterations,
    double damping,
    double tolerance,
    double* host_out_ranks);

}
