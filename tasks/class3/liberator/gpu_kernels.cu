#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <limits>
#include <algorithm>

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_ITERATIONS = 1000;
constexpr float FLOAT_EPSILON = 1e-6f;
constexpr int INF_INT = 0x7FFFFFFF;
constexpr float INF_FLOAT = 1e30f;

// Graph structure (CSR format for memory efficiency)
struct GraphCSR {
    int* d_csr_row_offsets;
    int* d_csr_col_indices;
    int num_nodes;
    int num_edges;
};

// BFS structures
struct BFSFrontier {
    int* d_frontier;
    int* d_next_frontier;
    int* d_visited;
    int current_size;
    int max_size;
};

// CC structures
struct CCLabels {
    int* d_labels;
    int* d_next_labels;
    int convergence_flag;
    int max_iterations;
};

// SSSP structures
struct SSSPState {
    float* d_distances;
    int* d_queue;
    int queue_size;
    int max_queue_size;
    float infinity;
};

// PageRank structures
struct PageRankData {
    float* d_ranks;
    float* d_next_ranks;
    float* d_out_degrees;
    float dangle_sum;
    float damping;
    int convergence_flag;
    int max_iterations;
};

// ==================== BFS Kernel ====================
__global__ void BFSKernel(int* d_csr_row_offsets, int* d_csr_col_indices,
                          int* d_parent, int* d_frontier, int* d_next_frontier,
                          int frontier_size, int max_frontier, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes one frontier node
    if (tid >= frontier_size) return;
    
    int current_node = d_frontier[tid];
    
    // Skip if already visited (parent != -1 means visited)
    if (d_parent[current_node] != -1) return;
    
    // Mark current node as visited
    if (d_parent[current_node] == -1) {
        d_parent[current_node] = tid;
    }
    
    // Process edges
    int start = d_csr_row_offsets[current_node];
    int end = d_csr_row_offsets[current_node + 1];
    
    // Each thread processes a subset of edges
    int edge_idx = start + tid;
    while (edge_idx < end) {
        int neighbor = d_csr_col_indices[edge_idx];
        
        // Try to mark neighbor as visited atomically
        int expected = -1;
        int old_val = atomicCAS(&d_parent[neighbor], expected, tid);
        
        // If successfully marked, add to next frontier
        if (old_val == -1) {
            // Use modulo to avoid overflow
            int next_idx = tid * 2 + threadIdx.x % 2;
            if (next_idx < max_frontier) {
                d_next_frontier[next_idx] = neighbor;
            }
        }
        
        edge_idx += blockDim.x;
    }
}

// BFS frontier size kernel
__global__ void BFSFrontierSizeKernel(int* d_next_frontier, int max_frontier,
                                      int* d_next_frontier_size,
                                      int* d_work_list) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= max_frontier) return;
    
    if (d_next_frontier[tid] != -1) {
        atomicAdd(d_next_frontier_size, 1);
    }
}

// ==================== CC Kernel ====================
__global__ void CCKernel(int* d_csr_row_offsets, int* d_csr_col_indices,
                         int* d_labels, int* d_next_labels, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    int current_label = d_labels[tid];
    
    // Find minimum label among neighbors
    int min_label = current_label;
    
    int start = d_csr_row_offsets[tid];
    int end = d_csr_row_offsets[tid + 1];
    
    // Process edges
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int neighbor = d_csr_col_indices[i];
        int neighbor_label = d_labels[neighbor];
        
        if (neighbor_label < min_label) {
            min_label = neighbor_label;
        }
    }
    
    // Reduce within block
    __shared__ int shared_min[256];
    shared_min[threadIdx.x] = min_label;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_min[threadIdx.x] = min(shared_min[threadIdx.x], shared_min[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    min_label = shared_min[0];
    __syncthreads();
    
    // Update label if smaller
    if (min_label < current_label) {
        d_next_labels[tid] = min_label;
    } else {
        d_next_labels[tid] = current_label;
    }
}

// ==================== SSSP Kernel ====================
__global__ void SSSPKernel(int* d_csr_row_offsets, int* d_csr_col_indices,
                           float* d_distances, float* d_next_distances,
                           int num_nodes, float infinity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    // Copy current distance to next
    d_next_distances[tid] = d_distances[tid];
    
    if (d_distances[tid] >= infinity) return;
    
    int start = d_csr_row_offsets[tid];
    int end = d_csr_row_offsets[tid + 1];
    
    // Find minimum distance among neighbors
    float min_dist = d_distances[tid];
    
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int neighbor = d_csr_col_indices[i];
        float edge_weight = 1.0f; // Assume unweighted for Friendster
        
        // Relax edge
        if (d_distances[neighbor] + edge_weight < min_dist) {
            min_dist = d_distances[neighbor] + edge_weight;
        }
    }
    
    // Update next distance
    if (min_dist < d_next_distances[tid]) {
        d_next_distances[tid] = min_dist;
    }
}

// ==================== PageRank Kernel ====================
__global__ void PageRankKernel(int* d_csr_row_offsets, int* d_csr_col_indices,
                               float* d_ranks, float* d_next_ranks,
                               float* d_out_degrees, float damping,
                               int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    // Compute contribution from neighbors
    float contribution = 0.0f;
    
    int start = d_csr_row_offsets[tid];
    int end = d_csr_row_offsets[tid + 1];
    float out_degree = (end - start) > 0 ? (end - start) : 1.0f;
    
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int neighbor = d_csr_col_indices[i];
        contribution += d_ranks[neighbor] / out_degree;
    }
    
    // Reduce within block
    __shared__ float shared_contrib[256];
    shared_contrib[threadIdx.x] = contribution;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_contrib[threadIdx.x] += shared_contrib[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    contribution = shared_contrib[0];
    __syncthreads();
    
    // PageRank formula: damping * sum(contributions) + (1-damping)/N
    d_next_ranks[tid] = damping * contribution + (1.0f - damping) / num_nodes;
}

// ==================== BFS Host Function ====================
extern "C" void LaunchBFS(GraphCSR* graph, BFSFrontier* frontier, 
                          int source_node, int max_memory_gb) {
    // Initialize visited array
    cudaMemset(frontier->d_parent, -1, graph->num_nodes * sizeof(int));
    frontier->d_parent[source_node] = -2; // Mark source as visited (special value)
    
    // Initialize frontier with source
    frontier->d_frontier[0] = source_node;
    frontier->current_size = 1;
    
    int iters = 0;
    int total_iters = MAX_ITERATIONS;
    
    while (frontier->current_size > 0 && iters < total_iters) {
        // Clear next frontier
        cudaMemset(frontier->d_next_frontier, 0, frontier->max_size * sizeof(int));
        for (int i = 0; i < frontier->max_size; i++) {
            frontier->d_next_frontier[i] = -1;
        }
        
        // Launch BFS kernel
        int num_blocks = (frontier->current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        BFSKernel<<<num_blocks, BLOCK_SIZE>>>(
            graph->d_csr_row_offsets,
            graph->d_csr_col_indices,
            frontier->d_parent,
            frontier->d_frontier,
            frontier->d_next_frontier,
            frontier->current_size,
            frontier->max_size,
            graph->num_nodes
        );
        
        cudaDeviceSynchronize();
        
        // Count next frontier size
        int next_size = 0;
        for (int i = 0; i < frontier->max_size; i++) {
            // Check on CPU for simplicity with limited memory
            if (frontier->d_next_frontier[i] != -1) {
                next_size++;
            }
        }
        
        // Swap frontiers
        int* temp = frontier->d_frontier;
        frontier->d_frontier = frontier->d_next_frontier;
        frontier->d_next_frontier = temp;
        frontier->current_size = next_size;
        
        iters++;
    }
    
    printf("BFS completed in %d iterations\n", iters);
}

// ==================== CC Host Function ====================
extern "C" void LaunchCC(GraphCSR* graph, CCLabels* cc_data, int max_memory_gb) {
    // Initialize labels (each node has its own label initially)
    int* h_labels;
    cudaMallocHost(&h_labels, graph->num_nodes * sizeof(int));
    
    for (int i = 0; i < graph->num_nodes; i++) {
        h_labels[i] = i;
    }
    
    cudaMemcpy(cc_data->d_labels, h_labels, graph->num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    
    int iters = 0;
    int convergence_flag = 1;
    
    while (convergence_flag && iters < cc_data->max_iterations) {
        convergence_flag = 0;
        
        // Launch CC kernel
        int num_blocks = (graph->num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CCKernel<<<num_blocks, BLOCK_SIZE>>>(
            graph->d_csr_row_offsets,
            graph->d_csr_col_indices,
            cc_data->d_labels,
            cc_data->d_next_labels,
            graph->num_nodes
        );
        
        cudaDeviceSynchronize();
        
        // Check convergence and swap labels
        // For memory efficiency, do this on CPU in limited memory scenarios
        cudaMemcpy(h_labels, cc_data->d_next_labels, 
                   graph->num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < graph->num_nodes; i++) {
            if (h_labels[i] != cc_data->d_labels[i]) {
                convergence_flag = 1;
            }
        }
        
        cudaMemcpy(cc_data->d_labels, h_labels, 
                   graph->num_nodes * sizeof(int), cudaMemcpyHostToDevice);
        
        iters++;
    }
    
    printf("CC completed in %d iterations\n", iters);
    
    cudaFreeHost(h_labels);
}

// ==================== SSSP Host Function ====================
extern "C" void LaunchSSSP(GraphCSR* graph, SSSPState* sssp_data,
                           int source_node, int max_memory_gb) {
    // Initialize distances
    for (int i = 0; i < graph->num_nodes; i++) {
        sssp_data->d_distances[i] = INF_FLOAT;
    }
    sssp_data->d_distances[source_node] = 0.0f;
    
    // Initialize queue with source
    sssp_data->d_queue[0] = source_node;
    sssp_data->queue_size = 1;
    
    int iters = 0;
    bool changed = true;
    
    while (changed && iters < MAX_ITERATIONS) {
        changed = false;
        
        // Launch SSSP kernel
        int num_blocks = (graph->num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        SSSPKernel<<<num_blocks, BLOCK_SIZE>>>(
            graph->d_csr_row_offsets,
            graph->d_csr_col_indices,
            sssp_data->d_distances,
            sssp_data->d_next_distances,
            graph->num_nodes,
            sssp_data->infinity
        );
        
        cudaDeviceSynchronize();
        
        // Check for changes
        int* h_distances;
        cudaMallocHost(&h_distances, 1024 * sizeof(float)); // Sample for convergence
        
        cudaMemcpy(h_distances, sssp_data->d_next_distances, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < 1024; i++) {
            if (h_distances[i] < sssp_data->d_distances[i]) {
                changed = true;
            }
        }
        
        cudaFreeHost(h_distances);
        
        // Swap distances
        float* temp = sssp_data->d_distances;
        sssp_data->d_distances = sssp_data->d_next_distances;
        sssp_data->d_next_distances = temp;
        
        iters++;
    }
    
    printf("SSSP completed in %d iterations\n", iters);
}

// ==================== PageRank Host Function ====================
extern "C" void LaunchPageRank(GraphCSR* graph, PageRankData* pr_data,
                               int max_memory_gb) {
    // Initialize ranks uniformly
    float initial_rank = 1.0f / graph->num_nodes;
    for (int i = 0; i < graph->num_nodes; i++) {
        pr_data->d_ranks[i] = initial_rank;
        pr_data->d_next_ranks[i] = initial_rank;
    }
    
    int iters = 0;
    float diff = 1.0f;
    
    while (diff > FLOAT_EPSILON && iters < pr_data->max_iterations) {
        diff = 0.0f;
        
        // Launch PageRank kernel
        int num_blocks = (graph->num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        PageRankKernel<<<num_blocks, BLOCK_SIZE>>>(
            graph->d_csr_row_offsets,
            graph->d_csr_col_indices,
            pr_data->d_ranks,
            pr_data->d_next_ranks,
            pr_data->d_out_degrees,
            pr_data->damping,
            graph->num_nodes
        );
        
        cudaDeviceSynchronize();
        
        // Check convergence
        int* h_ranks;
        cudaMallocHost(&h_ranks, 1024 * sizeof(float));
        
        cudaMemcpy(h_ranks, pr_data->d_next_ranks, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < 1024; i++) {
            float new_diff = fabsf(h_ranks[i] - pr_data->d_ranks[i]);
            if (new_diff > diff) {
                diff = new_diff;
            }
        }
        
        cudaFreeHost(h_ranks);
        
        // Swap ranks
        float* temp = pr_data->d_ranks;
        pr_data->d_ranks = pr_data->d_next_ranks;
        pr_data->d_next_ranks = temp;
        
        iters++;
    }
    
    printf("PageRank completed in %d iterations\n", iters);
}

// ==================== Memory Management Helpers ====================

extern "C" void* cuda_device_malloc(size_t size) {
    void* d_ptr;
    cudaMalloc(&d_ptr, size);
    return d_ptr;
}

extern "C" void cuda_device_free(void* ptr) {
    cudaFree(ptr);
}

extern "C" void cuda_host_free(void* ptr) {
    cudaFreeHost(ptr);
}

extern "C" int get_cuda_device_count() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

extern "C" void cuda_set_device(int device) {
    cudaSetDevice(device);
}

extern "C" void cuda_device_synchronize() {
    cudaDeviceSynchronize();
}