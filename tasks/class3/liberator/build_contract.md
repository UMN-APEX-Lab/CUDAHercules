# Liberator Black-Box Contract

You generate exactly one source file:

- `custom_cuda_backend/backend.cu`

The benchmark owns the build and links your source into a shared library. You
do not need to write `build.sh` or `CMakeLists.txt`.

## Fixed ABI

Your source must implement the symbols declared in `custom_backend_api.h`.

The benchmark loads your shared library with `dlopen()` and calls:

- `kh_liberator_backend_init`
- `kh_liberator_backend_shutdown`
- `kh_liberator_backend_last_error`
- `kh_liberator_bfs`
- `kh_liberator_cc`
- `kh_liberator_sssp`
- `kh_liberator_pr`

## Memory Budget

Before your backend runs, the benchmark reserves device memory so that only the
target budget remains available:

- 8 GB
- 10 GB
- 12 GB
- 15 GB

Your code should expect OOM pressure and design its partitioning / streaming
strategy accordingly.

## Graph Inputs

The benchmark passes host pointers to graph arrays:

- BFS / CC use an unweighted CSR graph
- SSSP uses a weighted CSR graph
- PageRank uses a CSC graph plus `out_degree`

The offset arrays have length `num_vertices`.
For the last vertex, the end offset is `num_edges`.

## Output Conventions

- BFS: output one `uint32_t` level per vertex; source = `1`, unreachable = `UINT32_MAX`
- CC: output one `uint32_t` label per vertex
- SSSP: output one `uint32_t` distance per vertex; source = `1`, unreachable = `UINT32_MAX`
- PR: output one `double` rank per vertex

Return `0` on success and non-zero on failure. On failure, report a short error
message via `kh_liberator_backend_last_error()`.
