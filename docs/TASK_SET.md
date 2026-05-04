# Released Task Set

This directory is curated to match the 205-task paper release.

## Counts

| Slice | Count |
|---|---:|
| Class 1 general | 20 |
| Class 1 Hopper | 23 |
| Class 1 Blackwell | 30 |
| Class 2 general | 43 |
| Class 2 Hopper | 64 |
| Class 2 Blackwell | 12 |
| Class 3 applications | 10 |
| Class 4 challenges | 3 |
| Total | 205 |

## Selection Rules

Class 1 includes every released task under `tasks/class1/{general,hopper,blackwell}`.

Class 2 includes:

- `tasks/class2/general/subset_43.txt`
- all directories under `tasks/class2/hopper`
- all directories under `tasks/class2/blackwell`

Class 3 includes one `task.yaml` per application workload:

- `cuszp/blackwell`
- `exachem_ccsd_t/blackwell`
- `gpumd/blackwell`
- `icicle_zk/blackwell`
- `liberator/blackwell`
- `llmc/blackwell`
- `mgg_agnn/blackwell`
- `mgg_gcn/blackwell`
- `parafrost_sat/blackwell`
- `tcgnn_gcn/general`

Each Class 3 application directory also includes a local dataset preparation
entry point, either `prepare_data.sh` or `prepare_data.py`. Each concrete
`task.yaml` directory has a `prepare_data.sh` shim as well. The repository-level
wrapper is `scripts/prepare_class3_inputs.py`.

Class 4 includes every task under `tasks/class4`.

The authoritative manifest is `tasks/release_task_set.txt`.
