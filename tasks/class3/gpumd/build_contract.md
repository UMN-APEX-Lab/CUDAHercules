# gpumd Black-Box Project Contract

You generate a complete self-contained project under:

- `generated_project/`

The benchmark does not require compatibility with the upstream `md_kernels.cuh`
layout or GPUMD source tree. Your project must build and run according to the
contract below.

## Generation Flow

The benchmark first asks you for `project_manifest.json`, then asks you for the
listed files one by one.

Your manifest must define:

- `build_script`
- `run_script`
- `files`
- `cuda_sources`
- `ops`

The benchmark writes `project_manifest.json` itself from your manifest JSON.
Do not include `project_manifest.json` in the `files` list.

## Required Project Files

Your project must include at least:

- `build.sh`
- `run.sh`
- one or more CUDA/C++ source files

You may use any internal project layout you want.

## CUDA Requirement

Your project must contain real custom CUDA kernels.

- `cuda_sources` must list the files that contain your custom CUDA kernels
- these files must include real `__global__` kernels and `<<<>>>` launches
- the benchmark rejects projects that only wrap a host program without custom
  CUDA kernels

## Required Ops Coverage

In `ops`, map each required simulation component to the file that implements
it:

- `neighbor_list`
- `integration`
- `kinetic_energy`
- `lennard_jones`
- `tersoff`
- `coulomb_real`
- `coulomb_kspace`

Multiple ops may map to the same file.

## Benchmark-Owned Inputs

The benchmark supplies standardized molecular dynamics workloads under an
`--input-dir` directory.

Your project must NOT parse the original upstream `.xyz` files. The benchmark
already converts each workload into a binary format with:

- material system identifier
- atom count
- simulation steps
- timestep
- box dimensions
- material / force-field parameters
- initial positions
- initial velocities
- atom types
- charges

The input directory contains:

- `workloads.json`
- `workloads/<name>.bin` for each workload

### workloads.json schema

`workloads.json` contains:

- `version`
- `format`
- `workloads`

Each workload entry contains:

- `name`
- `system`
- `file`
- `atom_count`
- `steps`
- `energy_drift_tol`
- `force_rel_tol`

## Standardized Input Binary Format

Each workload binary is little-endian and has this layout:

1. Header
2. `x[N]` as `float64`
3. `y[N]` as `float64`
4. `z[N]` as `float64`
5. `vx[N]` as `float64`
6. `vy[N]` as `float64`
7. `vz[N]` as `float64`
8. `atom_type[N]` as `int32`
9. `charge[N]` as `float32`

Header layout:

- `magic[8]`: ASCII `KHGPMD1\0`
- `version`: `uint32`
- `system_kind`: `uint32`
  - `1 = lj`
  - `2 = tersoff`
  - `3 = coulomb`
- `atom_count`: `uint32`
- `steps`: `uint32`
- `dt`: `float64`
- `mass`: `float64`
- `energy_drift_tol`: `float64`
- `force_rel_tol`: `float64`
- `box_x`: `float64`
- `box_y`: `float64`
- `box_z`: `float64`
- `params[16]`: `float64[16]`

`params` usage:

- LJ:
  - `params[0] = epsilon`
  - `params[1] = sigma`
  - `params[2] = cutoff`
- Tersoff:
  - `params[0..13] = A, B, lambda, mu, beta, n, c, d, h, R1, R2, m, alpha, gamma`
- Coulomb:
  - `params[0] = alpha`
  - `params[1] = cutoff`

## Required Candidate Outputs

The benchmark runs:

```bash
bash run.sh --input-dir <INPUT_DIR> --output-dir <OUTPUT_DIR> [--ref-time-ms <FLOAT>]
```

Your project must:

- execute all workloads listed in `workloads.json`
- produce `<OUTPUT_DIR>/benchmark_results.json`
- produce one force-output binary per workload

### benchmark_results.json schema

Write a JSON object with:

- `version`
- `results`

`results` must be a list with one entry per workload. Each entry must contain:

- `name`
- `force_file`
- `energy_drift`
- `time_ms`

`force_file` is relative to `<OUTPUT_DIR>`.

### Force output binary format

Each force output file must use this little-endian layout:

1. `atom_count` as `int32`
2. `fx[N]` as `float64`
3. `fy[N]` as `float64`
4. `fz[N]` as `float64`
5. `pe[N]` as `float64`

The benchmark compares these arrays against benchmark-owned reference forces.

## build.sh Contract

The benchmark runs:

```bash
bash build.sh
```

from the `generated_project/` directory.

Requirements:

- exit with code `0` on success
- respect the current `CUDA_VISIBLE_DEVICES`
- respect the environment variable `CUDA_GENCODE` if it is set

## run.sh Contract

The benchmark runs:

```bash
bash run.sh --input-dir <INPUT_DIR> --output-dir <OUTPUT_DIR> [--ref-time-ms <FLOAT>]
```

from the `generated_project/` directory.

Requirements:

- exit with code `0` on success and non-zero on failure
- process every workload in `workloads.json`
- write `benchmark_results.json`
- write each workload's force output file
- print these lines so the benchmark can parse a summary:

```text
KERNEL <name>: energy_drift=<float> time_ms=<float> force_file=<relative_path>
Passed
Kernel time: <float> ms
```

If `--ref-time-ms` is provided, you may also print:

```text
Ref time: <ms> ms
Speedup: <x>x
```

The benchmark independently checks correctness from your force output files and
reported energy drift; do not rely on printing `Passed` alone.

## Correctness Expectations

Your run is considered correct only if:

- the process exits successfully
- every required workload appears in `benchmark_results.json`
- every referenced force output file exists and matches the benchmark-owned
  reference within tolerance
- every workload's `energy_drift` is below its benchmark threshold

## Anti-Cheat

Do not include or invoke the upstream GPUMD reference implementation.
Do not depend on `tasks/class3/gpumd/src` or `md_bench`.
Do not read benchmark-private reference force files or benchmark-private cache
directories such as `ref_forces` or `.blackbox_inputs_v1` except through the
explicit `--input-dir` interface passed to `run.sh`.
Your project should be self-contained.
