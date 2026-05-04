# parafrost_sat Black-Box Project Contract

You generate a complete self-contained project under:

- `generated_project/`

The benchmark does not require compatibility with the upstream ParaFROST
source tree. Your generated project must build and run according to the
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
- the benchmark rejects projects that only wrap a CPU SAT solver without
  custom CUDA kernels

## Search Algorithm Requirement

Your search core must be a **complete** SAT algorithm — CDCL (Conflict-Driven
Clause Learning) or equivalent — capable of producing `UNSAT` verdicts with
proof-style soundness. The benchmark's test subset includes UNSAT instances
drawn from real SAT Competition workloads; a solver that cannot prove UNSAT
will fail correctness on those instances.

**Incomplete algorithms are out of scope**:

- WalkSAT / GSAT / random-walk local search
- Survey propagation variants that short-circuit on SAT witness only
- Any solver that returns `UNKNOWN` on an UNSAT instance within the 1800 s
  timeout

The solver may use any standard CDCL machinery (BCP, VSIDS / EVSIDS / VMTF
decision heuristics, restart policies, clause-learning, LBD, vivification,
etc.) and any of the GPU inprocessing pipeline ops below.

## Required Ops Coverage

In `ops`, map each required SAT backend component to the file that implements
it:

- `occurrence_table`
- `variable_scoring`
- `elimination`
- `subsumption`
- `blocked_clause_elimination`
- `equivalence_reasoning`
- `redundancy_elimination`
- `memory_recycling`

Multiple ops may map to the same file.

## Benchmark-Owned Inputs

The benchmark supplies standardized SAT workloads under an `--input-dir`
directory.

Your project must not read the upstream `tasks/class3/parafrost_sat/src`,
`dep`, or `data` directories. Use only the files provided through
`--input-dir`.

The input directory contains:

- `instances.json`
- `instances/<name>.cnf` for each DIMACS instance

### instances.json schema

`instances.json` contains:

- `version`
- `format`
- `instances`

Each instance entry contains:

- `name`
- `file`
- `description`

`file` is relative to the input directory and points to a DIMACS CNF file.

## Required Candidate Outputs

The benchmark runs:

```bash
bash run.sh --input-dir <INPUT_DIR> --output-dir <OUTPUT_DIR> [--ref-time-ms <FLOAT>]
```

Your project must:

- execute every instance listed in `instances.json`
- produce `<OUTPUT_DIR>/benchmark_results.json`

### benchmark_results.json schema

Write a JSON object with:

- `version`
- `results`

`results` must be a list with one entry per instance. Each entry must contain:

- `name`
- `verdict`
- `time_ms`
- `assignment` (required when `verdict == "SAT"`; may be omitted for UNSAT/UNKNOWN)

`verdict` must be one of:

- `SAT`
- `UNSAT`
- `UNKNOWN`

### `assignment` format (required for SAT verdicts)

When your solver reports `verdict: "SAT"` for an instance, you MUST also
include the satisfying assignment as a list of signed DIMACS literals:

```json
{
  "name": "<instance_name>",
  "verdict": "SAT",
  "time_ms": 123.4,
  "assignment": [1, -2, 3, 4, -5, ... , N]
}
```

- Each literal is a non-zero integer. `+i` means variable `i = true`,
  `-i` means variable `i = false`.
- The assignment must reference only variables in the range `[1, num_vars]`
  declared by the CNF's `p cnf` header.
- The assignment must satisfy **every** clause in the CNF: for each clause,
  at least one literal in that clause must match the assignment.
- The benchmark verifies the assignment against the original CNF. Claiming
  `SAT` without a verifiable model is rejected.

You may include additional fields (e.g., extra statistics), but the four
above (`name`, `verdict`, `time_ms`, plus `assignment` when SAT) are required.

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
- process every instance listed in `instances.json`
- write `benchmark_results.json`
- print these lines so the benchmark can parse a summary:

```text
INSTANCE <name>: verdict=<SAT|UNSAT|UNKNOWN> time_ms=<float>
Passed
Kernel time: <float> ms
```

If `--ref-time-ms` is provided, you may also print:

```text
Ref time: <ms> ms
Speedup: <x>x
```

The benchmark independently checks correctness from your JSON output; do not
rely on printing `Passed` alone.

## Correctness Expectations

Your run is considered correct only if:

- the process exits successfully
- every required instance appears in `benchmark_results.json`
- every reported verdict matches the benchmark-owned reference verdict

## Anti-Cheat

Do not include or invoke the upstream ParaFROST reference implementation.
Do not depend on `tasks/class3/parafrost_sat/src`, `dep`, or `data`.
Do not read benchmark-private cache directories such as `.blackbox_inputs_v1`
except through the explicit `--input-dir` interface passed to `run.sh`.
Your project should be self-contained.
