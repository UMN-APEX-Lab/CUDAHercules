# Task 57: Tencent HPC-Ops reference

Measured on NVIDIA H200 NVL, SM90a, 2026-11-08. The selected three-stage
`cp.async` configuration is now the timed reference. CUTLASS remains a
complete-output oracle and a fallback for unsupported tile alignments.

## Adopted reference and KDA score

The integrated reference measured 39.9 / 442.7 / 399.4 us, using the median of
three process minima (20 cold-L2 trials each). The H200 reference cache contains
these values with the new task source fingerprint.

At the user's request, task 57 was **not executed again** during adoption.
Its original formal KDA result supplied the recorded kernel times
48.2 / 457.1 / 406.9 us. Replacing only the reference denominators gives
0.827801x / 0.968497x / 0.981568x, whose geometric mean is **0.923240490x**.
This replaces the score **1.448903172x** in the continuation run. The original
run and a `reference_history` copy retain the original result and timings.
All three evaluations and 47 model calls remain charged; no new calls were made.

`scripts/rebase_reference_results.py` performs this migration using JSON only,
checks every case identity, and preserves the recorded kernel times and budgets.
Its provenance explicitly records that correctness comes from the prior PASS
and cross-validation of the two references, without a new kernel execution.
KDA carries forward all 13 completed tasks and continues at task 68.

## Selection experiment

The adapted HPC-Ops `cp.async` implementation is a stronger reference candidate.
Its best single configuration uses M/N/K tiles 64/64/128, three pipeline stages,
128 threads, a grid of three CTAs per SM, and a minimum occupancy bound of three
CTAs. The same configuration is used for all three cases.

Each number below is the median of three independent process runs. Within each
run, each implementation uses the minimum of 20 CUDA-event trials with the same
128 MiB cold-L2 eviction policy. The reference column is measured in the same
processes as the selected HPC-Ops configuration, rather than taken from cache.

| Case | Previous CUTLASS reference | HPC-Ops candidate | Reference / candidate | Frozen KDA code, fresh control |
|---|---:|---:|---:|---:|
| moe_decode | 98.2 us | 40.1 us | 2.449x | 48.5 us |
| moe_prefill | 557.2 us | 443.5 us | 1.256x | 451.9 us |
| moe_large | 506.8 us | 401.1 us | 1.264x | 402.5 us |

The geometric mean of the three reference/candidate ratios is **1.572375x**.
The saved KDA solution's latency relative to this candidate corresponds to a
geometric-mean speedup of **0.931637x**, versus its historical score of
**1.448903x** against the previous cached reference. The 0.931637x figure belongs
to the selection experiment's fresh controls. Adoption instead uses the original
formal recorded kernel times and the integrated reference, as described above.

HPC-Ops' default two-stage `cp.async` configuration measured
43.9 / 473.8 / 434.9 us. The best overall TMA configuration measured
46.2 / 479.1 / 392.4 us, or **1.465010x** over its concurrent reference.
TMA is slightly faster on the large case, but the `cp.async` configuration has
the stronger overall result and a simpler adaptation to the public pointer ABI.

## Public source and adaptations

Source: [Tencent/hpc-ops at 2a2e265](https://github.com/Tencent/hpc-ops/tree/2a2e26562433a8ba4b504858f1c938eb7612c901),
MIT license, retained with the generated experimental sources.

- `src/group_gemm/sm90/cp_async/group_gemm_fp8.cu` and `config.h`: keep the
  upstream `cp.async`/WGMMA pipeline and per-K-tile FP32 accumulator promotion.
  The swapped first MMA operand becomes E5M2 and the second E4M3, preserving the
  actual input bytes without conversion to another FP8 format. Device pointer,
  shape and stride arrays replace the upstream contiguous-buffer interface.
  An FP16 column-major epilogue replaces its BF16 row-major epilogue.
- `src/group_gemm/sm90/kernels.cuh` and `config.h`: compare the upstream TMA
  pipeline after the same mixed-FP8 and FP16 changes. Keep its row-major output
  and explicitly transpose to column-major on every measured call. Descriptor
  updates, GEMM and the output transpose are all inside the timed operation.
  Buffer allocation, host construction of static descriptors and configuration
  happen before timing, as with the existing prepared CUTLASS reference.

The `cp.async` sweep covers 11 configurations and TMA covers six, with three
process repeats each. Both reference and candidate are built with `-O3`,
`--use_fast_math`, `-DNDEBUG`, `CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED` and `sm_90a`.
The extended MMA flag is needed for the upstream 48-wide MMA configuration.
Both use the benchmark's existing CUTLASS headers.

All 51 configuration/repeat experiments passed all three default cases. Each
reference is checked using the harness's independent sampled FP64 GEMM oracle;
the complete candidate output is compared with the reference using the existing
elementwise and normalized-RMS checks. Output buffers are poisoned beforehand.
Three fresh KDA control runs also passed.

The winning `cp.async` configuration additionally passed two further input seeds
on all default cases, a one-group 16-row case, a ragged nine-group case with
alpha=0.37, and a seven-group case including an empty group with alpha=-0.75.

## Reproduction and adoption boundary

Clone the pinned source into `.cache/production_gemm/hpc-ops`, then run:

```bash
git clone https://github.com/Tencent/hpc-ops.git .cache/production_gemm/hpc-ops
git -C .cache/production_gemm/hpc-ops checkout 2a2e26562433a8ba4b504858f1c938eb7612c901
.venv/bin/python scripts/benchmark_hopper57_hpc_ops.py
.venv/bin/python scripts/benchmark_hopper57_hpc_ops.py --backend tma --output results/hopper57_hpc_ops_tma
```

The script compiles isolated copies and acquires `.cache/hopper_gpu.lock` before
GPU work. `--prepare-only` only builds; `--skip-build` verifies source and binary
fingerprints before reusing a build. No LLM requests or KDA evaluations are used.

The `cp.async` path requires K divisible by 128 and N divisible by 64. These
conditions hold for all default task-57 cases; other alignments use CUTLASS.
The integrated reference passes default cases, the independent sampled FP64
oracle, full-output CUTLASS cross-validation, a positive CUDA control, and a
poisoned-output no-op negative control. An N=48/K=64 diagnostic verifies the
CUTLASS fallback. The public task ABI and input generation are unchanged.
The separate TMA experiment also requires contiguous grouped buffers and
uniform N/K; it was not selected for the integrated reference.

Machine-readable results, all configuration medians, source hashes, diagnostic
checks and pause state are in `docs/validation/hopper57_hpc_ops.json`. Complete
logs and generated sources are under `results/hopper57_hpc_ops`,
`results/hopper57_hpc_ops_tma` and `results/hopper57_hpc_ops_kda_control`.
