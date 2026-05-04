# llmc Black-Box Project Contract

You generate a complete self-contained project under:

- `generated_project/`

The benchmark does not require compatibility with upstream `llmc/*.cuh`
symbols. Instead, your generated project must build and run according to the
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

You may use any project structure you want: single-file, multi-file, Makefile,
CMake, Python launcher, shell wrapper, or a custom build layout.

## CUDA Kernel Requirement

Your project must contain real custom CUDA kernels.

- `cuda_sources` must list the files that contain your custom CUDA kernels
- these files must include real `__global__` kernels and `<<<>>>` launches
- the benchmark rejects projects that only wrap a host program without custom
  CUDA kernels

## Required Ops Coverage

In `ops`, map each required training component to the file that implements it:

- `attention`
- `layernorm`
- `gelu`
- `encoder`
- `fused_classifier`
- `adamw`
- `global_norm`

Multiple ops may map to the same file.

`matmul` may use cuBLAS if you choose; it is not required in `ops`.

## Benchmark-Owned Inputs

The benchmark supplies the same inputs to the reference and to your project:

- train bin path
- val bin path
- GPT-2 large (774M) config — 36 layers, hidden 1280, 20 heads
- batch size
- sequence length
- learning rate
- number of steps
- validation interval
- random seed
- overfit-single-batch mode

You must not assume any hidden files beyond what the benchmark passes in.

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
bash run.sh \
  --train-bin <PATH> \
  --val-bin <PATH> \
  --steps 20 \
  --batch-size 4 \
  --seq-len 256 \
  --learning-rate 3e-4 \
  --val-every 10 \
  --seed 1337 \
  --dtype bf16 \
  --overfit-single-batch 1 \
  [--ref-time-ms <FLOAT>]
```

from the `generated_project/` directory.

Requirements:

- exit with code `0` on success and non-zero on failure
- train on the provided data and configuration
- print these lines so the benchmark can parse them:

```text
Passed
Kernel time: <ms> ms
Solution loss: <first_loss> -> <last_loss>
val loss: <first_val_loss> -> <last_val_loss>
Solution loss checkpoints: 25%=... | 50%=... | 75%=... | 100%=...
tok/s: <value>
Loss per step: v1,v2,...,vN
```

The final `Loss per step` line lists the training loss at every step, comma-separated
with no spaces (example: `11.0724,11.5579,13.8479,...,5.6863`). Must contain all `N`
per-step losses in order.

If `--ref-time-ms` is provided, you may also print:

```text
Ref time: <ms> ms
Speedup: <x>x
```

The benchmark independently checks correctness from your reported losses; do
not rely on printing `Passed` alone.

## Checkpoint requirement

Before `run.sh` exits, your project must write the final trained model weights
to the file `final_weights.bin` in the `generated_project/` directory. This
checkpoint is used by the benchmark for post-run validation. You may use any
binary format you like; it is sufficient that the file contain the bf16 (or
fp32) weights of your model.

## Correctness Expectations

Your run is considered correct only if:

- the process exits successfully
- the reported training loss decreases from first to last step
- the final loss is below `8.0`
- the loss trajectory reflects an honest training run on the specified model
  (hand-crafted / synthesized loss curves and shrunken / wrong-vocab models
  will be rejected)

Real GPT-2 training produces a noisy, non-monotone loss curve (e.g. the loss
often rises above initialization in the first few steps before descending). Any
hand-crafted / synthetic loss sequence will fail these checks.

## Anti-Cheat

Do not include or invoke the upstream reference implementation.
Do not depend on `tasks/class3/llmc/src` or `train_gpt2.cu`.
Your project should be self-contained.

Do not synthesize the reported losses: the benchmark verifies that your loss
trajectory resembles a real training run (non-linear, noisy, and roughly
consistent with the reference).
