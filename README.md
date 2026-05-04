# CUDA-Hercules

CUDA-Hercules is an expert-referenced benchmark for evaluating whether LLM-based systems can write high-performance, hardware-aware CUDA code.

This release is the paper task set: 205 tasks across four classes, with task metadata, runners, reference dependencies, validation scripts, and application fixtures included in the repository tree.

## Task Set

| Class | Scope | Tasks | Release slice |
|---|---:|---:|---|
| Class 1 | Single CUDA kernel | 73 | 20 general, 23 Hopper, 30 Blackwell |
| Class 2 | Module or kernel family | 119 | 43 general, 64 Hopper, 12 Blackwell |
| Class 3 | Full application workload | 10 | Blackwell app variants, plus `tcgnn_gcn/general` |
| Class 4 | Frontier challenge | 3 | FA4 forward, FA4 backward, Groth16 prover |

The machine-readable task list is [tasks/release_task_set.txt](tasks/release_task_set.txt). The runner should discover exactly 205 tasks.

```bash
python -c "import sys; sys.path.insert(0, 'src'); from cuda_hercules.runner import discover_tasks; print(len(discover_tasks('.')))"
```

## Repository Layout

```text
tasks/                  Released benchmark tasks and fixtures
src/cuda_hercules/      Unified runner, backends, prompt builder, scoring
scripts/                One-shot, self-refine, tool-augmented, and validation scripts
reference_sources/      Vendored CUDA reference dependencies
docs/                   Setup and evaluation notes
```

The release vendors the reference headers/sources needed by the harnesses,
including CUTLASS, FlashAttention, ThunderKittens, DeepGEMM, SageAttention,
cuFFT/VkFFT-derived references, LayerNorm references, and Icicle source. Class 3
includes practical fixtures in-tree and per-task preparation scripts for
generated or external inputs.

## Setup

Requirements: Linux, Python 3.10+, CUDA 12.x with `nvcc`, and an NVIDIA GPU matching the task architecture you evaluate.

```bash
git clone https://github.com/NKU-Yang/CUDAHercules.git
cd CUDAHercules

python -m venv .venv
source .venv/bin/activate

# Install a PyTorch wheel matching your CUDA runtime.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
bash scripts/setup_dependencies.sh
```

For the tool-augmented LangChain workflow:

```bash
pip install -e ".[agent]"
```

## Evaluation Settings

CUDA-Hercules includes scripts for the three evaluation settings used in the paper.

### One-shot / Pass@N

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_pass1.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --num-samples 1 \
  --filter backend=class2_defpy \
  --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name my_model_c2_general_pass1 \
  --output results
```

Use `--num-samples 3` or another `N` for pass@N.

### Iterative Self-Refine

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_self_refine.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --max-refine 9 \
  --filter backend=class1_make \
  --filter arch=general \
  --task-list tasks/class1/general/subset_20.txt \
  --run-name my_model_c1_general_refine10 \
  --output results
```

### Tool-Augmented

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_agent.py \
  --model "$MODEL" \
  --provider openai \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --max-iterations 20 \
  --profile-tools nsys,ncu \
  --filter backend=class2_defpy \
  --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name my_model_c2_general_toolaug \
  --output results
```

Cloud APIs can omit `--api-base` when the provider uses its default endpoint. Local vLLM/SGLang servers should expose an OpenAI-compatible `/v1` endpoint.

## Class 3 Applications

Class 3 application tasks have task-specific entry points under `scripts/eval_class3_*.py`; tool-augmented application evaluators are `scripts/eval_*_toolaug.py`.

Every Class 3 application directory has its own dataset preparation entry point:
`tasks/class3/<app>/prepare_data.sh` or `prepare_data.py`. Each concrete Class 3
task directory, such as `tasks/class3/<app>/blackwell`, also has a local
`prepare_data.sh` shim. A unified wrapper is also provided:

```bash
# Check which included datasets are already ready.
python scripts/prepare_class3_inputs.py --verify-only

# Regenerate lightweight included/generated fixtures.
python scripts/prepare_class3_inputs.py --tasks gpumd parafrost_sat

# Prepare large external datasets only when you explicitly want them.
pip install -e ".[class3-large]"
python scripts/prepare_class3_inputs.py --tasks liberator mgg_gcn --include-large-downloads
```

`liberator` downloads and converts SNAP Friendster and needs about 90 GB free
disk. `mgg_gcn` downloads `ogbn-papers100M` and also requires substantial disk
space. Other Class 3 tasks either include their fixtures, download small public
inputs, or generate deterministic inputs locally.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_class3_llmc.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --arch blackwell \
  --run-name my_model_llmc_blackwell \
  --output results
```

## Dataset Files

Class 3 input datasets and downloaded runtime blobs are intentionally not
committed. After cloning, prepare them with the per-task `prepare_data.sh` /
`prepare_data.py` scripts or with:

```bash
python scripts/prepare_class3_inputs.py --tasks gpumd parafrost_sat
```

The large external datasets for `liberator` and `mgg_gcn` remain opt-in via
`--include-large-downloads`.

## License

CUDA-Hercules is released under Apache-2.0. Individual task references retain their upstream licenses; source provenance is recorded in each `task.yaml`.
