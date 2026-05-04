# Setup And Evaluation

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
bash scripts/setup_dependencies.sh
```

For the tool-augmented workflow:

```bash
pip install -e ".[agent]"
```

For the optional large Class 3 graph conversion path:

```bash
pip install -e ".[class3-large]"
```

## Sanity Checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvcc --version
python -c "import sys; sys.path.insert(0, 'src'); import cuda_hercules; print('OK')"
python -c "import sys; sys.path.insert(0, 'src'); from cuda_hercules.runner import discover_tasks; print(len(discover_tasks('.')))"
```

The final command should print `205`.

## Pinned Subsets

Use pinned subset files for comparable paper-style runs:

| Target | File | Tasks |
|---|---|---:|
| Class 1 general | `tasks/class1/general/subset_20.txt` | 20 |
| Class 1 Hopper | `tasks/class1/hopper/subset_23.txt` | 23 |
| Class 2 general | `tasks/class2/general/subset_43.txt` | 43 |
| Class 2 Hopper sampled subset | `tasks/class2/hopper/subset_28.txt` | 28 |
| Full release | `tasks/release_task_set.txt` | 205 |

The release also includes all 64 Class 2 Hopper tasks and all 12 Class 2 Blackwell tasks.

## Class 3 Input Preparation

Each Class 3 application owns a local preparation script:

```text
tasks/class3/cuszp/prepare_data.sh
tasks/class3/exachem_ccsd_t/prepare_data.sh
tasks/class3/gpumd/prepare_data.sh
tasks/class3/icicle_zk/prepare_data.sh
tasks/class3/liberator/prepare_data.sh
tasks/class3/llmc/prepare_data.sh
tasks/class3/mgg_agnn/prepare_data.sh
tasks/class3/mgg_gcn/prepare_data.sh
tasks/class3/parafrost_sat/prepare_data.sh
tasks/class3/tcgnn_gcn/prepare_data.sh
```

The concrete `task.yaml` directories also include a local `prepare_data.sh`
shim, for example `tasks/class3/gpumd/blackwell/prepare_data.sh` and
`tasks/class3/tcgnn_gcn/general/prepare_data.sh`.

You can run them directly from each task directory, or use the repository-level
wrapper:

```bash
python scripts/prepare_class3_inputs.py --verify-only
python scripts/prepare_class3_inputs.py --tasks gpumd parafrost_sat
python scripts/prepare_class3_inputs.py --tasks llmc tcgnn_gcn
```

Large external downloads are skipped unless explicitly enabled:

```bash
pip install -e ".[class3-large]"
python scripts/prepare_class3_inputs.py --tasks liberator mgg_gcn --include-large-downloads
```

`liberator` needs the SNAP Friendster graph and about 90 GB free disk. `mgg_gcn`
needs `ogbn-papers100M` and tens of GB of free disk.

## Commands

One-shot/pass@N:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_pass1.py \
  --model "$MODEL" --api-base "$API_BASE" --api-key "${API_KEY:-dummy}" \
  --num-samples 1 \
  --filter backend=class2_defpy --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name c2_general_pass1 \
  --output results
```

Self-refine:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_self_refine.py \
  --model "$MODEL" --api-base "$API_BASE" --api-key "${API_KEY:-dummy}" \
  --max-refine 9 \
  --filter backend=class2_defpy --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name c2_general_refine10 \
  --output results
```

Tool-augmented:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_agent.py \
  --model "$MODEL" --provider openai \
  --api-base "$API_BASE" --api-key "${API_KEY:-dummy}" \
  --max-iterations 20 --profile-tools nsys,ncu \
  --filter backend=class1_make --filter arch=general \
  --task-list tasks/class1/general/subset_20.txt \
  --run-name c1_general_toolaug20 \
  --output results
```

Class 3 application example:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_class3_llmc.py \
  --model "$MODEL" --api-base "$API_BASE" --api-key "${API_KEY:-dummy}" \
  --arch blackwell \
  --run-name llmc_blackwell_pass1 \
  --output results
```
