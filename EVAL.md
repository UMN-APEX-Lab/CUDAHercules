# CUDA-Hercules Evaluation Guide

This guide covers the released 195-task benchmark tree.

## Verify The Release

```bash
python -c "import sys; sys.path.insert(0, 'src'); from cuda_hercules.runner import discover_tasks; print(len(discover_tasks('.')))"
```

Expected output: `195`.

## One-Shot / Pass@N

`scripts/eval_pass1.py` generates `N` independent samples per task and scores a task as pass@N if any sample is correct.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_pass1.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --num-samples 1 \
  --temperature 0.6 \
  --filter backend=class2_defpy \
  --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name c2_general_pass1 \
  --output results
```

For pass@3:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_pass1.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --num-samples 3 \
  --temperature 0.8 \
  --filter backend=class1_make \
  --filter arch=general \
  --task-list tasks/class1/general/subset_20.txt \
  --run-name c1_general_pass3 \
  --output results
```

## Iterative Self-Refine

`scripts/eval_self_refine.py` runs initial generation plus `--max-refine` feedback rounds.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_self_refine.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --max-refine 9 \
  --temperature 0.6 \
  --filter backend=class2_defpy \
  --filter arch=general \
  --task-list tasks/class2/general/subset_43.txt \
  --run-name c2_general_selfrefine10 \
  --output results
```

## Tool-Augmented

`scripts/eval_agent.py` runs a fixed controller that lets the model write code, inspect feedback, and optionally call Nsight Systems / Nsight Compute profilers.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_agent.py \
  --model "$MODEL" \
  --provider openai \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --max-iterations 20 \
  --task-timeout 900 \
  --profile-tools nsys,ncu \
  --filter backend=class1_make \
  --filter arch=general \
  --task-list tasks/class1/general/subset_20.txt \
  --run-name c1_general_toolaug20 \
  --output results
```

If profilers are unavailable:

```bash
--profile-tools ""
```

## Class 3 Applications

Prepare or verify the Class 3 input datasets before running application
evaluators:

```bash
python scripts/prepare_class3_inputs.py --verify-only
python scripts/prepare_class3_inputs.py --tasks gpumd parafrost_sat
```

Each application also has a direct local entry point under
`tasks/class3/<app>/prepare_data.sh` or `prepare_data.py`, and each concrete
Class 3 task directory has a `prepare_data.sh` shim. Large external datasets are
opt-in:

```bash
pip install -e ".[class3-large]"
python scripts/prepare_class3_inputs.py --tasks liberator mgg_gcn --include-large-downloads
```

Use the task-specific scripts for application workloads:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_class3_parafrost_sat.py \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --api-key "${API_KEY:-dummy}" \
  --arch blackwell \
  --run-name parafrost_blackwell_pass1 \
  --output results
```

Available application scripts:

```text
scripts/eval_class3_cuszp.py
scripts/eval_class3_exachem_ccsd_t.py
scripts/eval_class3_gpumd.py
scripts/eval_class3_icicle_zk.py
scripts/eval_class3_liberator.py
scripts/eval_class3_llmc.py
scripts/eval_class3_mgg_agnn.py
scripts/eval_class3_mgg_gcn.py
scripts/eval_class3_parafrost_sat.py
scripts/eval_class3_tcgnn_gcn.py
```

## Output

Each run writes a directory under `results/` containing:

- `config.json` or `eval_config.json`
- per-task prompts and generated solutions
- per-task `result.json`
- aggregate `score.json` and `report.txt` when supported

Interrupted runs can be resumed by reusing the same `--run-name`.
