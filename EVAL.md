# CUDA-Hercules Evaluation Guide

This guide covers the released 194-task benchmark tree.

## Verify The Release

```bash
python -c "import sys; sys.path.insert(0, 'src'); from cuda_hercules.runner import discover_tasks; print(len(discover_tasks('.')))"
```

Expected output: `194`.

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

## Agent Framework: KDA (Kernel Design Agents)

`scripts/eval_kda.py` runs [NVlabs KDA](https://github.com/NVlabs/kda) unattended on
Class 1 (`class1_make`) and Class 2 (`class2_defpy`) tasks, in "lite" mode: KDA's
workflow prompt (adapted for unattended runs with a budget section,
`scripts/kda_eval/kda_lite/autonomous-flow.md`), a budget hook
(`scripts/kda_eval/kda_lite/budget_guard.py`) and KDA's `ncu-report-skill`. KDA's
KernelWiki is not used (it embeds sources that overlap the reference
implementations), and neither is the humanize / Codex review loop. The agent is
driven by a small tool-calling loop over an OpenAI-compatible chat-completions API
(`scripts/kda_eval/agent_loop.py`) with four tools: `bash`, `read_file`,
`write_file`, `edit_file`.

Isolation:

- The agent loop and the API key stay outside the sandbox. Every tool call runs in a
  fresh `bwrap` sandbox with no network that contains only the task workspace, the
  system CUDA toolkit, the CUTLASS include directories solutions may `#include`, the
  ncu skill docs and one GPU. The repository, reference code, other runs' results
  and the host home directory do not exist inside.
- The agent can evaluate a candidate only through `submit`, which talks to a judge
  outside the sandbox (`scripts/kda_eval/judge.py`). The judge applies the task's
  static checks, runs the normal task evaluation and enforces the evaluation and
  wall-clock budget.
- After the session, the fastest timed PASS submission is re-evaluated in a clean
  process; that re-evaluation is the task's score.

Setup (Linux with unprivileged user namespaces):

```bash
git submodule update --init kda                               # upstream NVlabs/kda, pinned
git -C kda submodule update --init skills/ncu-report-skill    # do NOT init skills/KernelWiki
which bwrap || sudo apt-get install bubblewrap
python scripts/kda_eval/sandbox.py probe --gpu 0 --scan       # isolation self-check
```

The probe should report the host paths as absent, only a loopback interface, one
CUDA device, readable ncu counters and an empty leak scan.

The API key and base URL are read from environment variables, optionally loaded
from `--env-file`; their names default to `AWS_OPENAI_API_KEY` /
`AWS_OPENAI_BASE_URL` and can be changed with `--api-key-env` / `--api-base-env`
(`--api-base` overrides the URL directly).

```bash
python -u scripts/eval_kda.py \
  --model "$MODEL" \
  --env-file /path/to/.env \
  --gpus 0,1 \
  --filter arch=general \
  --time-limit-min 60 \
  --max-evals 10 \
  --output results/kda \
  --run-name kda_lite_general
```

Each GPU in `--gpus` runs one task at a time. Budget and model options:

| Flag | Default | Meaning |
|---|---|---|
| `--time-limit-min` | 60 | wall-clock budget per task |
| `--max-evals` | 10 | judge evaluations per task |
| `--max-llm-calls` | 1000 | LLM calls per task |
| `--max-total-tokens` | 0 (none) | input + output tokens per task |
| `--eval-timeout` | 900 | seconds per judge evaluation |
| `--target-speedup` | 1.0 | speedup at which a candidate counts as promoted |
| `--max-completion-tokens` | 32000 | per LLM call |
| `--context-limit` | 150000 | prompt size at which old tool outputs are elided |
| `--reasoning-effort` | unset | passed through to the API if set |

Re-running with the same `--run-name` resumes: tasks that already have a
`result.json` are skipped. `--summarize-only` re-runs the hacking audit and
rewrites `summary.json` without running tasks.

Per-task output (`<output>/<run-name>/<task_id with / replaced by _>/`):

- `prompt.md`, `transcript.jsonl`, `messages.json`: the prompt and the full session
- `workspace/`: the agent's files, including KDA's `docs/draft.md`, `docs/plan.md`,
  `candidates.jsonl` and `benchmark.csv`
- `submissions/` and `submissions.jsonl`: every judged submission and its result
- `result.json`: `best_speedup` (the clean re-evaluation), evaluations used and
  session statistics (LLM calls, tokens, wall time); `best_<solution file>`: the
  scored solution

`summary.json` aggregates the run: pass rate, mean speedup over all tasks
(failures count as 0) and token totals. A task whose final solution fails the
task's current static checks, or that is flagged in `<run>/_audit.json`, is counted
as FAIL; `result.json` and all artifacts are left unchanged.

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
