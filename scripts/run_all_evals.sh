#!/bin/bash
#
# CUDA-Hercules: One-Click Full Evaluation
#
# Runs pass@1 and pass@10 across all task classes (Class 1, 2, 3).
#
# Usage:
#   # Local vLLM server
#   bash scripts/run_all_evals.sh \
#     --model "Qwen/Qwen3.5-35B-A3B" \
#     --api-base "http://localhost:8000/v1"
#
#   # OpenAI API
#   bash scripts/run_all_evals.sh \
#     --model "gpt-4o" \
#     --api-key "sk-..."
#
#   # Custom settings
#   bash scripts/run_all_evals.sh \
#     --model "deepseek-coder" \
#     --api-base "https://api.deepseek.com/v1" \
#     --api-key "$DEEPSEEK_API_KEY" \
#     --gpu 0 \
#     --conda-env cf
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────

MODEL=""
API_BASE=""
API_KEY=""
GPU_ID=0
CONDA_ENV="cf"
TEMPERATURE=0.6
MAX_TOKENS=16384
OUTPUT_DIR="results"
PASS_AT_N="1 10"
ARCH=""  # empty = all, or "general", "hopper", "blackwell"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────

usage() {
    echo "Usage: $0 --model MODEL [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --model MODEL           LLM model name (e.g., gpt-4o, Qwen/Qwen3.5-35B-A3B)"
    echo ""
    echo "API endpoint (one of):"
    echo "  --api-base URL          API base URL (for vLLM/local servers)"
    echo "  --api-key KEY           API key (for cloud APIs like OpenAI)"
    echo ""
    echo "Options:"
    echo "  --gpu ID                GPU device ID (default: 0)"
    echo "  --conda-env ENV         Conda environment name (default: cf)"
    echo "  --temperature FLOAT     Sampling temperature (default: 0.6)"
    echo "  --max-tokens INT        Max output tokens (default: 16384)"
    echo "  --output DIR            Output directory (default: results)"
    echo "  --pass-at-n '1 10'      Space-separated N values (default: '1 10')"
    echo "  --arch ARCH             Filter by architecture: general, hopper, blackwell (default: all)"
    echo "  --help                  Show this help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --api-base)     API_BASE="$2"; shift 2 ;;
        --api-key)      API_KEY="$2"; shift 2 ;;
        --gpu)          GPU_ID="$2"; shift 2 ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --temperature)  TEMPERATURE="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --pass-at-n)    PASS_AT_N="$2"; shift 2 ;;
        --arch)         ARCH="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    usage
fi

# ── Derived variables ─────────────────────────────────────────────────

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${MODEL_SLUG}_${TIMESTAMP}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# Build API args
API_ARGS="--model $MODEL --max-tokens $MAX_TOKENS"
if [ -n "$API_BASE" ]; then
    API_ARGS="$API_ARGS --api-base $API_BASE"
fi
if [ -n "$API_KEY" ]; then
    export OPENAI_API_KEY="$API_KEY"
fi

# Architecture filter
ARCH_FILTER=""
ARCH_LABEL="all"
if [ -n "$ARCH" ]; then
    ARCH_LABEL="$ARCH"
    ARCH_FILTER="--filter arch=$ARCH"
fi

# Conda run prefix
RUN="env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 conda run --no-capture-output -n $CONDA_ENV python -u"

# ── Banner ────────────────────────────────────────────────────────────

cat << EOF
================================================================================
  CUDA-Hercules Full Evaluation
================================================================================
  Model:       $MODEL
  API Base:    ${API_BASE:-"(default OpenAI)"}
  GPU:         $GPU_ID
  Conda Env:   $CONDA_ENV
  Temperature: $TEMPERATURE
  Pass@N:      $PASS_AT_N
  Output:      $RUN_DIR
  Timestamp:   $TIMESTAMP
================================================================================
EOF

# Save config
cat > "$RUN_DIR/eval_config.json" << CFGEOF
{
    "model": "$MODEL",
    "api_base": "$API_BASE",
    "gpu_id": $GPU_ID,
    "conda_env": "$CONDA_ENV",
    "temperature": $TEMPERATURE,
    "max_tokens": $MAX_TOKENS,
    "pass_at_n": "$PASS_AT_N",
    "timestamp": "$TIMESTAMP"
}
CFGEOF

# ── Run evaluation ───────────────────────────────────────────────────

run_eval() {
    local name="$1"
    local log="${LOG_DIR}/${name}.log"
    shift
    echo "[$(date +%H:%M:%S)] Starting: $name"
    echo "  Log: $log"
    $RUN "$@" > "$log" 2>&1 || true
    echo "[$(date +%H:%M:%S)] Done: $name"
}

for N in $PASS_AT_N; do
    TEMP=$TEMPERATURE
    if [ "$N" -gt 1 ]; then
        TEMP=0.8  # higher temperature for pass@N>1
    fi

    echo ""
    echo ">>> Pass@${N} (temperature=${TEMP})"
    echo ""

    # Class 1
    run_eval "class1_${ARCH_LABEL}_pass${N}" \
        "$PROJECT_ROOT/scripts/eval_pass1.py" \
        $API_ARGS --temperature "$TEMP" --num-samples "$N" \
        --filter "backend=class1_make" $ARCH_FILTER \
        --run-name "${RUN_NAME}_class1_${ARCH_LABEL}_pass${N}" \
        --output "$RUN_DIR"

    # Class 2
    run_eval "class2_${ARCH_LABEL}_pass${N}" \
        "$PROJECT_ROOT/scripts/eval_pass1.py" \
        $API_ARGS --temperature "$TEMP" --num-samples "$N" \
        --filter "backend=class2_defpy" $ARCH_FILTER \
        --run-name "${RUN_NAME}_class2_${ARCH_LABEL}_pass${N}" \
        --output "$RUN_DIR"

    # Class 3 (no arch subdivision, always run all)
    run_eval "class3_pass${N}" \
        "$PROJECT_ROOT/scripts/eval_pass1.py" \
        $API_ARGS --temperature "$TEMP" --num-samples "$N" \
        --filter "backend=class3_app" \
        --run-name "${RUN_NAME}_class3_pass${N}" \
        --output "$RUN_DIR"
done

# ── Aggregate Results ─────────────────────────────────────────────────

echo ""
echo ">>> Aggregating Results"

python3 -u << 'PYEOF'
import json, os, glob

run_dir = os.environ.get("RUN_DIR", "PLACEHOLDER")
PYEOF_REAL="
import json, os, glob

run_dir = '$RUN_DIR'
results = {}

for f in sorted(glob.glob(os.path.join(run_dir, '*/score.json'))):
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        results[name] = json.load(fh)

with open(os.path.join(run_dir, 'aggregate.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print()
print('=' * 70)
print('CUDA-Hercules Evaluation Summary')
print('=' * 70)
print()
print(f'{\"Eval\":<45} {\"Compile\":>8} {\"Correct\":>8} {\"fastp1\":>8} {\"GeoMean\":>8}')
print('-' * 70)

for name in sorted(results.keys()):
    r = results[name]
    c = r.get('compilation_rate', 0)
    cr = r.get('correctness_rate', 0)
    f1 = r.get('fastp_1_0', 0)
    g = r.get('geo_mean_speedup', 0)
    c_str = f'{c:.1%}' if isinstance(c, float) else str(c)
    cr_str = f'{cr:.1%}' if isinstance(cr, float) else str(cr)
    f1_str = f'{f1:.2f}' if isinstance(f1, float) else str(f1)
    g_str = f'{g:.2f}x' if isinstance(g, float) else str(g)
    print(f'  {name:<43} {c_str:>8} {cr_str:>8} {f1_str:>8} {g_str:>8}')

print()
print(f'Results: {run_dir}')
print('=' * 70)
"

python3 -u -c "$PYEOF_REAL"

echo ""
echo "All done! Results saved to: $RUN_DIR"
