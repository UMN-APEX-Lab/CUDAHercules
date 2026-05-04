#!/bin/bash
# Run Claude Opus 4.6 evaluation on Hopper tasks (H200 / SM90)
# Usage: bash scripts/run_hopper_eval.sh [GPU_ID]
#
# Prerequisites:
#   pip install -e ".[dev]"
#   gcloud auth application-default login  (for Vertex AI)

set -e

GPU=${1:-0}
MODEL="claude-opus-4-6"
BACKEND="vertex"
REGION="global"
PROJECT="neu-research"
TEMP=0.6
OUTDIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "CUDA-Hercules Hopper Evaluation"
echo "  Model:   $MODEL (via Vertex AI)"
echo "  GPU:     $GPU"
echo "  Output:  $OUTDIR"
echo "============================================"

# Verify GPU
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi -i $GPU --query-gpu=name,compute_cap,memory.total --format=csv,noheader
SM=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '-i', '$GPU', '--query-gpu=compute_cap', '--format=csv,noheader']).decode()
major, minor = out.strip().split('.')
print(int(major)*10 + int(minor))
")
echo "  SM: $SM"
if [ "$SM" -lt 90 ]; then
    echo "ERROR: Hopper tasks require SM >= 90, got SM $SM"
    exit 1
fi

# Verify Vertex AI auth
echo ""
echo "[2/6] Checking Vertex AI..."
python3 -c "
from anthropic import AnthropicVertex
c = AnthropicVertex(region='$REGION', project_id='$PROJECT')
r = c.messages.create(model='$MODEL', max_tokens=16, messages=[{'role':'user','content':'hi'}])
print('  Vertex AI OK:', r.content[0].text[:30])
"

# Run Class 1 Hopper pass@1
echo ""
echo "[3/6] Class 1 Hopper - pass@1..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_pass1.py \
    --model $MODEL --backend $BACKEND \
    --vertex-region $REGION --vertex-project $PROJECT \
    --temperature $TEMP \
    --num-samples 1 \
    --filter "backend=class1_make" --filter "arch=hopper" \
    --run-name opus46_hopper_class1_pass1_${TIMESTAMP} \
    --output $OUTDIR \
    2>&1 | tee ${OUTDIR}/opus46_hopper_class1_pass1_${TIMESTAMP}.log

# Run Class 1 Hopper pass@3
echo ""
echo "[4/6] Class 1 Hopper - pass@3..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_pass1.py \
    --model $MODEL --backend $BACKEND \
    --vertex-region $REGION --vertex-project $PROJECT \
    --temperature $TEMP \
    --num-samples 3 \
    --filter "backend=class1_make" --filter "arch=hopper" \
    --run-name opus46_hopper_class1_pass3_${TIMESTAMP} \
    --output $OUTDIR \
    2>&1 | tee ${OUTDIR}/opus46_hopper_class1_pass3_${TIMESTAMP}.log

# Run Class 2 Hopper pass@1
echo ""
echo "[5/6] Class 2 Hopper - pass@1..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_pass1.py \
    --model $MODEL --backend $BACKEND \
    --vertex-region $REGION --vertex-project $PROJECT \
    --temperature $TEMP \
    --num-samples 1 \
    --filter "backend=class2_defpy" --filter "arch=hopper" \
    --run-name opus46_hopper_class2_pass1_${TIMESTAMP} \
    --output $OUTDIR \
    2>&1 | tee ${OUTDIR}/opus46_hopper_class2_pass1_${TIMESTAMP}.log

# Run Class 2 Hopper pass@3
echo ""
echo "[6/6] Class 2 Hopper - pass@3..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_pass1.py \
    --model $MODEL --backend $BACKEND \
    --vertex-region $REGION --vertex-project $PROJECT \
    --temperature $TEMP \
    --num-samples 3 \
    --filter "backend=class2_defpy" --filter "arch=hopper" \
    --run-name opus46_hopper_class2_pass3_${TIMESTAMP} \
    --output $OUTDIR \
    2>&1 | tee ${OUTDIR}/opus46_hopper_class2_pass3_${TIMESTAMP}.log

echo ""
echo "============================================"
echo "All done! Results in: $OUTDIR/opus46_hopper_*_${TIMESTAMP}"
echo "============================================"
