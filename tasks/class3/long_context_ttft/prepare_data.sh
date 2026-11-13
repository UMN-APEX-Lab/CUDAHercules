#!/bin/bash
# Fetch the task data (prompts.json, quality_reference.json) into $HERCULES_DATA_DIR/long_context_ttft
# and the model weights into the Hugging Face cache ($HF_HOME).
set -euo pipefail
: "${HERCULES_DATA_DIR:?set HERCULES_DATA_DIR}"
DEST="$HERCULES_DATA_DIR/long_context_ttft"
mkdir -p "$DEST"
hf download --repo-type dataset "${HERCULES_HF_DATASET:?set HERCULES_HF_DATASET to the task data repo}" \
    --include "long_context_ttft/*" --local-dir "$HERCULES_DATA_DIR"
hf download Qwen/Qwen3.8-27B
ls -la "$DEST"
