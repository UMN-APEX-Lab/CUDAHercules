#!/bin/bash
# Fetch the task data (prompts.json, quality_reference.json) into $HERCULES_DATA_DIR/long_context_ttft
# and the model weights into the Hugging Face cache ($HF_HOME).
set -euo pipefail
: "${HERCULES_DATA_DIR:?set HERCULES_DATA_DIR}"
DEST="$HERCULES_DATA_DIR/long_context_ttft"
mkdir -p "$DEST"
BASE="${HERCULES_DATA_URL:-https://anonymous-hf.com/api/a/cs6btyea9aef}"   # <base>/resolve/<path>
for f in prompts.json quality_reference.json; do
    curl -fsSL -o "$DEST/$f" "$BASE/resolve/long_context_ttft/$f"
done
hf download Qwen/Qwen3.8-27B
ls -la "$DEST"
