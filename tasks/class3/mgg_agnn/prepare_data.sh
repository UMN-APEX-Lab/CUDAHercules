#!/bin/bash
# AGNN uses the same amazon0505 graph as tcgnn_gcn
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../tcgnn_gcn/data"
GRAPH_FILE="$DATA_DIR/amazon0505.npz"
if [ -f "$GRAPH_FILE" ]; then
    echo "Dataset already available: $GRAPH_FILE"
    exit 0
fi
echo "Running tcgnn_gcn data preparation..."
bash "$SCRIPT_DIR/../tcgnn_gcn/prepare_data.sh"
