#!/bin/bash
# Download ogbn-papers100M and convert to MGG binary CSR format
# Dataset: ~111M nodes, ~1.6B edges
# Output: {beg_pos,csr,weight}.bin files for MGG graph loader
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${MGG_DATA_DIR:-${SCRIPT_DIR}/data}}"

# Check if already converted
if [ -f "$DATA_DIR/bin/paper100M_beg_pos.bin" ] && [ -f "$DATA_DIR/bin/paper100M_csr.bin" ]; then
    echo "Dataset already prepared: $DATA_DIR/bin/paper100M_*.bin"
    exit 0
fi

python3 "$SCRIPT_DIR/convert_ogb.py" "$DATA_DIR"
echo "=== Data preparation complete ==="
