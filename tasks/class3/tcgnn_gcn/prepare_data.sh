#!/bin/bash
# Download graph datasets for the multi-graph TC-GNN GCN benchmark
# Source: TC-GNN ATC'23 (https://github.com/YukeWang96/TC-GNN_ATC23)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/data}"
GRAPHS=("amazon0505" "artist" "soc-BlogCatalog" "amazon0601")

all_present=1
for graph in "${GRAPHS[@]}"; do
    if [ ! -f "$DATA_DIR/$graph.npz" ]; then
        all_present=0
        break
    fi
done

if [ "$all_present" -eq 1 ]; then
    echo "Datasets already exist in: $DATA_DIR"
    exit 0
fi

mkdir -p "$DATA_DIR"

TARBALL="$DATA_DIR/tcgnn-ae-graphs.tar.gz"
URL="https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz"

echo "Downloading TC-GNN graph datasets (~50MB)..."
if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$TARBALL" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$TARBALL"
else
    echo "ERROR: wget or curl is required to download TC-GNN datasets." >&2
    exit 1
fi

echo "Extracting selected graphs..."
for graph in "${GRAPHS[@]}"; do
    tar -xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1 "tcgnn-ae-graphs/$graph.npz"
done

rm -f "$TARBALL"
echo "Done: ${GRAPHS[*]}"
