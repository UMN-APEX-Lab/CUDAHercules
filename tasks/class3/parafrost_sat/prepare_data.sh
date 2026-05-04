#!/bin/bash
# Thin shim around prepare_data.py. The real logic — Zenodo download, MD5
# verify, selective extraction + LZMA decompression of the 10-instance
# subset — lives in prepare_data.py so it stays portable across systems.
#
# Backward-compatible interface for older callers that pass `data_dir` as
# the first positional arg:
#   bash prepare_data.sh [data_dir]
# That argument is ignored — the data dir is always `<task_dir>/data`.
# Override the Zenodo zip cache location via PARAFROST_DATA_ROOT.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/prepare_data.py"
python3 "$SCRIPT_DIR/prepare_blackbox_inputs.py" >/dev/null
echo "parafrost_sat: data and standardized inputs are ready."
