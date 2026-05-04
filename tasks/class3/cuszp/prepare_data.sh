#!/bin/bash
# cuSZp uses deterministic synthetic arrays generated inside the benchmark.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "cuSZp: no external input dataset is required."
echo "The harness generates deterministic compression inputs at runtime."
echo "Task directory: ${SCRIPT_DIR}"
