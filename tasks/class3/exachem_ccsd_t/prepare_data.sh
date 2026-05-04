#!/bin/bash
# ExaChem CCSD(T) uses deterministic tensor fixtures embedded in the harness.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "exachem_ccsd_t: no external input dataset is required."
echo "The benchmark driver creates deterministic CCSD(T) tensor cases at runtime."
echo "Task directory: ${SCRIPT_DIR}"
