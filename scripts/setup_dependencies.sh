#!/bin/bash
# Verify CUDA-Hercules runtime dependencies for the released benchmark tree.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

check_file() {
    local path="$1"
    local label="$2"
    if [ -f "$path" ]; then
        echo "OK: $label"
    else
        echo "MISSING: $label"
        echo "  Expected: $path"
        return 1
    fi
}

missing=0

check_file "$ROOT_DIR/reference_sources/cutlass/include/cutlass/cutlass.h" \
    "vendored CUTLASS headers" || missing=1
check_file "$ROOT_DIR/reference_sources/flash-attention/csrc/cutlass/include/cute/tensor.hpp" \
    "FlashAttention/CuTe headers" || missing=1
check_file "$ROOT_DIR/reference_sources/ThunderKittens/include/kittens.cuh" \
    "ThunderKittens headers" || missing=1
check_file "$ROOT_DIR/tasks/release_task_set.txt" \
    "released task manifest" || missing=1

if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARNING: nvcc not found in PATH. CUDA toolkit 12.x is required."
else
    echo "OK: CUDA toolkit found: $(nvcc --version | grep release)"
fi

python3 - <<'PY'
import sys

missing = []
for module in ("torch", "yaml", "pydantic", "numpy", "openai", "toml", "scipy", "tiktoken"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)

if missing:
    print("WARNING: missing Python modules:", ", ".join(missing))
    print('Install with: pip install -e ".[dev]"')
else:
    import torch
    print(f"OK: PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

try:
    import ogb  # noqa: F401
except Exception:
    print('NOTE: optional module "ogb" is not installed; install with pip install -e ".[class3-large]" for mgg_gcn data preparation.')
PY

if [ "$missing" -ne 0 ]; then
    echo "One or more vendored reference dependencies are missing."
    exit 1
fi

echo "Setup check complete."
