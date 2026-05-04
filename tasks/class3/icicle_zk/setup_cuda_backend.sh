#!/bin/bash
# Download and install Icicle CUDA backend (BN254 only) for performance comparison.
# The CUDA backend is closed-source but free for R&D use.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/cuda_backend"

if [ -f "${BACKEND_DIR}/cuda/libicicle_backend_cuda_device.so" ] && \
   [ -f "${BACKEND_DIR}/bn254/cuda/libicicle_backend_cuda_curve_bn254.so" ]; then
    echo "CUDA backend already installed in ${BACKEND_DIR}"
    exit 0
fi

# Download full release
URL="https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_4_0_0-ubuntu22-cuda122.tar.gz"
TMP_DIR=$(mktemp -d)
TMP_TAR="${TMP_DIR}/icicle_cuda.tar.gz"

echo "Downloading Icicle CUDA backend (~355MB)..."
if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${TMP_TAR}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -L "${URL}" -o "${TMP_TAR}"
else
    echo "ERROR: wget or curl is required to download the Icicle CUDA backend." >&2
    exit 1
fi

echo "Extracting BN254 backend only..."
mkdir -p "${BACKEND_DIR}/cuda" "${BACKEND_DIR}/bn254/cuda"

# Extract only bn254 + device libraries
tar xzf "${TMP_TAR}" -C "${TMP_DIR}"
cp "${TMP_DIR}/icicle/lib/backend/cuda/libicicle_backend_cuda_device.so" "${BACKEND_DIR}/cuda/"
cp "${TMP_DIR}/icicle/lib/backend/cuda/libicicle_backend_cuda_hash.so" "${BACKEND_DIR}/cuda/" 2>/dev/null || true
cp "${TMP_DIR}/icicle/lib/backend/bn254/cuda/"*.so "${BACKEND_DIR}/bn254/cuda/"

# Cleanup
rm -rf "${TMP_DIR}"

echo "CUDA backend installed in ${BACKEND_DIR}"
echo "Files:"
find "${BACKEND_DIR}" -name "*.so" -exec ls -lh {} \;
