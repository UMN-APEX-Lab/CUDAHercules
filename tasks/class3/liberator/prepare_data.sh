#!/bin/bash
# Download and convert the Friendster graph dataset for Liberator benchmarks.
#
# Usage:
#   bash prepare_data.sh [DATA_DIR]
#
# DATA_DIR defaults to ./data. The script:
#   1. Downloads com-friendster.ungraph.txt.gz from SNAP (~9GB compressed)
#   2. Decompresses to text edge list (~30GB)
#   3. Strips comment lines (required by converter)
#   4. Converts to Liberator binary formats: bcsr, bcsc, bwcsr
#
# Requirements:
#   - ~90GB free disk space (compressed + text + binary formats)
#   - wget or curl
#   - g++ (to compile the converter)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${LIBERATOR_DATA:-${SCRIPT_DIR}/data}}"
CONVERTER_DIR="${SCRIPT_DIR}/converter"

URL="https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
GZ_FILE="${DATA_DIR}/com-friendster.ungraph.txt.gz"
TXT_FILE="${DATA_DIR}/friendster.txt"
TOB_FILE="${DATA_DIR}/friendster.toB"
BCSR_FILE="${DATA_DIR}/friendster.bcsr"
BCSC_FILE="${DATA_DIR}/friendster.bcsc"
BWCSR_FILE="${DATA_DIR}/friendster.bwcsr"

mkdir -p "${DATA_DIR}"

# ── Step 0: Build converter ──────────────────────────────────────────
CONVERTER="${CONVERTER_DIR}/converter"
if [ ! -x "${CONVERTER}" ]; then
    echo "Building converter..."
    g++ -O2 -o "${CONVERTER}" "${CONVERTER_DIR}/converter.cpp"
fi

# ── Step 1: Download ─────────────────────────────────────────────────
if [ -f "${BCSR_FILE}" ] && [ -f "${BCSC_FILE}" ] && [ -f "${BWCSR_FILE}" ]; then
    echo "All binary formats already exist in ${DATA_DIR}. Skipping download."
    echo "  ${BCSR_FILE}"
    echo "  ${BCSC_FILE}"
    echo "  ${BWCSR_FILE}"
    exit 0
fi

if [ ! -f "${GZ_FILE}" ] && [ ! -f "${TXT_FILE}" ] && [ ! -f "${TOB_FILE}" ]; then
    echo "Downloading Friendster dataset (~9GB)..."
    if command -v wget &>/dev/null; then
        wget -c -O "${GZ_FILE}" "${URL}"
    elif command -v curl &>/dev/null; then
        curl -L -C - -o "${GZ_FILE}" "${URL}"
    else
        echo "Error: neither wget nor curl found." >&2
        exit 1
    fi
fi

# ── Step 2: Decompress ───────────────────────────────────────────────
if [ ! -f "${TXT_FILE}" ] && [ ! -f "${TOB_FILE}" ]; then
    echo "Decompressing (~30GB uncompressed)..."
    gunzip -k "${GZ_FILE}"
    # The decompressed file is com-friendster.ungraph.txt
    DECOMPRESSED="${DATA_DIR}/com-friendster.ungraph.txt"
    if [ -f "${DECOMPRESSED}" ]; then
        # Strip comment lines (lines starting with #)
        echo "Stripping comment lines..."
        grep -v '^#' "${DECOMPRESSED}" > "${TXT_FILE}"
        rm -f "${DECOMPRESSED}"
    fi
fi

# ── Step 3: Convert text → binary intermediate ───────────────────────
if [ ! -f "${TOB_FILE}" ]; then
    echo "Converting text to binary intermediate format..."
    cd "${DATA_DIR}"
    "${CONVERTER}" txt2byte friendster.txt
    cd -
fi

# ── Step 4: Convert to Liberator formats ─────────────────────────────
if [ ! -f "${BCSR_FILE}" ]; then
    echo "Converting to bcsr (for BFS/CC)..."
    cd "${DATA_DIR}"
    "${CONVERTER}" byte2bcsr friendster.toB
    cd -
fi

if [ ! -f "${BCSC_FILE}" ]; then
    echo "Converting to bcsc (for PageRank)..."
    cd "${DATA_DIR}"
    "${CONVERTER}" byte2bcsc friendster.toB
    cd -
fi

if [ ! -f "${BWCSR_FILE}" ]; then
    echo "Converting to bwcsr (for SSSP)..."
    cd "${DATA_DIR}"
    "${CONVERTER}" byte2bwcsr friendster.toB
    cd -
fi

echo ""
echo "Done! Dataset files:"
ls -lh "${BCSR_FILE}" "${BCSC_FILE}" "${BWCSR_FILE}"
echo ""
echo "Set LIBERATOR_DATA=${DATA_DIR} when running the benchmark."
