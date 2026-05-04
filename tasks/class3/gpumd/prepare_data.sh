#!/bin/bash
# Prepare GPUMD structure files, reference force fixtures, and standardized inputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
REF_DIR="${DATA_DIR}/ref_forces"

XYZ_FILES=(
  Ar_256000.xyz
  Ar_500000.xyz
  Si_27000.xyz
  Si_64000.xyz
  NaCl_8000.xyz
  NaCl_32768.xyz
)

REF_FILES=(
  lj_Ar_256000.bin
  lj_Ar_500000.bin
  tersoff_Si_27000.bin
  tersoff_Si_64000.bin
  coulomb_NaCl_8000.bin
  coulomb_NaCl_32768.bin
)

echo "GPUMD: preparing structure files..."
python3 "${SCRIPT_DIR}/generate_data.py"

missing_ref=0
for f in "${REF_FILES[@]}"; do
  if [ ! -s "${REF_DIR}/${f}" ]; then
    missing_ref=1
    break
  fi
done

if [ "${missing_ref}" -eq 1 ]; then
  echo "GPUMD: reference force files missing; building md_bench to generate them."
  make -C "${SCRIPT_DIR}/src" md_bench
  "${SCRIPT_DIR}/src/md_bench" "${DATA_DIR}" --save-ref
fi

for f in "${XYZ_FILES[@]}"; do
  test -s "${DATA_DIR}/${f}" || { echo "missing ${DATA_DIR}/${f}" >&2; exit 1; }
done
for f in "${REF_FILES[@]}"; do
  test -s "${REF_DIR}/${f}" || { echo "missing ${REF_DIR}/${f}" >&2; exit 1; }
done

echo "GPUMD: preparing standardized black-box workloads..."
python3 "${SCRIPT_DIR}/prepare_blackbox_inputs.py"
