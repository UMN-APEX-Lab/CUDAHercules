#!/bin/bash
# Prepare Icicle ZK reference data and CUDA backend libraries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF_DIR="${SCRIPT_DIR}/ref_data"
BACKEND_DIR="${SCRIPT_DIR}/cuda_backend"
cd "${SCRIPT_DIR}"

REF_FILES=(
  ntt_16.bin
  ntt_18.bin
  ntt_20.bin
  ntt_22.bin
  ntt_24.bin
  msm_points_14.bin
  msm_points_16.bin
  msm_points_18.bin
  msm_points_20.bin
  msm_points_22.bin
  msm_scalars_14.bin
  msm_scalars_16.bin
  msm_scalars_18.bin
  msm_scalars_20.bin
  msm_scalars_22.bin
)

BACKEND_FILES=(
  cuda/libicicle_backend_cuda_device.so
  bn254/cuda/libicicle_backend_cuda_curve_bn254.so
  bn254/cuda/libicicle_backend_cuda_field_bn254.so
)

file_ok() {
  local path="$1"
  [ -s "$path" ] && [ "$(stat -c '%s' "$path")" -gt 1024 ]
}

missing_ref=0
for f in "${REF_FILES[@]}"; do
  if ! file_ok "${REF_DIR}/${f}"; then
    missing_ref=1
    break
  fi
done

if [ "${missing_ref}" -eq 1 ]; then
  echo "Icicle ZK: reference data missing; generating CPU reference fixtures."
  python3 - <<'PY'
import os
import run

run.build_icicle()
run.build_bench()
os.makedirs(run.REF_DIR, exist_ok=True)
result = run.run_bench("CPU", save_ref=True)
if not result["passed"]:
    raise SystemExit("CPU reference fixture generation failed")
PY
else
  echo "Icicle ZK: reference data already present."
fi

missing_backend=0
for f in "${BACKEND_FILES[@]}"; do
  if ! file_ok "${BACKEND_DIR}/${f}"; then
    missing_backend=1
    break
  fi
done

if [ "${missing_backend}" -eq 1 ]; then
  echo "Icicle ZK: CUDA backend libraries missing; downloading Icicle BN254 CUDA backend."
  bash "${SCRIPT_DIR}/setup_cuda_backend.sh"
else
  echo "Icicle ZK: CUDA backend libraries already present."
fi

echo "Icicle ZK: data preparation complete."
