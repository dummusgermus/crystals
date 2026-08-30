#!/usr/bin/env bash
# Local step: merge cluster inference outputs + LAMMPS dumps -> predictions_new/
#
# Requires on this machine:
#   predictions_new_inference/   (from cluster)
#   predictions_new/delivery_split_indices.json
#   SIMULATIONS/ + Laves_* dump trees
#
# Local restore (needs dump trees; overwrites OUT_ROOT):
#   bash run_predictions_new_restore.sh

set -euo pipefail

cd "$(dirname "$0")"

SPLIT_JSON="${SPLIT_JSON:-predictions_new/delivery_split_indices.json}"
INFER_ROOT="${INFER_ROOT:-predictions_new_inference}"
OUT_ROOT="${OUT_ROOT:-predictions_new}"

if [ ! -f "${SPLIT_JSON}" ]; then
  echo "[FATAL] Missing ${SPLIT_JSON} (run cluster training first)." >&2
  exit 1
fi
if [ ! -d "${INFER_ROOT}/point_cgcnn" ]; then
  echo "[FATAL] Missing ${INFER_ROOT}/ (pull cluster inference outputs)." >&2
  exit 1
fi

python -u crystal_prediction_export.py \
  --mode restore-merged \
  --job all \
  --job-profile global_v2 \
  --inference-root "${INFER_ROOT}" \
  --output-root "${OUT_ROOT}" \
  --split-json "${SPLIT_JSON}"

echo "[done] Merged CSVs -> ${OUT_ROOT}/point/ and ${OUT_ROOT}/planar/"
