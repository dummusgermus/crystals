#!/bin/bash
#SBATCH --job-name=jara-pred-new
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# predictions_new pipeline (cluster): train 4 global-v2 models + GPU inference.
#
# Same delivery split as original predictions/ (within-group 90/10, seed 42).
# Only change vs original delivery: global-loss v2 + checkpoint by val R_tot.
#
# Prerequisites on cluster (rsync from local):
#   adv_datasets/cycle34_residual_dataset.pt          (+ particle_ids)
#   planar_pyg_dataset_residual_c14c15.pt           (+ particle_ids)
#   adv_datasets/cycle34_residual_totals_dataset.pt
#   planar_pyg_dataset_residual_c14c15_totals.pt
#
# Cluster outputs (pull to local):
#   predictions_new/delivery_split_indices.json
#   *_global_v2_model.pt  (4 checkpoints)
#   predictions_new_inference/  (graph preds + timings)
#   delivery_global_v2_curves.json
#
# Local restore (needs dump trees):
#   bash run_predictions_new_restore.sh
#
# Submit:
#   sbatch run_delivery_global_v2.slurm
#
# Train only:
#   sbatch --export=ALL,SKIP_INFER=1 run_delivery_global_v2.slurm
#
# Resume interrupted training:
#   sbatch --export=ALL,SKIP_BUILD=1,RESUME=1 run_delivery_global_v2.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs predictions_new

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/torch-cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

EPOCHS="${EPOCHS:-2000}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RESUME="${RESUME:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
INFER_ROOT="${INFER_ROOT:-predictions_new_inference}"
SPLIT_JSON="${SPLIT_JSON:-predictions_new/delivery_split_indices.json}"
DEVICE="${DEVICE:-auto}"

echo "============================================================"
echo "[job] predictions_new: global-v2 delivery train + inference"
echo "[job] epochs=${EPOCHS} split_json=${SPLIT_JSON}"
echo "[job] skip_build=${SKIP_BUILD} resume=${RESUME}"
echo "[job] skip_train=${SKIP_TRAIN} skip_infer=${SKIP_INFER}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

echo
echo "=== Checking datasets ==="
for path in \
  adv_datasets/cycle34_residual_dataset.pt \
  planar_pyg_dataset_residual_c14c15.pt \
  adv_datasets/cycle34_residual_totals_dataset.pt \
  planar_pyg_dataset_residual_c14c15_totals.pt
do
  if [ ! -f "${path}" ]; then
    echo "[FATAL] Missing ${path}" >&2
    exit 1
  fi
  echo "[ok] ${path}"
done

python - <<'PY'
import torch
from pathlib import Path
for path in [
    Path("adv_datasets/cycle34_residual_dataset.pt"),
    Path("planar_pyg_dataset_residual_c14c15.pt"),
]:
    ds = torch.load(path, weights_only=False)
    g = ds[0]
    has_ids = getattr(g, "particle_ids", None) is not None
    print(f"[check] {path.name}: n={len(ds)} particle_ids={'yes' if has_ids else 'NO'}", flush=True)
    if not has_ids:
        raise SystemExit(f"Missing particle_ids on {path}")
PY

TRAIN_CMD=(
  python -u train_delivery_global_v2.py
  --epochs "${EPOCHS}"
  --split-json "${SPLIT_JSON}"
  --model all
)
if [ "${SKIP_BUILD}" = "1" ]; then
  TRAIN_CMD+=(--skip-build)
fi
if [ "${RESUME}" = "1" ]; then
  TRAIN_CMD+=(--resume)
fi

if [ "${SKIP_TRAIN}" != "1" ]; then
  echo
  echo "=== Training 4 models (CGCNN + Transformer, global v2) ==="
  "${TRAIN_CMD[@]}"
else
  echo
  echo "=== SKIP train ==="
fi

echo
echo "=== Checkpoint presence ==="
for ckpt in \
  cgcnn_defect_residual_global_v2_model.pt \
  cgcnn_planar_residual_global_v2_model.pt \
  transformer_graph_defect_residual_global_v2_model.pt \
  transformer_graph_planar_residual_global_v2_model.pt
do
  if [ -f "${ckpt}" ]; then
    echo "[ok] ${ckpt} ($(du -h "${ckpt}" | awk '{print $1}'))"
  else
    echo "[MISSING] ${ckpt}" >&2
    exit 1
  fi
done

if [ "${SKIP_INFER}" != "1" ]; then
  echo
  echo "=== GPU inference (no dumps) ==="
  python -u crystal_prediction_export.py \
    --mode inference \
    --job all \
    --job-profile global_v2 \
    --output-root "${INFER_ROOT}" \
    --device "${DEVICE}"

  for sub in point_cgcnn point_transformer planar_cgcnn planar_transformer; do
    summary="${INFER_ROOT}/${sub}/inference_summary.json"
    if [ ! -f "${summary}" ]; then
      echo "[MISSING] ${summary}" >&2
      exit 1
    fi
    echo "[ok] ${summary}"
  done
else
  echo
  echo "=== SKIP inference ==="
fi

echo
echo "[done] Pull ${SPLIT_JSON}, checkpoints, and ${INFER_ROOT}/"
echo "[done] Then run: bash run_predictions_new_restore.sh"
