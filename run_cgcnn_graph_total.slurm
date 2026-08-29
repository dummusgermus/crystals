#!/bin/bash
#SBATCH --job-name=jara-gtot
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Graph-level CGCNN: predict full-cell delta_total_eV directly (no per-atom head).
#
# Split: grouped 70/15/15 (train/val/test by defect group).
# Default: 300 epochs, point + planar.
#
# Prerequisites (rsync totals datasets OR let job build from dumps):
#   adv_datasets/cycle34_residual_totals_dataset.pt
#   planar_pyg_dataset_residual_c14c15_totals.pt
#
# Outputs:
#   adv_datasets/cycle34_graph_total_target_dataset.pt
#   planar_pyg_dataset_graph_total_target_c14c15.pt
#   cgcnn_point_graph_total_model.pt
#   cgcnn_planar_graph_total_model.pt
#   cgcnn_graph_total_curves.json
#   cgcnn_graph_total_summary.json
#   cgcnn_graph_total_epoch_curves.png
#   cgcnn_graph_total_test_scatter.png
#
# Submit:
#   sbatch run_cgcnn_graph_total.slurm
#
# Overrides:
#   sbatch --export=ALL,EPOCHS=500 run_cgcnn_graph_total.slurm
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_graph_total.slurm
#   sbatch --export=ALL,DOMAIN=planar run_cgcnn_graph_total.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

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

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DOMAIN="${DOMAIN:-both}"

echo "============================================================"
echo "[job] graph-level total residual CGCNN"
echo "[job] epochs=${EPOCHS} seed=${SEED} domain=${DOMAIN}"
echo "[job] skip_build=${SKIP_BUILD}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

TRAIN_CMD=(
  python -u train_cgcnn_graph_total.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --domain "${DOMAIN}"
  --plot
)
if [ "${SKIP_BUILD}" = "1" ]; then
  TRAIN_CMD+=(--skip-build)
fi

echo
echo "=== Training graph-total CGCNN ==="
"${TRAIN_CMD[@]}"

echo
echo "=== Outputs ==="
for f in \
  cgcnn_graph_total_curves.json \
  cgcnn_graph_total_summary.json \
  cgcnn_graph_total_epoch_curves.png \
  cgcnn_graph_total_test_scatter.png
do
  if [ -f "${f}" ]; then
    echo "[ok] ${f}"
  else
    echo "[MISSING] ${f}" >&2
  fi
done
for domain in point planar; do
  if [ "${DOMAIN}" != "both" ] && [ "${DOMAIN}" != "${domain}" ]; then
    continue
  fi
  ckpt="cgcnn_${domain}_graph_total_model.pt"
  if [ -f "${ckpt}" ]; then
    echo "[ok] ${ckpt}"
  else
    echo "[MISSING] ${ckpt}" >&2
    exit 1
  fi
done

echo
echo "[done] Pull JSON, PNGs, and checkpoints for analysis."
