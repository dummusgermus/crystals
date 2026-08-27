#!/bin/bash
#SBATCH --job-name=crystal-infer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Cluster inference-only: GPU ΔPE preds + per-crystal timings (no dumps).
#
# Prerequisites (already needed for training):
#   adv_datasets/cycle34_residual_dataset.pt
#   planar_pyg_dataset_residual_c14c15.pt
#   cgcnn_defect_residual_model.pt
#   cgcnn_planar_residual_c14c15_model.pt
#   transformer_graph_defect_residual_model.pt
#   transformer_graph_planar_residual_c14c15_model.pt
#
# Does NOT need SIMULATIONS / Laves_* dump trees.
#
# After the job finishes, pull the tree (include ``*_graph_pred.npz`` / legacy
# ``*_graph_pred.pt`` — do not use git for the preds; ``*.pt`` is gitignored):
#   rsync -avz --include='*/' --include='*_graph_pred.*' --include='*_timing.json' \
#     --include='*summary.json' --exclude='*' \
#     host:/path/code/predictions_inference/ ./predictions_inference/
#
# Then restore locally (merged CGCNN+Transformer CSVs):
#   python crystal_prediction_export.py --mode restore-merged --job all \
#       --inference-root predictions_inference --output-root predictions
#
# Submit:
#   sbatch run_crystal_inference_timing.slurm
#   sbatch --export=ALL,LIMIT=5 run_crystal_inference_timing.slurm   # smoke test

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

DEVICE="${DEVICE:-auto}"
INFER_ROOT="${INFER_ROOT:-predictions_inference}"
LIMIT="${LIMIT:-}"

echo "============================================================"
echo "[job] crystal inference timings (no dumps)"
echo "[job] device=${DEVICE} infer_root=${INFER_ROOT}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

for ckpt in \
  cgcnn_defect_residual_model.pt \
  cgcnn_planar_residual_c14c15_model.pt \
  transformer_graph_defect_residual_model.pt \
  transformer_graph_planar_residual_c14c15_model.pt
do
  if [ -f "${ckpt}" ]; then
    echo "[ok] ${ckpt} ($(du -h "${ckpt}" | awk '{print $1}'))"
  else
    echo "[MISSING] ${ckpt}" >&2
    exit 1
  fi
done

for ds in \
  adv_datasets/cycle34_residual_dataset.pt \
  planar_pyg_dataset_residual_c14c15.pt
do
  if [ -f "${ds}" ]; then
    echo "[ok] ${ds} ($(du -h "${ds}" | awk '{print $1}'))"
  else
    echo "[MISSING] ${ds}" >&2
    exit 1
  fi
done

EXTRA=()
if [ -n "${LIMIT}" ]; then
  EXTRA+=(--limit "${LIMIT}")
fi

echo
echo "=== Inference (no dumps) ==="
python -u crystal_prediction_export.py \
  --mode inference \
  --job all \
  --output-root "${INFER_ROOT}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"

echo
echo "=== Inference summaries ==="
for sub in point_cgcnn point_transformer planar_cgcnn planar_transformer; do
  summary="${INFER_ROOT}/${sub}/inference_summary.json"
  if [ -f "${summary}" ]; then
    python - <<PY
import json
s = json.load(open("${summary}", encoding="utf-8"))
print(
    f"[summary] ${sub}: ok={s.get('n_ok')}/{s.get('n_requested')} "
    f"mean_mae_resid={s.get('mean_mae_residual_eV')} "
    f"mean_t_pred_s={s.get('mean_t_predict_s')}",
    flush=True,
)
PY
  else
    echo "[MISSING] ${summary}" >&2
    exit 1
  fi
done

echo
echo "=== Done. Pull ${INFER_ROOT}/ then restore locally with dumps. ==="
