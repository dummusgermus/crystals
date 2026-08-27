#!/bin/bash
#SBATCH --job-name=jara-export-2k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Train + export using prebuilt residual datasets (built/annotated locally).
#
# Prerequisites on the cluster (copy/rsync from local — do NOT rebuild here):
#   adv_datasets/cycle34_residual_dataset.pt   (with particle_ids)
#   planar_pyg_dataset_residual_c14c15.pt      (C14/C15, with particle_ids)
#
# Raw SIMULATIONS trees are still required for crystal export (dumps → full-cell
# PE tables), but dataset *building* is skipped.
#
# Also needed for export:
#   SIMULATIONS/
#   Laves_Planar_Defects/SIMULATIONS/
#   Laves_Screen/SIMULATIONS/
#
# Pipeline:
#   1) Verify residual .pt files + particle_ids
#   2) Train CGCNN + Graph Transformer (default 2000 epochs), save ckpts
#      Delivery fit: ~90% train / ~10% val (grouped), no test; best-by-val ckpt
#   3) Export per-crystal absolute PE + timings (4 combos)
#
# Submit:
#   sbatch run_residual_train_export_2000.slurm
#
# Overrides:
#   sbatch --export=ALL,SKIP_TRAIN=1 run_residual_train_export_2000.slurm   # export only
#   sbatch --export=ALL,SKIP_EXPORT=1 run_residual_train_export_2000.slurm  # train only
#   sbatch --export=ALL,VAL_FRACTION=0.1 run_residual_train_export_2000.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs predictions

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
SEED="${SEED:-42}"
SCORE_LAST_N="${SCORE_LAST_N:-100}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-predictions}"
DEVICE="${DEVICE:-auto}"

POINT_PT="adv_datasets/cycle34_residual_dataset.pt"
PLANAR_PT="planar_pyg_dataset_residual_c14c15.pt"

VAL_FRACTION="${VAL_FRACTION:-0.1}"

echo "============================================================"
echo "[job] train ${EPOCHS}ep + crystal export (prebuilt datasets)"
echo "[job] seed=${SEED} score_last_n=${SCORE_LAST_N} val_fraction=${VAL_FRACTION}"
echo "[job] skip_train=${SKIP_TRAIN} skip_export=${SKIP_EXPORT}"
echo "[job] export_root=${EXPORT_ROOT}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

# ---------------------------------------------------------------------------
# 0) Verify locally prepared residual datasets
# ---------------------------------------------------------------------------
echo
echo "=== [0/3] Checking prebuilt residual datasets ==="
for path in "${POINT_PT}" "${PLANAR_PT}"; do
  if [ ! -f "${path}" ]; then
    echo "[FATAL] Missing ${path}" >&2
    echo "        Build/annotate locally, then rsync to the cluster." >&2
    exit 1
  fi
done

python - <<'PY'
import torch
from pathlib import Path

checks = [
    Path("adv_datasets/cycle34_residual_dataset.pt"),
    Path("planar_pyg_dataset_residual_c14c15.pt"),
]
for path in checks:
    ds = torch.load(path, weights_only=False)
    g = ds[0]
    has_ids = getattr(g, "particle_ids", None) is not None
    print(
        f"[check] {path.name}: n={len(ds)} x={tuple(g.x.shape)} "
        f"particle_ids={'yes' if has_ids else 'NO'} "
        f"n_ids={int(g.particle_ids.numel()) if has_ids else 0}",
        flush=True,
    )
    if not has_ids:
        raise SystemExit(
            f"Missing particle_ids on {path}. "
            "Run annotate_residual_atom_ids.py locally, then re-copy."
        )
PY

# ---------------------------------------------------------------------------
# 1) Train CGCNN + Transformer (residual), save checkpoints
# ---------------------------------------------------------------------------
if [ "${SKIP_TRAIN}" != "1" ]; then
  echo
  echo "=== [1a/3] Training residual CGCNN (${EPOCHS} epochs) ==="
  python -u train_cgcnn_residual_both.py \
    --skip-build \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --metric mae \
    --val-fraction "${VAL_FRACTION}" \
    --save-checkpoints \
    --output-json "cgcnn_residual_both_curves_${EPOCHS}.json"

  echo
  echo "=== [1b/3] Training residual Graph Transformer (${EPOCHS} epochs) ==="
  python -u train_transformer_absolute_residual.py \
    --datasets residual \
    --architecture graph \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --metric mae \
    --score-last-n "${SCORE_LAST_N}" \
    --val-fraction "${VAL_FRACTION}" \
    --save-checkpoints \
    --output-json "transformer_residual_curves_${EPOCHS}.json"

  echo
  echo "=== [1/3] Checkpoint presence ==="
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
else
  echo
  echo "=== [1/3] SKIP train ==="
fi

# ---------------------------------------------------------------------------
# 2) Per-crystal inference export (absolute PE + timings)
# ---------------------------------------------------------------------------
if [ "${SKIP_EXPORT}" != "1" ]; then
  echo
  echo "=== [2/3] Checking dump trees needed for export ==="
  for d in \
    SIMULATIONS \
    Laves_Planar_Defects/SIMULATIONS \
    Laves_Screen/SIMULATIONS
  do
    if [ ! -d "${d}" ]; then
      echo "[FATAL] Missing ${d} (required to restore full-cell PE tables)." >&2
      echo "        Copy/rsync the raw dumps to the cluster, or skip export." >&2
      exit 1
    fi
  done

  echo
  echo "=== [2/3] Exporting crystal PE tables (4 jobs) ==="
  python -u crystal_prediction_export.py \
    --job all \
    --output-root "${EXPORT_ROOT}" \
    --device "${DEVICE}" \
    --write-pt

  echo
  echo "=== [2/3] Export summaries ==="
  for sub in point_cgcnn point_transformer planar_cgcnn planar_transformer; do
    summary="${EXPORT_ROOT}/${sub}/export_summary.json"
    if [ -f "${summary}" ]; then
      python - <<PY
import json
s = json.load(open("${summary}", encoding="utf-8"))
print(
    f"[summary] ${sub}: ok={s.get('n_ok')}/{s.get('n_requested')} "
    f"mean_mae={s.get('mean_mae_abs_eV')} "
    f"mean_t_pred_s={s.get('mean_t_predict_s')}",
    flush=True,
)
PY
    else
      echo "[MISSING] ${summary}" >&2
      exit 1
    fi
  done
else
  echo
  echo "=== [2/3] SKIP export ==="
fi

echo
echo "[done] train ${EPOCHS}ep + crystal export complete (prebuilt datasets)"
