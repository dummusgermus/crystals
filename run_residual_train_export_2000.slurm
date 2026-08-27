#!/bin/bash
#SBATCH --job-name=jara-export-2k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Full residual pipeline (C14/C15 planar labels):
#   1) Rebuild point + planar residual .pt graphs WITH particle_ids
#   2) Train CGCNN + Graph Transformer for EPOCHS (default 2000), save ckpts
#   3) Export per-crystal absolute PE tables + timings for all 4 combos
#
# particle_ids / orig_indices are metadata only — models never read them as
# features (forward uses data.x / data.edge_attr only).
#
# Outputs:
#   adv_datasets/cycle34_residual_dataset.pt
#   planar_pyg_dataset_residual_c14c15.pt
#   cgcnn_defect_residual_model.pt
#   cgcnn_planar_residual_c14c15_model.pt
#   transformer_graph_defect_residual_model.pt
#   transformer_graph_planar_residual_c14c15_model.pt
#   predictions/{point,planar}_{cgcnn,transformer}/...
#
# Required data:
#   SIMULATIONS/
#   Laves_Planar_Defects/SIMULATIONS/
#   Laves_Screen/SIMULATIONS/
#   laves_defect_atoms_c14c15.json
#
# Submit:
#   sbatch run_residual_train_export_2000.slurm
#
# Overrides:
#   sbatch --export=ALL,EPOCHS=2000,SKIP_BUILD=0 run_residual_train_export_2000.slurm
#   sbatch --export=ALL,SKIP_BUILD=1,SKIP_TRAIN=1 run_residual_train_export_2000.slurm  # export only
#   sbatch --export=ALL,SKIP_EXPORT=1 run_residual_train_export_2000.slurm              # train only

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

# OVITO OpenGL shim (needed when rebuilding datasets on headless nodes)
SHIM_DIR="${SLURM_SUBMIT_DIR}/.lib_shim"
mkdir -p "${SHIM_DIR}"
if [ ! -e "${SHIM_DIR}/libOpenGL.so.0" ]; then
  SYS_LIBGL="$(ldconfig -p | awk '/libGL\.so\.1\b/ {print $NF; exit}')"
  if [ -z "${SYS_LIBGL}" ] || [ ! -e "${SYS_LIBGL}" ]; then
    SYS_LIBGL="/lib/x86_64-linux-gnu/libGL.so.1"
  fi
  if [ -e "${SYS_LIBGL}" ]; then
    ln -sf "${SYS_LIBGL}" "${SHIM_DIR}/libOpenGL.so.0"
    echo "[env] libOpenGL.so.0 -> ${SYS_LIBGL}"
  else
    echo "[env] WARNING: no libGL.so.1 found; OVITO import may fail during build." >&2
  fi
fi
export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH:-}"

EPOCHS="${EPOCHS:-2000}"
SEED="${SEED:-42}"
SCORE_LAST_N="${SCORE_LAST_N:-100}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-predictions}"
DEVICE="${DEVICE:-auto}"

echo "============================================================"
echo "[job] residual rebuild + train ${EPOCHS}ep + crystal export"
echo "[job] seed=${SEED} score_last_n=${SCORE_LAST_N}"
echo "[job] skip_build=${SKIP_BUILD} skip_train=${SKIP_TRAIN} skip_export=${SKIP_EXPORT}"
echo "[job] export_root=${EXPORT_ROOT}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

# ---------------------------------------------------------------------------
# 1) Rebuild residual datasets (point + planar C14/C15) with particle_ids
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD}" != "1" ]; then
  echo
  echo "=== [1/3] Building residual datasets (force) ==="
  python -u build_residual_datasets.py --force

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
        raise SystemExit(f"Missing particle_ids on rebuilt dataset: {path}")
PY
else
  echo
  echo "=== [1/3] SKIP build ==="
fi

# ---------------------------------------------------------------------------
# 2) Train CGCNN + Transformer (residual), save checkpoints
# ---------------------------------------------------------------------------
if [ "${SKIP_TRAIN}" != "1" ]; then
  echo
  echo "=== [2a/3] Training residual CGCNN (${EPOCHS} epochs) ==="
  python -u train_cgcnn_residual_both.py \
    --skip-build \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --metric mae \
    --save-checkpoints \
    --output-json "cgcnn_residual_both_curves_${EPOCHS}.json"

  echo
  echo "=== [2b/3] Training residual Graph Transformer (${EPOCHS} epochs) ==="
  python -u train_transformer_absolute_residual.py \
    --datasets residual \
    --architecture graph \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --metric mae \
    --score-last-n "${SCORE_LAST_N}" \
    --save-checkpoints \
    --output-json "transformer_residual_curves_${EPOCHS}.json"

  echo
  echo "=== [2/3] Checkpoint presence ==="
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
  echo "=== [2/3] SKIP train ==="
fi

# ---------------------------------------------------------------------------
# 3) Per-crystal inference export (absolute PE + timings)
# ---------------------------------------------------------------------------
if [ "${SKIP_EXPORT}" != "1" ]; then
  echo
  echo "=== [3/3] Exporting crystal PE tables (4 jobs) ==="
  python -u crystal_prediction_export.py \
    --job all \
    --output-root "${EXPORT_ROOT}" \
    --device "${DEVICE}" \
    --write-pt

  echo
  echo "=== [3/3] Export summaries ==="
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
  echo "=== [3/3] SKIP export ==="
fi

echo
echo "[done] residual rebuild + ${EPOCHS}-epoch train + crystal export complete"
