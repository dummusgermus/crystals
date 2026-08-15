#!/bin/bash
#SBATCH --job-name=jara-pt-vs-pl-res
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Point vs planar residual experiment (complete data):
#   1) Force-rebuild residual-ΔPE datasets (point + planar C14/C15)
#   2) Train best gated CGCNN for 1000 epochs on both
#   3) Train best Graph Transformer for 1000 epochs on both
#
# Required raw data in the submit directory (for --build):
#   SIMULATIONS/
#   Laves_Planar_Defects/SIMULATIONS/
#   Laves_Screen/SIMULATIONS/
#   laves_defect_atoms_c14c15.json
#
# Or pre-build locally and skip rebuild on the cluster:
#   python build_residual_datasets.py --force
#   sbatch --export=ALL,SKIP_BUILD=1 run_point_vs_planar_residual_1000.slurm
#
# Submit:
#   sbatch run_point_vs_planar_residual_1000.slurm
#
# Only CGCNN / only transformer:
#   sbatch --export=ALL,MODELS=cgcnn run_point_vs_planar_residual_1000.slurm
#   sbatch --export=ALL,MODELS=transformer run_point_vs_planar_residual_1000.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/torch-cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ---------------------------------------------------------------------------
# OVITO's pip wheel needs libOpenGL.so.0 (libglvnd). Some compute nodes only
# ship the pre-libglvnd monolithic libGL.so.1. Shim it for dataset rebuilds.
# ---------------------------------------------------------------------------
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

# Overridable defaults
EPOCHS="${EPOCHS:-1000}"
SCORE_LAST_N="${SCORE_LAST_N:-50}"
SEED="${SEED:-42}"
MODELS="${MODELS:-both}"          # both | cgcnn | transformer
SKIP_BUILD="${SKIP_BUILD:-0}"      # 1 = use existing residual .pt files
SAVE_CKPT="${SAVE_CKPT:-0}"
TF_CONFIG_JSON="${TF_CONFIG_JSON:-}"  # optional: transformer_hparam_sweep.json

CGCNN_JSON="${CGCNN_JSON:-cgcnn_point_vs_planar_residual_1000_curves.json}"
TRANSFORMER_JSON="${TRANSFORMER_JSON:-transformer_point_vs_planar_residual_1000_curves.json}"
SUMMARY_JSON="${SUMMARY_JSON:-point_vs_planar_residual_1000_summary.json}"

echo "[job] epochs=${EPOCHS} score_last_n=${SCORE_LAST_N} seed=${SEED} models=${MODELS}"
echo "[job] skip_build=${SKIP_BUILD} save_ckpt=${SAVE_CKPT}"
echo "[job] device check:"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u run_point_vs_planar_residual_1000.py
  --epochs "${EPOCHS}"
  --score-last-n "${SCORE_LAST_N}"
  --seed "${SEED}"
  --models "${MODELS}"
  --cgcnn-json "${CGCNN_JSON}"
  --transformer-json "${TRANSFORMER_JSON}"
  --summary-json "${SUMMARY_JSON}"
)

if [ "${SKIP_BUILD}" = "1" ]; then
  CMD+=(--skip-build)
fi

if [ -n "${TF_CONFIG_JSON}" ]; then
  if [ ! -f "${TF_CONFIG_JSON}" ]; then
    echo "[error] TF_CONFIG_JSON not found: ${TF_CONFIG_JSON}" >&2
    exit 1
  fi
  CMD+=(--transformer-config-json "${TF_CONFIG_JSON}")
fi

if [ "${SAVE_CKPT}" = "1" ]; then
  CMD+=(--save-checkpoints)
fi

"${CMD[@]}"

echo "[done] cgcnn -> ${CGCNN_JSON}"
echo "[done] transformer -> ${TRANSFORMER_JSON}"
echo "[done] summary -> ${SUMMARY_JSON}"
