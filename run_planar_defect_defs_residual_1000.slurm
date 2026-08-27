#!/bin/bash
#SBATCH --job-name=jara-planar-defs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Planar residual ΔPE: compare two defect-site definitions on identical graphs.
#   broad ISF/ESF  -> laves_defect_atoms.json
#   C14/C15        -> laves_defect_atoms_c14c15.json
#
# Required for dataset rebuild:
#   Laves_Planar_Defects/SIMULATIONS/
#   Laves_Screen/SIMULATIONS/
#   laves_defect_atoms.json
#   laves_defect_atoms_c14c15.json
#
# Submit:
#   sbatch run_planar_defect_defs_residual_1000.slurm
#
# Skip rebuild (use existing residual_isf / residual_c14c15 .pt files):
#   sbatch --export=ALL,SKIP_BUILD=1 run_planar_defect_defs_residual_1000.slurm

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

# OVITO OpenGL shim (needed if rebuilding datasets on the node)
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

EPOCHS="${EPOCHS:-1000}"
SCORE_LAST_N="${SCORE_LAST_N:-50}"
SEED="${SEED:-42}"
MODELS="${MODELS:-both}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SAVE_CKPT="${SAVE_CKPT:-0}"
TF_CONFIG_JSON="${TF_CONFIG_JSON:-}"

echo "[job] epochs=${EPOCHS} models=${MODELS} skip_build=${SKIP_BUILD}"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u run_planar_defect_defs_residual_1000.py
  --epochs "${EPOCHS}"
  --score-last-n "${SCORE_LAST_N}"
  --seed "${SEED}"
  --models "${MODELS}"
  --force-build
)

if [ "${SKIP_BUILD}" = "1" ]; then
  CMD=(
    python -u run_planar_defect_defs_residual_1000.py
    --epochs "${EPOCHS}"
    --score-last-n "${SCORE_LAST_N}"
    --seed "${SEED}"
    --models "${MODELS}"
    --skip-build
  )
fi

if [ -n "${TF_CONFIG_JSON}" ]; then
  CMD+=(--transformer-config-json "${TF_CONFIG_JSON}")
fi
if [ "${SAVE_CKPT}" = "1" ]; then
  CMD+=(--save-checkpoints)
fi

"${CMD[@]}"
echo "[done]"
