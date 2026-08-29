#!/bin/bash
#SBATCH --job-name=jara-tr-gv2sw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Transformer global v2 lambda sweep (1000 epochs, grouped 70/15/15).
#
# Default grids (around delivery λ):
#   point:  0.005, 0.01, 0.02
#   planar: 0.0025, 0.005, 0.01
#
# Checkpointing: val R_tot median; best run selected by test R_tot median.
#
# Outputs:
#   transformer_global_v2_lambda_sweep_curves.json
#   transformer_global_v2_lambda_sweep_summary.json
#   transformer_*_global_v2_lambda_*_model.pt  (per run)
#   transformer_graph_defect_residual_global_v2_best_model.pt
#   transformer_graph_planar_residual_global_v2_best_model.pt
#   transformer_global_v2_lambda_sweep_*.png
#
# Submit (datasets already on cluster):
#   sbatch --export=ALL,SKIP_BUILD=1 run_transformer_global_v2_lambda_sweep.slurm
#
# Resume:
#   sbatch --export=ALL,SKIP_BUILD=1,RESUME=1 run_transformer_global_v2_lambda_sweep.slurm
#
# Custom lambdas:
#   sbatch --export=ALL,SKIP_BUILD=1,POINT_LAMBDAS=0.005,0.01,0.02,PLANAR_LAMBDAS=0.0025,0.005,0.01 run_transformer_global_v2_lambda_sweep.slurm

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
SEED="${SEED:-42}"
POINT_LAMBDAS="${POINT_LAMBDAS:-0.005,0.01,0.02}"
PLANAR_LAMBDAS="${PLANAR_LAMBDAS:-0.0025,0.005,0.01}"
SKIP_BUILD="${SKIP_BUILD:-1}"
RESUME="${RESUME:-0}"
DOMAIN="${DOMAIN:-both}"
OUTPUT_JSON="${OUTPUT_JSON:-transformer_global_v2_lambda_sweep_curves.json}"
SUMMARY_JSON="${SUMMARY_JSON:-transformer_global_v2_lambda_sweep_summary.json}"
PLOT_PREFIX="${PLOT_PREFIX:-transformer_global_v2_lambda_sweep}"

echo "[job] epochs=${EPOCHS} seed=${SEED} domain=${DOMAIN}"
echo "[job] point_lambdas=${POINT_LAMBDAS} planar_lambdas=${PLANAR_LAMBDAS}"
echo "[job] skip_build=${SKIP_BUILD} resume=${RESUME}"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_transformer_global_v2_lambda_sweep.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --point-lambdas "${POINT_LAMBDAS}"
  --planar-lambdas "${PLANAR_LAMBDAS}"
  --domain "${DOMAIN}"
  --output-json "${OUTPUT_JSON}"
  --summary-json "${SUMMARY_JSON}"
  --plot-prefix "${PLOT_PREFIX}"
  --plot
)

if [ "${SKIP_BUILD}" = "1" ]; then
  CMD+=(--skip-build)
fi
if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

"${CMD[@]}"

echo "[done] curves  -> ${OUTPUT_JSON}"
echo "[done] summary -> ${SUMMARY_JSON}"
echo "[done] best    -> transformer_graph_*_global_v2_best_model.pt"
echo "[done] plots   -> ${PLOT_PREFIX}_*.png"
