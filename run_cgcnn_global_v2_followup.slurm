#!/bin/bash
#SBATCH --job-name=jara-cgcnn-gv2f
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Global v2 follow-up: 300 epochs, lambda 0.01 and 0.005 vs atom-only baseline.
#
# Checkpointing:
#   baseline (atom-only): val MAE
#   global v2 runs:       val R_tot median
#
# Outputs:
#   cgcnn_global_v2_followup_curves.json
#   cgcnn_global_v2_followup_summary.json
#   cgcnn_global_v2_followup_epoch_curves.png
#   cgcnn_global_v2_followup_best_vs_baseline_bars.png
#
# Submit:
#   sbatch run_cgcnn_global_v2_followup.slurm
#
# Resume interrupted job:
#   sbatch --export=ALL,RESUME=1,SKIP_BUILD=1 run_cgcnn_global_v2_followup.slurm
#
# Skip dataset rebuild:
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_global_v2_followup.slurm

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

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
LAMBDAS="${LAMBDAS:-0.01,0.005}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RESUME="${RESUME:-0}"
DOMAIN="${DOMAIN:-both}"
OUTPUT_JSON="${OUTPUT_JSON:-cgcnn_global_v2_followup_curves.json}"
SUMMARY_JSON="${SUMMARY_JSON:-cgcnn_global_v2_followup_summary.json}"
PLOT_PREFIX="${PLOT_PREFIX:-cgcnn_global_v2_followup}"

echo "[job] epochs=${EPOCHS} seed=${SEED} lambdas=${LAMBDAS} domain=${DOMAIN}"
echo "[job] skip_build=${SKIP_BUILD} resume=${RESUME}"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_cgcnn_global_v2_followup.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --lambdas "${LAMBDAS}"
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
echo "[done] plots   -> ${PLOT_PREFIX}_*.png"
