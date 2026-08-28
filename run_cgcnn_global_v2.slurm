#!/bin/bash
#SBATCH --job-name=jara-cgcnn-gv2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Quick sanity check: atom-only vs improved global loss (single lambda).
#
# Improvements vs failed lambda sweep:
#   graph_delta target, Huber loss, loss balancing, outlier down-weighting
#
# Outputs:
#   cgcnn_global_v2_curves.json
#   cgcnn_global_v2_summary.json
#   cgcnn_global_v2_curves.png
#
# Submit:
#   sbatch run_cgcnn_global_v2.slurm
#
# Faster smoke test:
#   sbatch --export=ALL,EPOCHS=30 run_cgcnn_global_v2.slurm
#
# Skip dataset rebuild:
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_global_v2.slurm

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

EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
LAMBDA_TOT="${LAMBDA_TOT:-0.01}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DOMAIN="${DOMAIN:-both}"
OUTPUT_JSON="${OUTPUT_JSON:-cgcnn_global_v2_curves.json}"
SUMMARY_JSON="${SUMMARY_JSON:-cgcnn_global_v2_summary.json}"
PLOT_OUTPUT="${PLOT_OUTPUT:-cgcnn_global_v2_curves.png}"

echo "[job] epochs=${EPOCHS} seed=${SEED} lambda_tot=${LAMBDA_TOT} domain=${DOMAIN}"
echo "[job] skip_build=${SKIP_BUILD}"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_cgcnn_global_v2.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --lambda-tot "${LAMBDA_TOT}"
  --domain "${DOMAIN}"
  --output-json "${OUTPUT_JSON}"
  --summary-json "${SUMMARY_JSON}"
  --plot-output "${PLOT_OUTPUT}"
  --plot
)

if [ "${SKIP_BUILD}" = "1" ]; then
  CMD+=(--skip-build)
fi

"${CMD[@]}"

echo "[done] curves  -> ${OUTPUT_JSON}"
echo "[done] summary -> ${SUMMARY_JSON}"
echo "[done] plot    -> ${PLOT_OUTPUT}"
