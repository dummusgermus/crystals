#!/bin/bash
#SBATCH --job-name=jara-k13-le
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# CGCNN k=13: graph vs full-cell global loss (edge_k=3, 2 runs × 300 ep).
# Required: adv_datasets/cycle34_residual_totals_k13_edge_e03.pt
#
# Submit:
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_k13_loss_edge_sweep.slurm
#
# Resume:
#   sbatch --export=ALL,SKIP_BUILD=1,RESUME=1 run_cgcnn_k13_loss_edge_sweep.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
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
  fi
fi
export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH:-}"

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
SKIP_BUILD="${SKIP_BUILD:-1}"
RESUME="${RESUME:-0}"

echo "[preflight] checking k13 edge datasets …"
python - <<'PY'
import sys
from k13_edge_datasets_config import EDGE_K_TIERS, dataset_path
missing = [tag for tag, _ in EDGE_K_TIERS if not __import__("os").path.isfile(dataset_path(tag))]
if missing:
    print("Missing datasets:", ", ".join(missing), file=sys.stderr)
    print("Build locally: python build_point_k13_edge_datasets.py", file=sys.stderr)
    sys.exit(1)
print("All", len(EDGE_K_TIERS), "edge tiers present.", flush=True)
PY

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_cgcnn_k13_loss_edge_sweep.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --skip-build
  --plot
)
[ "${RESUME}" = "1" ] && CMD+=(--resume)

echo "[job] epochs=${EPOCHS} seed=${SEED} resume=${RESUME}"
"${CMD[@]}"

echo "[done] curves  -> cgcnn_k13_loss_edge_sweep_curves.json"
echo "[done] summary -> cgcnn_k13_loss_edge_sweep_summary.json"
echo "[done] plots   -> cgcnn_k13_loss_edge_sweep_*.png"
