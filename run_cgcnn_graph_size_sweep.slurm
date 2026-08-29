#!/bin/bash
#SBATCH --job-name=jara-cgcnn-gsz
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# CGCNN global-v2 sweep: shell cutoff_k tiers k08..k13 + k22.
# Datasets are built locally and synced via git (SKIP_BUILD=1).
#
# Tracks per tier:
#   - test MAE (per-atom)
#   - test R_tot median (graph target + full cell)
#   - training wall time + s/epoch
#   - test-set inference throughput (ms/graph)
#
# Required git-tracked files before submit:
#   adv_datasets/cycle34_residual_totals_size_k{08,09,10,11,12,13,22}.pt
#   adv_datasets/point_graph_size_manifest.json
#
# Submit:
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_graph_size_sweep.slurm
#
# Resume interrupted job:
#   sbatch --export=ALL,SKIP_BUILD=1,RESUME=1 run_cgcnn_graph_size_sweep.slurm

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
  else
    echo "[env] WARNING: no libGL.so.1 found (OVITO not needed for training)." >&2
  fi
fi
export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH:-}"

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
SKIP_BUILD="${SKIP_BUILD:-1}"
RESUME="${RESUME:-0}"

echo "[preflight] checking datasets …"
python - <<'PY'
import sys
from graph_size_datasets_config import GRAPH_SIZE_TIERS, dataset_path
missing = [tag for tag, _ in GRAPH_SIZE_TIERS if not __import__("os").path.isfile(dataset_path(tag))]
if missing:
    print("Missing datasets:", ", ".join(missing), file=sys.stderr)
    print("Push adv_datasets/cycle34_residual_totals_size_*.pt from local build.", file=sys.stderr)
    sys.exit(1)
print("All", len(GRAPH_SIZE_TIERS), "tiers present.", flush=True)
PY

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_cgcnn_graph_size_sweep.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --skip-build
  --plot
)
[ "${RESUME}" = "1" ] && CMD+=(--resume)

echo "[job] epochs=${EPOCHS} seed=${SEED} resume=${RESUME}"
"${CMD[@]}"

echo "[done] curves  -> cgcnn_graph_size_sweep_curves.json"
echo "[done] summary -> cgcnn_graph_size_sweep_summary.json"
echo "[done] plots   -> cgcnn_graph_size_sweep_*.png"
