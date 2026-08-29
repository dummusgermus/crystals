#!/bin/bash
#SBATCH --job-name=jara-cgcnn-gfrac
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# CGCNN global-v2 sweep over point graph size:
#   baseline (shell k=8) + 16/32/48/64/80% of full cell
#
# Phase 1: build datasets (needs SIMULATIONS/ on cluster)
# Phase 2: train 6 tiers × 300 epochs, λ=0.01, checkpoint val R_tot (graph)
#
# Metrics tracked: test MAE, R_tot graph (loss target), R_tot full cell
#
# Outputs (git-tracked after pull):
#   adv_datasets/cycle34_residual_totals_frac_*.pt
#   adv_datasets/point_graph_fraction_manifest.json
#   cgcnn_graph_fraction_sweep_curves.json
#   cgcnn_graph_fraction_sweep_summary.json
#   cgcnn_graph_fraction_sweep_metrics.png
#
# Datasets are built locally (SIMULATIONS/ not on cluster) and synced via git.
# Default: SKIP_BUILD=1 — only train on pre-built adv_datasets/cycle34_residual_totals_frac_*.pt
#
# Submit (train only, datasets from git):
#   sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_graph_fraction_sweep.slurm
#
# Full job (only if SIMULATIONS/ present on node — not typical):
#   sbatch --export=ALL,SKIP_BUILD=0 run_cgcnn_graph_fraction_sweep.slurm
#
# Resume training only:
#   sbatch --export=ALL,SKIP_BUILD=1,RESUME=1 run_cgcnn_graph_fraction_sweep.slurm
#
# Build datasets only (e.g. before git add/push):
#   sbatch --export=ALL,BUILD_ONLY=1 run_cgcnn_graph_fraction_sweep.slurm

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
SKIP_BUILD="${SKIP_BUILD:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"
RESUME="${RESUME:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
TIER="${TIER:-}"
OUTPUT_JSON="${OUTPUT_JSON:-cgcnn_graph_fraction_sweep_curves.json}"
SUMMARY_JSON="${SUMMARY_JSON:-cgcnn_graph_fraction_sweep_summary.json}"
PLOT_PREFIX="${PLOT_PREFIX:-cgcnn_graph_fraction_sweep}"

echo "[job] epochs=${EPOCHS} seed=${SEED} skip_build=${SKIP_BUILD} build_only=${BUILD_ONLY} resume=${RESUME}"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

BUILD_CMD=(python -u build_point_graph_fraction_datasets.py)
if [ "${FORCE_BUILD}" = "1" ]; then
  BUILD_CMD+=(--force)
fi
if [ -n "${TIER}" ]; then
  BUILD_CMD+=(--tier "${TIER}")
fi

if [ "${SKIP_BUILD}" != "1" ]; then
  echo "[phase 1] building graph-fraction datasets …"
  if [ ! -f adv_datasets/cycle34_residual_totals_dataset.pt ]; then
    echo "[phase 1] baseline totals missing — running build_residual_datasets_with_totals.py --point-only"
    python -u build_residual_datasets_with_totals.py --point-only
  fi
  "${BUILD_CMD[@]}"
else
  echo "[phase 1] SKIP_BUILD=1 — using existing adv_datasets/cycle34_residual_totals_frac_*.pt"
fi

if [ "${BUILD_ONLY}" = "1" ]; then
  echo "[done] build-only finished"
  exit 0
fi

TRAIN_CMD=(
  python -u train_cgcnn_graph_fraction_sweep.py
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --output-json "${OUTPUT_JSON}"
  --summary-json "${SUMMARY_JSON}"
  --plot-prefix "${PLOT_PREFIX}"
  --skip-build
  --plot
)
if [ "${RESUME}" = "1" ]; then
  TRAIN_CMD+=(--resume)
fi
if [ -n "${TIER}" ]; then
  TRAIN_CMD+=(--tier "${TIER}")
fi

echo "[phase 2] training CGCNN graph-fraction sweep …"
"${TRAIN_CMD[@]}"

echo "[done] manifest -> adv_datasets/point_graph_fraction_manifest.json"
echo "[done] curves   -> ${OUTPUT_JSON}"
echo "[done] summary  -> ${SUMMARY_JSON}"
echo "[done] plot     -> ${PLOT_PREFIX}_metrics.png"
echo "[hint] git add adv_datasets/cycle34_residual_totals_frac_*.pt adv_datasets/point_graph_fraction_manifest.json"
