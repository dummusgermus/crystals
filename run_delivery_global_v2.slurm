#!/bin/bash
#SBATCH --job-name=jara-pred-new
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# predictions_new production pipeline (k13 point + global-v2 optimizations):
#   * Point graphs: cutoff_k=13, edge_k=3
#   * Point CGCNN: full-cell global loss (λ=0.01)
#   * Point Transformer: graph global loss (λ=0.02)
#   * Planar CGCNN: graph global loss (λ=0.005)
#   * Planar Transformer: graph global loss (λ=0.01)
#   * 90/10 within-group split, checkpoint by val R_tot
#   * Val-set inference benchmarks + crystal inference (no dumps)
#
# Build datasets locally, rsync .pt files, then:
#   sbatch run_delivery_global_v2.slurm
#
# Fresh retrain (overwrite checkpoints + split):
#   sbatch --export=ALL,FORCE_SPLIT=1 run_delivery_global_v2.slurm
#
# Resume interrupted training:
#   sbatch --export=ALL,RESUME=1 run_delivery_global_v2.slurm
#
# Train only (skip crystal inference):
#   sbatch --export=ALL,SKIP_INFER=1 run_delivery_global_v2.slurm
#
# Local restore after pull (overwrites predictions_new/):
#   bash run_predictions_new_restore.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs predictions_new

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

EPOCHS="${EPOCHS:-2000}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_SPLIT="${FORCE_SPLIT:-1}"
RESUME="${RESUME:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_BUILD="${SKIP_BUILD:-1}"
INFER_ROOT="${INFER_ROOT:-predictions_new_inference}"
SPLIT_JSON="${SPLIT_JSON:-predictions_new/delivery_split_indices.json}"
DEVICE="${DEVICE:-auto}"

echo "============================================================"
echo "[job] predictions_new delivery (${EPOCHS} ep)"
echo "[job] version=global_v2_k13 force_split=${FORCE_SPLIT} resume=${RESUME}"
echo "[job] skip_train=${SKIP_TRAIN} skip_infer=${SKIP_INFER} skip_build=${SKIP_BUILD}"
echo "============================================================"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

echo
echo "=== Preflight: delivery datasets ==="
python - <<'PY'
import sys
import torch
from build_delivery_datasets import datasets_ready
from delivery_global_v2 import EXPORT_DATASETS, TOTALS_DATASETS, POINT_CUTOFF_K, POINT_EDGE_K

paths = list(TOTALS_DATASETS.values()) + list(EXPORT_DATASETS.values())
missing = [p for p in paths if not __import__("os").path.isfile(p)]
if missing:
    print("Missing datasets:", *missing, sep="\n  ", file=sys.stderr)
    print("Build locally: python build_delivery_datasets.py", file=sys.stderr)
    sys.exit(1)

pt = torch.load(TOTALS_DATASETS["point"], weights_only=False)
g = pt[0]
print(f"[ok] point totals: n={len(pt)} cutoff_k={getattr(g,'cutoff_k',None)} edge_k={getattr(g,'edge_k',None)}", flush=True)
if int(getattr(g, "cutoff_k", -1)) != POINT_CUTOFF_K or int(getattr(g, "edge_k", -1)) != POINT_EDGE_K:
    sys.exit(f"Point dataset must be k={POINT_CUTOFF_K} edge_k={POINT_EDGE_K}")
if not hasattr(g, "particle_ids"):
    sys.exit("Point export dataset missing particle_ids")

for domain, path in EXPORT_DATASETS.items():
    ds = torch.load(path, weights_only=False)
    print(f"[ok] {domain} export: {path} n={len(ds)}", flush=True)
print("All delivery datasets present.", flush=True)
PY

TRAIN_CMD=(
  python -u train_delivery_global_v2.py
  --epochs "${EPOCHS}"
  --split-json "${SPLIT_JSON}"
  --model all
)
if [ "${SKIP_BUILD}" = "1" ]; then
  TRAIN_CMD+=(--skip-build)
elif [ "${FORCE_BUILD}" = "1" ]; then
  TRAIN_CMD+=(--force-build)
fi
if [ "${FORCE_SPLIT}" = "1" ]; then
  TRAIN_CMD+=(--force-split)
fi
if [ "${RESUME}" = "1" ]; then
  TRAIN_CMD+=(--resume)
fi

if [ "${SKIP_TRAIN}" != "1" ]; then
  echo
  echo "=== Training 4 models + val inference benchmarks ==="
  "${TRAIN_CMD[@]}"
else
  echo
  echo "=== SKIP train ==="
fi

echo
echo "=== Checkpoint presence ==="
for ckpt in \
  cgcnn_defect_residual_global_v2_model.pt \
  cgcnn_planar_residual_global_v2_model.pt \
  transformer_graph_defect_residual_global_v2_model.pt \
  transformer_graph_planar_residual_global_v2_model.pt
do
  if [ -f "${ckpt}" ]; then
    echo "[ok] ${ckpt} ($(du -h "${ckpt}" | awk '{print $1}'))"
  else
    echo "[MISSING] ${ckpt}" >&2
    exit 1
  fi
done

if [ "${SKIP_INFER}" != "1" ]; then
  echo
  echo "=== Crystal inference (no dumps) ==="
  python -u crystal_prediction_export.py \
    --mode inference \
    --job all \
    --job-profile global_v2 \
    --output-root "${INFER_ROOT}" \
    --device "${DEVICE}"

  for sub in point_cgcnn point_transformer planar_cgcnn planar_transformer; do
    summary="${INFER_ROOT}/${sub}/inference_summary.json"
    if [ ! -f "${summary}" ]; then
      echo "[MISSING] ${summary}" >&2
      exit 1
    fi
    echo "[ok] ${summary}"
  done
else
  echo
  echo "=== SKIP crystal inference ==="
fi

echo
echo "[done] Pull artifacts:"
echo "  ${SPLIT_JSON}"
echo "  *_global_v2_model.pt (4 checkpoints)"
echo "  delivery_global_v2_curves.json"
echo "  delivery_inference_benchmark.json"
echo "  ${INFER_ROOT}/"
echo "[done] Local: bash run_predictions_new_restore.sh  (overwrites predictions_new/)"
