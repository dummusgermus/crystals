#!/bin/bash
#SBATCH --job-name=jara-ising-sfe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Graph-level CGCNN on 20261020 Ising/SFE total-energy dataset.
#
# Prerequisites (place on the cluster code root before submit):
#   adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt
#   (optional) adv_datasets/ising_sfe_cycle34_total_energy_stats.json
#
# Split: material-grouped 70/15/15 (train_cgcnn_ising_sfe.py)
# Architecture: GatedGNNGraphRegressor (same as graph-total CGCNN)
#
# Outputs:
#   cgcnn_ising_sfe_total_energy_model.pt
#   cgcnn_ising_sfe_curves.json
#   cgcnn_ising_sfe_summary.json
#   adv_datasets/ising_sfe_split_indices.json
#   cgcnn_ising_sfe_epoch_curves.png   (if matplotlib available)
#
# Submit:
#   sbatch run_cgcnn_ising_sfe.slurm
#
# Overrides:
#   sbatch --export=ALL,EPOCHS=500 run_cgcnn_ising_sfe.slurm
#   sbatch --export=ALL,BATCH_SIZE=64,EPOCHS=300 run_cgcnn_ising_sfe.slurm
#   sbatch --export=ALL,DATASET=adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt run_cgcnn_ising_sfe.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs adv_datasets

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

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET="${DATASET:-adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt}"

echo "============================================================"
echo "[job] Ising/SFE graph-level total-energy CGCNN"
echo "[job] epochs=${EPOCHS} seed=${SEED} batch_size=${BATCH_SIZE}"
echo "[job] dataset=${DATASET}"
echo "============================================================"

if [ ! -f "${DATASET}" ]; then
  echo "[FATAL] Missing dataset: ${DATASET}" >&2
  echo "        Copy ising_sfe_cycle34_total_energy_dataset.pt to adv_datasets/ first." >&2
  exit 1
fi

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), flush=True)
PY

echo
echo "=== Training Ising/SFE CGCNN ==="
python -u train_cgcnn_ising_sfe.py \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --plot

echo
echo "=== Outputs ==="
for f in \
  cgcnn_ising_sfe_total_energy_model.pt \
  cgcnn_ising_sfe_curves.json \
  cgcnn_ising_sfe_summary.json \
  adv_datasets/ising_sfe_split_indices.json \
  cgcnn_ising_sfe_epoch_curves.png
do
  if [ -f "${f}" ]; then
    echo "[ok] ${f}"
  else
    # PNG is optional if matplotlib missing
    if [[ "${f}" == *.png ]]; then
      echo "[warn] missing optional ${f}"
    else
      echo "[MISSING] ${f}" >&2
      exit 1
    fi
  fi
done

echo
echo "[done] Pull JSON, checkpoint, split, and PNG for analysis."
