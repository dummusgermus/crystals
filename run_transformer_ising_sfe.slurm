#!/bin/bash
#SBATCH --job-name=jara-ising-gt
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Graph Transformer on 20261020 Ising/SFE total-energy dataset.
# Same split/protocol as run_cgcnn_ising_sfe.slurm (reuses split JSON if present).
#
# Prerequisites:
#   adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt
#   train_transformer_ising_sfe.py
#   gnn_models.py (with GraphTransformerGraphRegressor)
#
# Outputs:
#   transformer_ising_sfe_total_energy_model.pt
#   transformer_ising_sfe_curves.json
#   transformer_ising_sfe_summary.json
#   transformer_ising_sfe_epoch_curves.png
#
# Submit:
#   sbatch run_transformer_ising_sfe.slurm
#
# Overrides:
#   sbatch --export=ALL,EPOCHS=300,BATCH_SIZE=8 run_transformer_ising_sfe.slurm

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
BATCH_SIZE="${BATCH_SIZE:-8}"
DATASET="${DATASET:-adv_datasets/ising_sfe_cycle34_total_energy_dataset.pt}"

echo "============================================================"
echo "[job] Ising/SFE graph-level total-energy Graph Transformer"
echo "[job] epochs=${EPOCHS} seed=${SEED} batch_size=${BATCH_SIZE}"
echo "[job] dataset=${DATASET}"
echo "============================================================"

if [ ! -f "${DATASET}" ]; then
  echo "[FATAL] Missing dataset: ${DATASET}" >&2
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
echo "=== Training Ising/SFE Graph Transformer ==="
python -u train_transformer_ising_sfe.py \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --plot

echo
echo "=== Outputs ==="
for f in \
  transformer_ising_sfe_total_energy_model.pt \
  transformer_ising_sfe_curves.json \
  transformer_ising_sfe_summary.json \
  transformer_ising_sfe_epoch_curves.png
do
  if [ -f "${f}" ]; then
    echo "[ok] ${f}"
  else
    if [[ "${f}" == *.png ]]; then
      echo "[warn] missing optional ${f}"
    else
      echo "[MISSING] ${f}" >&2
      exit 1
    fi
  fi
done

echo
echo "[done] Pull JSON, checkpoint, and PNG for comparison with CGCNN."
