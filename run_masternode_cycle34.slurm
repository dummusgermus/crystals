#!/bin/bash
#SBATCH --job-name=jara-masternode-c34
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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

# Defaults (override via sbatch --export=ALL,EPOCHS=500,...)
EPOCHS="${EPOCHS:-300}"
METRIC="${METRIC:-mae}"
SEED="${SEED:-42}"
DATASET="${DATASET:-adv_datasets/masternode_cycle34_dataset.pt}"
OUTPUT="${OUTPUT:-masternode_cycle34.json}"

# Step 1: build the master-node dataset if it does not exist yet
if [ ! -f "${DATASET}" ]; then
  echo ">>> Creating master-node dataset …"
  python -u masternode_graph_maker.py
fi

# Step 2: train
python -u train_masternode_cycle34.py \
  --dataset "${DATASET}" \
  --output "${OUTPUT}" \
  --epochs "${EPOCHS}" \
  --metric "${METRIC}" \
  --seed "${SEED}"
