#!/bin/bash
#SBATCH --job-name=jara-base-vs-pe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Always run from submit directory.
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Optional module setup:
# module purge
# module load Python/3.11.5
# module load CUDA/12.1

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# One-time dependency setup (run manually once):
#   source .venv/bin/activate
#   python -m pip install --upgrade pip
#   pip install torch torch-geometric numpy matplotlib

# Overridable defaults (can be changed via: sbatch --export=ALL,EPOCHS=300,...)
DATASET="${DATASET:-pyg_dataset.pt}"
OUTPUT="${OUTPUT:-base_best_vs_pe_transformer_300.json}"
EPOCHS="${EPOCHS:-300}"
METRIC="${METRIC:-mae}"
SEED="${SEED:-42}"

# Keep this conservative by default for the PE transformer memory footprint.
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-1}"
TRANSFORMER_MAX_NODES="${TRANSFORMER_MAX_NODES:-0}"

python -u train_base_best_vs_pe_json.py \
  --dataset "${DATASET}" \
  --output "${OUTPUT}" \
  --epochs "${EPOCHS}" \
  --metric "${METRIC}" \
  --seed "${SEED}" \
  --transformer-batch-size "${TRANSFORMER_BATCH_SIZE}" \
  --transformer-max-nodes "${TRANSFORMER_MAX_NODES}"
