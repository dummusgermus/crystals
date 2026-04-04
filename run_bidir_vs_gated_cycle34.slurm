#!/bin/bash
#SBATCH --job-name=jara-bidir-vs-gated-c34
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

EPOCHS="${EPOCHS:-300}"
METRIC="${METRIC:-mae}"
SEED="${SEED:-42}"
DATASET="${DATASET:-adv_datasets/cycle34_dataset.pt}"
OUTPUT="${OUTPUT:-bidir_vs_gated_cycle34.json}"

python -u train_bidir_vs_gated_cycle34.py \
  --dataset "${DATASET}" \
  --output "${OUTPUT}" \
  --epochs "${EPOCHS}" \
  --metric "${METRIC}" \
  --seed "${SEED}"
