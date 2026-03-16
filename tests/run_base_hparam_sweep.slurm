#!/bin/bash
#SBATCH --job-name=jara-base-sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
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

# One-time dependency setup (if needed):
#   source .venv/bin/activate
#   python -m pip install --upgrade pip
#   pip install torch torch-geometric numpy matplotlib

python -u train_base_hparam_sweep_json.py \
  --dataset pyg_dataset.pt \
  --output base_hparam_sweep_curves.json \
  --epochs 100 \
  --metric mae \
  --seed 42 \
  --hidden-dims 64,128,256 \
  --num-layers-list 2,4 \
  --dropouts 0.0,0.1 \
  --lrs 1e-3,5e-4,1e-4 \
  --weight-decays 0.0,1e-5 \
  --batch-sizes 16,32 \
  --activations relu,silu,gelu \
  --use-batch-norm-list false
