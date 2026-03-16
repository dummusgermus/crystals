#!/bin/bash
#SBATCH --job-name=jara-gts
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Always run from the directory where you submitted the job
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

# If your cluster uses modules, uncomment and adapt:
# module purge
# module load Python/3.11.5
# module load CUDA/12.1

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Dependencies are expected to be installed in .venv already.
# One-time setup (run manually, not in each job):
#   source .venv/bin/activate
#   python -m pip install --upgrade pip
#   pip install torch torch-geometric numpy matplotlib

python -u train_base_transformer_json.py \
  --dataset pyg_dataset.pt \
  --output base_transformer_test_curves.json \
  --epochs 300 \
  --transformer-epochs 300 \
  --batch-size 32 \
  --transformer-batch-size 1 \
  --transformer-max-nodes 0 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --hidden-dim 128 \
  --num-layers 3 \
  --num-heads 4 \
  --dropout 0.1 \
  --transformer-lr 1e-3 \
  --transformer-weight-decay 1e-5 \
  --transformer-hidden-dim 64 \
  --transformer-num-layers 10 \
  --transformer-num-heads 8 \
  --transformer-activation gelu \
  --transformer-attention-dropout 0.2 \
  --transformer-ffn-dropout 0.0 \
  --transformer-gradient-norm 1.0 \
  --transformer-warmup-iters 50 \
  --transformer-min-lr 0.0 \
  --metric mae \
  --seed 42
