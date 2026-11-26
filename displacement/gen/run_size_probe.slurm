#!/bin/bash
#SBATCH --job-name=jara-sizeprobe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
# GRACE memory/time probe for large C15 cells (displacement/gen/size_probe.py)
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/work/log1/erik.mueller/grace-venv/bin/python -I displacement/gen/size_probe.py Ag-Be | tee displacement/gen_data/size_probe.jsonl
