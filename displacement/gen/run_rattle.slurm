#!/bin/bash
#SBATCH --job-name=jara-rattle
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=log_gpu_24gb,log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Symmetry-breaking test (displacement/gen/rattle_test.py): task k handles part k of NPARTS.
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2
/work/log1/erik.mueller/grace-venv/bin/python -I displacement/gen/rattle_test.py \
  displacement/gen_data/rattle_part${SLURM_ARRAY_TASK_ID}.pkl --part $SLURM_ARRAY_TASK_ID --nparts ${NPARTS:-4}
