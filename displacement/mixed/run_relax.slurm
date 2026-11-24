#!/bin/bash
#SBATCH --job-name=jara-mixrel
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=log_gpu_24gb,log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Mixed-defect cells, stage 1: task 0 = protocol check on 24 planar training cells, tasks 1..N = cells parts.
#   sbatch --array=0-6 displacement/mixed/run_relax.slurm
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2
PY="/work/log1/erik.mueller/grace-venv/bin/python -I displacement/mixed/grace_relax.py"
D=displacement/mixed_data
NP=${NPARTS:-6}
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  $PY validate $D/validate.pkl --n 24
else
  $PY cells $D/cells.pkl $D/relaxed_part$((SLURM_ARRAY_TASK_ID-1)).pkl --part $((SLURM_ARRAY_TASK_ID-1)) --nparts $NP
fi
