#!/bin/bash
#SBATCH --job-name=jara-pilot
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=log_gpu_24gb,log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Point-defect generation pilot (displacement/gen/point_pilot.py): task k relaxes part k of NPARTS.
#   NPARTS=6 sbatch --array=0-5 displacement/gen/run_pilot.slurm
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2
NP=${NPARTS:-6}
/work/log1/erik.mueller/grace-venv/bin/python -I displacement/gen/point_pilot.py relax \
  displacement/gen_data/pilot_part${SLURM_ARRAY_TASK_ID}.pkl --n-compounds 30 --part $SLURM_ARRAY_TASK_ID --nparts $NP
