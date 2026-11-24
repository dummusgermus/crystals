#!/bin/bash
#SBATCH --job-name=jara-gen
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=log_gpu_24gb,log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Point-defect data generation (displacement/gen/point_gen.py): task k handles part k of NPARTS; resumable.
#   sbatch --export=ALL,NPARTS=12 --array=0-11%12 displacement/gen/run_gen.slurm
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2
/work/log1/erik.mueller/grace-venv/bin/python -I displacement/gen/point_gen.py run displacement/gen_data/gen_v1 \
  --part $SLURM_ARRAY_TASK_ID --nparts ${NPARTS:-12}
