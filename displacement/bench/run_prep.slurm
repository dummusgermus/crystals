#!/bin/bash
#SBATCH --job-name=jara-bprep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=log_gpu_24gb,log_gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage 1 (GRACE features + test-set GRACE relaxation) and stage 2 (PyG views) for one benchmark dataset.
#   sbatch --export=ALL,DS=c2db displacement/bench/run_prep.slurm
set -euo pipefail
cd /work/log1/erik.mueller/code
export HOME=/work/log1/erik.mueller/.grace TF_CPP_MIN_LOG_LEVEL=2
P=displacement/bench_data/prep/${DS}.pkl
/work/log1/erik.mueller/grace-venv/bin/python -I displacement/bench/grace_feats.py "$DS" "$P" --relax-test
.venv/bin/python displacement/bench/build_views.py "$P" "$DS"
