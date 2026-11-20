#!/bin/bash
#SBATCH --job-name=jara-dispbuild
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=log_gpu_48gb,log_gpu_24gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
# Build displacement superset graphs: sbatch --export=ALL,DOMAIN=point displacement/run_build.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python displacement/build_superset.py "${DOMAIN}" --workers 32 ${EXTRA:-}
