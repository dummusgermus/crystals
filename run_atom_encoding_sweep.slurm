#!/bin/bash
#SBATCH --job-name=jara-atomenc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=log_gpu_24gb,gpu_11gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Atom-type encoding sweep (see atom_encodings.py / train_atom_encoding_sweep.py).
# Each array task runs up to 4 trainings in parallel on one GPU.
#   python atom_encoding_sweep/make_tasks.py phase1
#   sbatch --array=1-$(wc -l < atom_encoding_sweep/tasks_phase1.txt) \
#          --export=ALL,TASKS=atom_encoding_sweep/tasks_phase1.txt run_atom_encoding_sweep.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
# W&B (project yara-crystals): API key from ~/.my_profile; disable with --no-wandb.
eval "$(grep -E '^export WANDB_API_KEY=' /work/log1/erik.mueller/.my_profile 2>/dev/null || true)"
TASKS="${TASKS:-atom_encoding_sweep/tasks_phase1.txt}"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKS}")"
echo "[task ${SLURM_ARRAY_TASK_ID}] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
pids=()
IFS=$'\n' read -r -d '' -a CMDS < <(echo "${LINE}" | sed 's/ ;; /\n/g' && printf '\0')
for cmd in "${CMDS[@]}"; do
  echo "[launch] ${cmd}"
  bash -c "${cmd}" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
