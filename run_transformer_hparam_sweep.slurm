#!/bin/bash
#SBATCH --job-name=jara-tf-sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Graph Transformer hyperparameter sweep on point residual (500 epochs each),
# then a full 1000-epoch training of the best config on all four datasets
# (point/planar × absolute/residual).
#
# Required datasets in the submit directory:
#   adv_datasets/cycle34_dataset.pt
#   adv_datasets/cycle34_residual_dataset.pt
#   planar_pyg_dataset_c14c15.pt
#   planar_pyg_dataset_residual_c14c15.pt
#
# Submit:
#   sbatch run_transformer_hparam_sweep.slurm
#
# Resume an interrupted sweep:
#   sbatch --export=ALL,RESUME=1 run_transformer_hparam_sweep.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/torch-cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# Overridable defaults
N_CONFIGS="${N_CONFIGS:-144}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-500}"
FINAL_EPOCHS="${FINAL_EPOCHS:-1000}"
SCORE_LAST_N="${SCORE_LAST_N:-50}"
SEED="${SEED:-42}"
SWEEP_JSON="${SWEEP_JSON:-transformer_hparam_sweep.json}"
FINAL_JSON="${FINAL_JSON:-transformer_best_1000_curves.json}"
RESUME="${RESUME:-0}"

REQUIRED_DATASETS=(
  "adv_datasets/cycle34_dataset.pt"
  "adv_datasets/cycle34_residual_dataset.pt"
  "planar_pyg_dataset_c14c15.pt"
  "planar_pyg_dataset_residual_c14c15.pt"
)

for ds in "${REQUIRED_DATASETS[@]}"; do
  if [ ! -f "${ds}" ]; then
    echo "[error] missing dataset: ${ds}" >&2
    exit 1
  fi
done

RESUME_FLAG=()
if [ "${RESUME}" = "1" ]; then
  RESUME_FLAG=(--resume)
fi

echo "[sweep] n_configs=${N_CONFIGS} sweep_epochs=${SWEEP_EPOCHS} final_epochs=${FINAL_EPOCHS}"
echo "[sweep] score_last_n=${SCORE_LAST_N} seed=${SEED}"
echo "[sweep] device check:"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
PY

CMD=(
  python -u train_transformer_hparam_sweep.py
  --n-configs "${N_CONFIGS}"
  --sweep-epochs "${SWEEP_EPOCHS}"
  --final-epochs "${FINAL_EPOCHS}"
  --score-last-n "${SCORE_LAST_N}"
  --seed "${SEED}"
  --sweep-json "${SWEEP_JSON}"
  --final-json "${FINAL_JSON}"
  --save-checkpoints
)
if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi
"${CMD[@]}"

echo "[done] sweep -> ${SWEEP_JSON}"
echo "[done] final -> ${FINAL_JSON}"
