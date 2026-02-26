#!/bin/bash
#SBATCH --job-name=jara-all-pes-100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# One-time setup if needed:
#   source .venv/bin/activate
#   python -m pip install --upgrade pip
#   pip install torch torch-geometric numpy matplotlib

# Defaults (override via sbatch --export=ALL,...)
DATASET="${DATASET:-pyg_dataset.pt}"
OUTPUT="${OUTPUT:-base_vs_all_pes_100.json}"
EPOCHS="${EPOCHS:-100}"
METRIC="${METRIC:-mae}"
SEED="${SEED:-42}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-1}"
TRANSFORMER_MAX_NODES="${TRANSFORMER_MAX_NODES:-0}"
TRANSFORMER_WARMUP_ITERS="${TRANSFORMER_WARMUP_ITERS:-10}"

# All available PE options from train_base_best_vs_pe_json.py
PES="${PES:-rwse rrwp lpe spe none 1wl rrwp_pretrans}"

TMP_DIR="tmp_all_pes_${SLURM_JOB_ID}"
mkdir -p "${TMP_DIR}"

echo "Running PE sweep with: ${PES}"
for PE in ${PES}; do
  OUT_FILE="${TMP_DIR}/base_vs_${PE}.json"
  echo "=== PE: ${PE} ==="
  python -u train_base_best_vs_pe_json.py \
    --dataset "${DATASET}" \
    --output "${OUT_FILE}" \
    --epochs "${EPOCHS}" \
    --metric "${METRIC}" \
    --seed "${SEED}" \
    --pe "${PE}" \
    --transformer-batch-size "${TRANSFORMER_BATCH_SIZE}" \
    --transformer-max-nodes "${TRANSFORMER_MAX_NODES}" \
    --transformer-warmup-iters "${TRANSFORMER_WARMUP_ITERS}"
done

python - "${TMP_DIR}" "${OUTPUT}" "${PES}" <<'PY'
import json
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
pe_list = sys.argv[3].split()

payloads = {}
for pe in pe_list:
    path = tmp_dir / f"base_vs_{pe}.json"
    with path.open("r", encoding="utf-8") as f:
        payloads[pe] = json.load(f)

first = payloads[pe_list[0]]
merged = {
    "dataset": first.get("dataset"),
    "metric": first.get("metric"),
    "epochs": first.get("epochs"),
    "seed": first.get("seed"),
    "base_best_config": first.get("base_best_config", {}),
    "base_best_test_curve": first.get("base_best_test_curve", []),
    "pe_test_curves": {},
    "pe_configs": {},
}

for pe in pe_list:
    merged["pe_test_curves"][pe] = payloads[pe].get("pe_transformer_test_curve", [])
    cfg = payloads[pe].get("pe_transformer_config", {})
    cfg["pe"] = pe
    merged["pe_configs"][pe] = cfg

with output.open("w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Saved merged PE sweep JSON to {output}")
PY

rm -rf "${TMP_DIR}"
