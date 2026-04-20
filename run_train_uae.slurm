#!/bin/bash
#SBATCH --job-name=jara-uae-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
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

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/torch-cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ---------------------------------------------------------------------------
# OVITO's pip wheel needs libOpenGL.so.0 (libglvnd). Some compute nodes only
# ship the pre-libglvnd monolithic libGL.so.1, which still exports all the
# symbols OVITO touches in scripting mode. Symlink it into a local shim
# directory and prepend to LD_LIBRARY_PATH so dlopen() succeeds.
# ---------------------------------------------------------------------------
SHIM_DIR="${SLURM_SUBMIT_DIR}/.lib_shim"
mkdir -p "${SHIM_DIR}"
if [ ! -e "${SHIM_DIR}/libOpenGL.so.0" ]; then
  SYS_LIBGL="$(ldconfig -p | awk '/libGL\.so\.1\b/ {print $NF; exit}')"
  if [ -z "${SYS_LIBGL}" ] || [ ! -e "${SYS_LIBGL}" ]; then
    SYS_LIBGL="/lib/x86_64-linux-gnu/libGL.so.1"
  fi
  if [ -e "${SYS_LIBGL}" ]; then
    ln -sf "${SYS_LIBGL}" "${SHIM_DIR}/libOpenGL.so.0"
    echo "[env] libOpenGL.so.0 -> ${SYS_LIBGL}"
  else
    echo "[env] WARNING: no libGL.so.1 found; OVITO import will likely fail." >&2
  fi
fi
export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Configuration (override via `sbatch --export=ALL,VAR=value run_train_uae.slurm`)
# ---------------------------------------------------------------------------
# Training dataset. Defaults to the cycle34 variant + bidirectional edges,
# which is the current best-known configuration for the downstream GNN.
CYCLE_VARIANT="${CYCLE_VARIANT:-cycle34}"            # cycle3 | cycle34 | cycle345
ADV_DIR="${ADV_DIR:-adv_datasets_uae}"
DATASET="${DATASET:-${ADV_DIR}/${CYCLE_VARIANT}_dataset.pt}"

UAE_CKPT="${UAE_CKPT:-uae_embeddings/uae_emb128.pt}"
UAE_EPOCHS="${UAE_EPOCHS:-80}"
UAE_FEATURE_SIZE="${UAE_FEATURE_SIZE:-256}"
UAE_LAYERS="${UAE_LAYERS:-4}"

INNER="${INNER:-gated}"                       # gated | base
EPOCHS="${EPOCHS:-300}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
METRIC="${METRIC:-mae}"
SEED="${SEED:-42}"
EMB_DIM="${EMB_DIM:-128}"
BIDIR_FLAG="${BIDIR_FLAG:---bidirectional}"   # set to "" to disable
FREEZE_FLAG="${FREEZE_FLAG:-}"                # set to "--freeze-uae" to freeze
NO_PRETRAINED_FLAG="${NO_PRETRAINED_FLAG:-}"  # set to "--no-pretrained" for fresh UAE
KEEP_TYPE_SCALAR_FLAG="${KEEP_TYPE_SCALAR_FLAG:-}"  # set to "--keep-type-scalar" to concat
OUTPUT_JSON="${OUTPUT_JSON:-uae_${INNER}_${CYCLE_VARIANT}.json}"

# Whether to (re)build the cycle-augmented dataset before training.
# Only set to 1 if the SIMULATIONS/ folder is present on this machine.
BUILD_DATASET="${BUILD_DATASET:-0}"
# Whether to (re)run UAE pretraining before training.
# Only set to 1 if the SIMULATIONS/ folder is present on this machine.
PRETRAIN_UAE="${PRETRAIN_UAE:-0}"

# ---------------------------------------------------------------------------
# Step 1: cycle-augmented dataset with Data.z / Data.folder
#
# The cluster does not host the raw LAMMPS simulations. Build this file
# locally with `python adv_graph_maker.py --variants ${CYCLE_VARIANT}` and
# upload `${ADV_DIR}/${CYCLE_VARIANT}_dataset.pt` before submitting.
# ---------------------------------------------------------------------------
if [ "${BUILD_DATASET}" = "1" ]; then
  echo "[step 1] Building cycle-augmented dataset -> ${DATASET}"
  python -u adv_graph_maker.py \
    --output-dir "${ADV_DIR}" \
    --variants "${CYCLE_VARIANT}"
elif [ -f "${DATASET}" ]; then
  echo "[step 1] Reusing existing ${DATASET}"
else
  echo "[step 1] ERROR: dataset not found: ${DATASET}" >&2
  echo "         Build it locally and upload it, e.g.:" >&2
  echo "           python adv_graph_maker.py --variants ${CYCLE_VARIANT}" >&2
  echo "           scp -r ${ADV_DIR} cluster:${SLURM_SUBMIT_DIR}/" >&2
  echo "         Or re-submit with --export=ALL,BUILD_DATASET=1 if the" >&2
  echo "         raw SIMULATIONS/ tree is present on this machine." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: pretrain UAE on our data
#
# Also needs the raw SIMULATIONS/ folder. Run locally and upload the
# resulting checkpoint before submitting.
# ---------------------------------------------------------------------------
if [ "${PRETRAIN_UAE}" = "1" ]; then
  echo "[step 2] Pretraining UAE -> ${UAE_CKPT}"
  python -u ct_uae_pretrain.py \
    --epochs "${UAE_EPOCHS}" \
    --feature-size "${UAE_FEATURE_SIZE}" \
    --num-layers "${UAE_LAYERS}" \
    --seed "${SEED}"
elif [ -f "${UAE_CKPT}" ]; then
  echo "[step 2] Reusing existing ${UAE_CKPT}"
else
  echo "[step 2] ERROR: UAE checkpoint not found: ${UAE_CKPT}" >&2
  echo "         Pretrain it locally and upload it, e.g.:" >&2
  echo "           python ct_uae_pretrain.py --epochs ${UAE_EPOCHS}" >&2
  echo "           scp -r uae_embeddings cluster:${SLURM_SUBMIT_DIR}/" >&2
  echo "         Or re-submit with --export=ALL,PRETRAIN_UAE=1 if the" >&2
  echo "         raw SIMULATIONS/ tree is present on this machine." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: train the UAE-fronted GNN and dump per-epoch MAE curves to JSON
# ---------------------------------------------------------------------------
echo "[step 3] Training UAE-fronted ${INNER} GNN -> ${OUTPUT_JSON}"
python -u train_uae_single.py \
  --dataset "${DATASET}" \
  --uae-ckpt "${UAE_CKPT}" \
  --uae-emb-dim "${EMB_DIM}" \
  --inner "${INNER}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --metric "${METRIC}" \
  --seed "${SEED}" \
  --output "${OUTPUT_JSON}" \
  ${BIDIR_FLAG} \
  ${FREEZE_FLAG} \
  ${NO_PRETRAINED_FLAG} \
  ${KEEP_TYPE_SCALAR_FLAG}
