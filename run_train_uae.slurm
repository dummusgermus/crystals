#!/bin/bash
#SBATCH --job-name=jara-uae-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Train the ct-UAE-fronted gated GNN with the winning configuration on the
# cluster.  The cycle-augmented dataset and the UAE checkpoint must already
# exist in the working directory; both depend on the raw SIMULATIONS/ tree
# and are therefore built locally with:
#
#   python adv_graph_maker.py        # -> adv_datasets_uae/cycle34_dataset.pt
#   python ct_uae_pretrain.py        # -> uae_embeddings/uae_emb8.pt
#
# Then upload both before submitting:
#
#   scp -r adv_datasets_uae uae_embeddings <user>@<host>:<repo>/

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
# Configuration
# ---------------------------------------------------------------------------
DATASET="adv_datasets_uae/cycle34_dataset.pt"
UAE_CKPT="uae_embeddings/uae_emb8.pt"
OUTPUT_JSON="uae_gated_result.json"

if [ ! -f "${DATASET}" ]; then
  echo "[error] dataset not found: ${DATASET}" >&2
  echo "        Build it locally with: python adv_graph_maker.py" >&2
  echo "        then upload adv_datasets_uae/ to this machine." >&2
  exit 1
fi

if [ ! -f "${UAE_CKPT}" ]; then
  echo "[error] UAE checkpoint not found: ${UAE_CKPT}" >&2
  echo "        Pretrain it locally with: python ct_uae_pretrain.py" >&2
  echo "        then upload uae_embeddings/ to this machine." >&2
  exit 1
fi

echo "[train] UAE-fronted gated GNN -> ${OUTPUT_JSON}"
python -u train_uae_single.py \
  --dataset "${DATASET}" \
  --uae-ckpt "${UAE_CKPT}" \
  --output "${OUTPUT_JSON}"
