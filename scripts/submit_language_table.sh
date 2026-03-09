#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${BASE_LAMER_DIR}/.env.language_table" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table"
fi
if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi

LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
SLURM_SCRIPT="${LAMER_DIR}/scripts/slurm/lamer_language_table.slurm"
DEFAULT_LANGTABLE_DIR="${LAMER_DIR}/../language-table"

export LAMER_DIR="${LAMER_DIR}"
export LANGTABLE_DIR="${LANGTABLE_DIR:-${DEFAULT_LANGTABLE_DIR}}"
export LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-${LANGTABLE_DIR}/ltvenv}"
export LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
export RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
export VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
export VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"
export HF_HOME="${HF_HOME:-}"
export WANDB_USERNAME="${WANDB_USERNAME:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

if [ -z "${CHECKPOINT_ROOT}" ]; then
    echo "ERROR: CHECKPOINT_ROOT is not set."
    echo "Set it in .env.language_table before submitting."
    exit 1
fi
if [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    export VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi

if [ -n "${HF_HOME}" ]; then
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export HUGGINGFACE_DATASETS_CACHE="${HUGGINGFACE_DATASETS_CACHE:-${HF_HOME}/datasets}"
fi
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY
fi

echo "Submitting ${SLURM_SCRIPT}"
echo "  LAMER_DIR=${LAMER_DIR}"
echo "  LANGTABLE_DIR=${LANGTABLE_DIR}"
echo "  LANGTABLE_CONDA_ENV=${LANGTABLE_CONDA_ENV}"
echo "  LAMER_CONDA_ENV=${LAMER_CONDA_ENV}"
echo "  CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  VLA_CHECKPOINT=${VLA_CHECKPOINT}"

exec sbatch "$@" "${SLURM_SCRIPT}"
