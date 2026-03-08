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
export LANGTABLE_PYTHON="${LANGTABLE_PYTHON:-${LANGTABLE_DIR}/ltvenv/bin/python}"
export LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$HOME/data/verl-agent/text/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-$HOME/data/verl-agent/text/test.parquet}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH:-$HOME}/checkpoints/lamer}"
export RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
export TRAINER_LOCAL_DIR="${TRAINER_LOCAL_DIR:-${CHECKPOINT_ROOT}/${RUN_NAME}}"
export RUN_LOG_PATH="${RUN_LOG_PATH:-${LAMER_DIR}/../${RUN_NAME}.log}"
export VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-${SCRATCH:-$HOME}/checkpoints/language_table}"
export VLA_CHECKPOINT="${VLA_CHECKPOINT:-${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000}"
export HF_HOME="${HF_HOME:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
export WANDB_USERNAME="${WANDB_USERNAME:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
export HF_TOKEN="${HF_TOKEN:-}"

if [ -n "${HF_HOME}" ]; then
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export HUGGINGFACE_DATASETS_CACHE="${HUGGINGFACE_DATASETS_CACHE:-${HF_HOME}/datasets}"
fi
if [ -n "${HF_TOKEN_FILE}" ] && [ -z "${HF_TOKEN}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN="$(<"${HF_TOKEN_FILE}")"
fi
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
if [ -n "${WANDB_API_KEY_FILE}" ] && [ -z "${WANDB_API_KEY}" ] && [ -f "${WANDB_API_KEY_FILE}" ]; then
    WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
fi
if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY
fi

echo "Submitting ${SLURM_SCRIPT}"
echo "  LAMER_DIR=${LAMER_DIR}"
echo "  LANGTABLE_DIR=${LANGTABLE_DIR}"
echo "  LAMER_CONDA_ENV=${LAMER_CONDA_ENV}"
echo "  TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
echo "  VAL_DATA_PATH=${VAL_DATA_PATH}"
echo "  TRAINER_LOCAL_DIR=${TRAINER_LOCAL_DIR}"
echo "  VLA_CHECKPOINT=${VLA_CHECKPOINT}"
echo "  SETUP_SCRIPT=${SETUP_SCRIPT:-}"

exec sbatch "$@" "${SLURM_SCRIPT}"
