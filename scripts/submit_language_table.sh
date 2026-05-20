#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAVA=false

for arg in "$@"; do
           if [ "$arg" = "--lava" ]; then
                           LAVA=true
                                   break
                                       fi
                               done

                               if [ "$LAVA" = true ]; then
                                          ENV_FILE="${BASE_LAMER_DIR}/.env.language_table_lava"
                                   else
                                               ENV_FILE="${BASE_LAMER_DIR}/.env.language_table_smolvla"
                               fi

                               if [ -f "$ENV_FILE" ]; then
                                           # shellcheck disable=SC1091
                                               source "$ENV_FILE"
                               fi

if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi

LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
SLURM_SCRIPT="${LAMER_DIR}/scripts/slurm/lamer_language_table.slurm"
DEFAULT_LANGTABLE_DIR="${LAMER_DIR}/../language-table"

export MAX_SEEDS="${MAX_SEEDS:-1}"
export LAMER_DIR="${LAMER_DIR}"
export LANGTABLE_DIR="${LANGTABLE_DIR:-${DEFAULT_LANGTABLE_DIR}}"
export LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-${LANGTABLE_DIR}/ltvenv}"
export LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
export CONDA_CACHE_DIR="${CONDA_CACHE_DIR:-}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_CACHE_DIR}}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-}"
export TMPDIR="${TMPDIR:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
export RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
export REWARD_TYPE="${REWARD_TYPE:-block2block}"
export TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4}"
export VAL_NUM_ENVS="${VAL_NUM_ENVS:-4}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export MAX_INNER_STEPS="${MAX_INNER_STEPS:-5}"
export NUM_ATTEMPTS="${NUM_ATTEMPTS:-2}"
export MAX_TURNS="${MAX_TURNS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
export KL_REWARD_COEF="${KL_REWARD_COEF:-0.001}"
export TRAIN_TASK_LOCATIONS="${TRAIN_TASK_LOCATIONS:-}"
export TRAIN_TASK_SHAPES="${TRAIN_TASK_SHAPES:-}"
export TRAIN_TASK_COLORS="${TRAIN_TASK_COLORS:-}"
export TRAIN_TASK_N_STEPS="${TRAIN_TASK_N_STEPS:-2}"
export VAL_TASK_LOCATIONS="${VAL_TASK_LOCATIONS:-}"
export VAL_TASK_SHAPES="${VAL_TASK_SHAPES:-}"
export VAL_TASK_COLORS="${VAL_TASK_COLORS:-}"
export VAL_TASK_N_STEPS="${VAL_TASK_N_STEPS:-3}"
export VLA_POLICY="${VLA_POLICY:-smolvla}"
export VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
export VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-}"
export HF_HOME="${HF_HOME:-}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-}"
export WANDB_USERNAME="${WANDB_USERNAME:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
export TILLICUM="${TILLICUM:-True}"
export REWARD_KWARGS="${REWARD_KWARGS:-}"

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

SBATCH_LOG_ARGS=()
if [ -n "${SLURM_LOG_DIR}" ]; then
    mkdir -p "${SLURM_LOG_DIR}"
    SBATCH_LOG_ARGS+=(--output "${SLURM_LOG_DIR}/%x-%j.out")
    SBATCH_LOG_ARGS+=(--error "${SLURM_LOG_DIR}/%x-%j.err")
fi

echo "Submitting ${SLURM_SCRIPT}"
echo "  LAMER_DIR=${LAMER_DIR}"
echo "  LANGTABLE_DIR=${LANGTABLE_DIR}"
echo "  LANGTABLE_CONDA_ENV=${LANGTABLE_CONDA_ENV}"
echo "  LAMER_CONDA_ENV=${LAMER_CONDA_ENV}"
echo "  CONDA_CACHE_DIR=${CONDA_CACHE_DIR:-<default>}"
echo "  CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-<default>}"
echo "  CONDA_ENVS_DIRS=${CONDA_ENVS_DIRS:-<default>}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR:-<default>}"
echo "  TMPDIR=${TMPDIR:-<default>}"
echo "  CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  REWARD_TYPE=${REWARD_TYPE}"
echo "  VLA_CHECKPOINT=${VLA_CHECKPOINT}"
echo "  SLURM_LOG_DIR=${SLURM_LOG_DIR:-<default: cwd>}"

exec sbatch "${SBATCH_LOG_ARGS[@]}" "$@" "${SLURM_SCRIPT}"
