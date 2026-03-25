#!/usr/bin/env bash
# Submit the benchmark SLURM job.
# Sources .env.language_table for cluster-specific config (same pattern as
# scripts/submit_language_table.sh).
#
# Usage:
#   bash benchmarks/submit_benchmarks.sh
#   BENCHMARKS=vla bash benchmarks/submit_benchmarks.sh
#   BENCHMARKS=outer bash benchmarks/submit_benchmarks.sh -- --time=1:00:00
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

export LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
export LANGTABLE_DIR="${LANGTABLE_DIR:-${LAMER_DIR}/../language-table}"
export LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
export LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
export VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
export VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"
export BENCHMARKS="${BENCHMARKS:-vla,envs,outer}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-}"

# Same checkpoint derivation as submit_language_table.sh
if [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    export VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi

if [ -z "${VLA_CHECKPOINT_DIR}" ]; then
    echo "ERROR: VLA_CHECKPOINT_DIR is not set."
    echo "Set it in .env.language_table or export it before submitting."
    exit 1
fi

# Export cache/tmp dirs so the SLURM script inherits them
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-}"
export TMPDIR="${TMPDIR:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-}"
export HF_HOME="${HF_HOME:-}"
if [ -n "${HF_HOME}" ]; then
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

SLURM_SCRIPT="${SCRIPT_DIR}/run_benchmarks.slurm"

# Log directory — same pattern as submit_language_table.sh
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
echo "  VLA_CHECKPOINT_DIR=${VLA_CHECKPOINT_DIR}"
echo "  VLA_CHECKPOINT=${VLA_CHECKPOINT}"
echo "  BENCHMARKS=${BENCHMARKS}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SLURM_LOG_DIR=${SLURM_LOG_DIR:-<default: cwd>}"

exec sbatch "${SBATCH_LOG_ARGS[@]}" "$@" "${SLURM_SCRIPT}"
