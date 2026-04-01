#!/usr/bin/env bash
# Submit the benchmark SLURM job.
# Sources .env.language_table for cluster-specific config (same pattern as
# scripts/submit_language_table.sh).
#
# Usage:
#   bash benchmarks/submit_benchmarks.sh
#   BENCHMARKS=benchmark_end_to_end bash benchmarks/submit_benchmarks.sh
#   BENCH_END_TO_END_PRESET=resolved_training_config BENCHMARKS=benchmark_end_to_end bash benchmarks/submit_benchmarks.sh
#   BENCHMARKS=sharded_smoke bash benchmarks/submit_benchmarks.sh
#   BENCHMARKS=benchmark_end_to_end,vla bash benchmarks/submit_benchmarks.sh -- --time=6:00:00
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
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$HOME/data/verl-agent/text/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-$HOME/data/verl-agent/text/test.parquet}"
export BENCHMARKS="${BENCHMARKS:-benchmark_end_to_end}"
export BENCH_END_TO_END_PRESET="${BENCH_END_TO_END_PRESET:-doc_baseline}"
export BENCH_WARMUP_ITERATIONS="${BENCH_WARMUP_ITERATIONS:-2}"
export BENCH_MEASURED_ITERATIONS="${BENCH_MEASURED_ITERATIONS:-3}"
export BENCH_TRACE_INNER_STEPS="${BENCH_TRACE_INNER_STEPS:-0}"
export BENCH_TRAIN_SHARD_COUNTS="${BENCH_TRAIN_SHARD_COUNTS:-4,8}"
export BENCH_PROCS_PER_GPU="${BENCH_PROCS_PER_GPU:-2}"
export BENCH_NO_SHM_RGB="${BENCH_NO_SHM_RGB:-0}"
export BENCH_ENV_SERVER_GPUS="${BENCH_ENV_SERVER_GPUS:-${BENCH_ENV_SERVER_GPU:-4,5,6,7}}"
export BENCH_SKIP_VAL_SERVER="${BENCH_SKIP_VAL_SERVER:-1}"
export BENCH_TRAINER_VISIBLE_GPUS="${BENCH_TRAINER_VISIBLE_GPUS:-0,1,2,3}"
export BENCH_RAY_NUM_CPUS="${BENCH_RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
export BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-}"
export VLA_PREPROCESS_MODES="${VLA_PREPROCESS_MODES:-original}"
export PREPROCESS_MODE="${PREPROCESS_MODE:-jax_gpu}"
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
echo "  TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
echo "  VAL_DATA_PATH=${VAL_DATA_PATH}"
echo "  BENCHMARKS=${BENCHMARKS}"
echo "  BENCH_END_TO_END_PRESET=${BENCH_END_TO_END_PRESET}"
echo "  BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}"
echo "  BENCH_MEASURED_ITERATIONS=${BENCH_MEASURED_ITERATIONS}"
echo "  BENCH_TRAIN_SHARD_COUNTS=${BENCH_TRAIN_SHARD_COUNTS}"
echo "  BENCH_PROCS_PER_GPU=${BENCH_PROCS_PER_GPU}"
echo "  BENCH_NO_SHM_RGB=${BENCH_NO_SHM_RGB}"
echo "  BENCH_ENV_SERVER_GPUS=${BENCH_ENV_SERVER_GPUS}"
echo "  BENCH_SKIP_VAL_SERVER=${BENCH_SKIP_VAL_SERVER}"
echo "  BENCH_TRAINER_VISIBLE_GPUS=${BENCH_TRAINER_VISIBLE_GPUS}"
echo "  BENCH_OUTPUT_DIR=${BENCH_OUTPUT_DIR:-<default>}"
echo "  VLA_PREPROCESS_MODES=${VLA_PREPROCESS_MODES}"
echo "  PREPROCESS_MODE=${PREPROCESS_MODE}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SLURM_LOG_DIR=${SLURM_LOG_DIR:-<default: cwd>}"

exec sbatch "${SBATCH_LOG_ARGS[@]}" "$@" "${SLURM_SCRIPT}"
