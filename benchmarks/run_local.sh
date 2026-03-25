#!/bin/bash
# Run individual benchmarks locally (without SLURM).
#
# Usage:
#   # Run all benchmarks:
#   bash benchmarks/run_local.sh all
#
#   # Run individual:
#   bash benchmarks/run_local.sh vla
#   bash benchmarks/run_local.sh envs
#   bash benchmarks/run_local.sh outer
#
# Required env vars:
#   VLA_CHECKPOINT_DIR  — directory with bc_resnet_sim_checkpoint_955000
#
# Optional env vars:
#   LANGTABLE_DIR       — path to language-table repo (default: ../language-table)
#   LANGTABLE_PYTHON    — python binary for ltvenv (default: auto-detect)
#   MODEL_PATH          — HF model path (default: Qwen/Qwen3-4B)
#   ENV_GPU             — GPU for VLA/env benchmarks (default: 0)
#   TRAIN_GPU           — GPU for outer-loop benchmark (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/../language-table" && pwd)}"
LANGTABLE_PYTHON="${LANGTABLE_PYTHON:-$(conda run -n ltvenv which python 2>/dev/null || echo "")}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
ENV_GPU="${ENV_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"

# Source .env.language_table if present (same pattern as submit_language_table.sh)
if [ -f "${LAMER_DIR}/.env.language_table" ]; then
    # shellcheck disable=SC1091
    source "${LAMER_DIR}/.env.language_table"
fi

VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

if [ -z "${VLA_CHECKPOINT_DIR}" ]; then
    echo "ERROR: VLA_CHECKPOINT_DIR is not set."
    echo "Set it in .env.language_table or export it before running."
    exit 1
fi

export PYTHONPATH="${LANGTABLE_DIR}:${LAMER_DIR}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2

BENCH="${1:-all}"

echo "============================================"
echo "LaMer Benchmark Suite (local)"
echo "============================================"
echo "LANGTABLE_DIR: ${LANGTABLE_DIR}"
echo "VLA_CHECKPOINT_DIR: ${VLA_CHECKPOINT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "ENV_GPU: ${ENV_GPU}, TRAIN_GPU: ${TRAIN_GPU}"
echo ""

if [ "${BENCH}" = "vla" ] || [ "${BENCH}" = "all" ]; then
    echo ""
    echo "=== VLA Inference Benchmark (GPU ${ENV_GPU}) ==="
    CUDA_VISIBLE_DEVICES=${ENV_GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ${LANGTABLE_PYTHON} -m language_table.lamer.benchmark_vla_standalone \
        --checkpoint_dir "${VLA_CHECKPOINT_DIR}" \
        --batch_sizes "1,2,4,8,16,32,64,128,256,512,1024" \
        --num_warmup 3 --num_iters 20
fi

if [ "${BENCH}" = "envs" ] || [ "${BENCH}" = "all" ]; then
    echo ""
    echo "=== Environment Scaling Benchmark (GPU ${ENV_GPU}) ==="
    CUDA_VISIBLE_DEVICES=${ENV_GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    ${LANGTABLE_PYTHON} -m language_table.lamer.benchmark_envs_standalone \
        --checkpoint_dir "${VLA_CHECKPOINT_DIR}" \
        --env_counts "4,8,16,32,64,128,256" \
        --inner_steps 20 \
        --cpus_per_gpu 8 \
        --time_budget_s 30
fi

if [ "${BENCH}" = "outer" ] || [ "${BENCH}" = "all" ]; then
    echo ""
    echo "=== Outer-Loop LLM Benchmark (GPU ${TRAIN_GPU}) ==="
    cd "${LAMER_DIR}"
    CUDA_VISIBLE_DEVICES=${TRAIN_GPU} \
    python -m benchmarks.benchmark_outer_loop \
        --model "${MODEL_PATH}" \
        --max_prompt_length 2048 \
        --max_response_length 1024 \
        --batch_sizes "1,2,4,8,16,32,64" \
        --lora_ranks "8,16,32,64" \
        --num_warmup 2 --num_iters 5
fi

echo ""
echo "=== Done ==="
