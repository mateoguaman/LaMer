#!/bin/bash
set -euo pipefail
set -x

# Benchmark Language Table remote rollouts with a VLA inner-loop policy.
# Starts the env server locally, then runs vla_rollout_benchmark.py.
#
# Common env vars:
#   POLICY              smolvla | lava (default: smolvla)
#   NUM_ENVS            vectorized env count (default: 8)
#   GROUP_N             repeated envs per sampled task seed (default: 1)
#   NUM_BATCHES         reset+rollout batches to benchmark (default: 3)
#   MAX_TURNS           outer-loop turns per batch (default: 1)
#   MAX_INNER_STEPS     VLA/env steps per outer turn (default: 50)
#   ENV_PORT            env server port (default: 50053)
#   ENV_SERVER_GPU      GPU visible to the env server (default: 0)
#   LANGTABLE_DIR       path to language-table repo (default: ~/projects/language-table)
#   LANGTABLE_PYTHON    python with language-table deps
#   LAMER_PYTHON        python with LaMer deps
#   VLA_CHECKPOINT      checkpoint for selected policy
#   SMOLVLA_CHECKPOINT  default SmolVLA checkpoint
#   LAVA_CHECKPOINT     default LAVA checkpoint path
#   PREPROCESS_MODE     LAVA preprocessing mode (default: batched_tf)
#   RUN_NAME            output/log suffix

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LANGTABLE_DIR="${LANGTABLE_DIR:-${HOME}/projects/language-table}"
LANGTABLE_PYTHON="${LANGTABLE_PYTHON:-/home/sidhraja/miniconda3/envs/ltvenv/bin/python}"
LAMER_PYTHON="${LAMER_PYTHON:-/home/sidhraja/miniconda3/envs/lamer/bin/python}"

POLICY="${POLICY:-smolvla}"
NUM_ENVS="${NUM_ENVS:-64}"
GROUP_N="${GROUP_N:-1}"
NUM_BATCHES="${NUM_BATCHES:-3}"
MAX_TURNS="${MAX_TURNS:-1}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-50}"
ENV_PORT="${ENV_PORT:-50053}"
ENV_SERVER_GPU="${ENV_SERVER_GPU:-0}"
RUN_NAME="${RUN_NAME:-vla_rollout_benchmark_${POLICY}_${NUM_ENVS}envs}"

BLOCK_MODE="${BLOCK_MODE:-BLOCK_4}"
REWARD_TYPE="${REWARD_TYPE:-block2block}"
SPLIT="${SPLIT:-val}"
GOAL_SOURCE="${GOAL_SOURCE:-task}"
PREPROCESS_MODE="${PREPROCESS_MODE:-batched_tf}"
BENCHMARK_TRACE_INNER_STEPS="${BENCHMARK_TRACE_INNER_STEPS:-0}"

SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-Sidharth-R/langtable-smolvla-finetuned}"
LAVA_CHECKPOINT="${LAVA_CHECKPOINT:-/home/sidhraja/projects/LaMer/checkpoints/bc_resnet_sim_checkpoint_955000}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

if [ "${POLICY}" = "smolvla" ] && [ -z "${VLA_CHECKPOINT}" ]; then
    VLA_CHECKPOINT="${SMOLVLA_CHECKPOINT}"
elif [ "${POLICY}" = "lava" ] && [ -z "${VLA_CHECKPOINT}" ]; then
    VLA_CHECKPOINT="${LAVA_CHECKPOINT}"
fi

if [ "${POLICY}" != "smolvla" ] && [ "${POLICY}" != "lava" ]; then
    echo "ERROR: POLICY must be 'smolvla' or 'lava', got '${POLICY}'" >&2
    exit 1
fi

if [ -z "${VLA_CHECKPOINT}" ]; then
    echo "ERROR: Set VLA_CHECKPOINT or the policy-specific checkpoint env var." >&2
    echo "       For lava, set LAVA_CHECKPOINT=/path/to/bc_resnet_sim_checkpoint_..." >&2
    exit 1
fi

OUTPUT_DIR="${LAMER_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_NAME}.jsonl"
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"

export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
# SmolVLA manager honors this and avoids returning video frames in benchmark runs.
export LAMER_MAX_VIDEO_ENVS="${LAMER_MAX_VIDEO_ENVS:-0}"

mkdir -p "${OUTPUT_DIR}"

ENV_SERVER_PID=""
cleanup() {
    echo "Cleaning up env server..."
    [ -n "${ENV_SERVER_PID}" ] && kill "${ENV_SERVER_PID}" 2>/dev/null || true
    [ -n "${ENV_SERVER_PID}" ] && wait "${ENV_SERVER_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

TRACE_FLAG=""
if [ "${BENCHMARK_TRACE_INNER_STEPS}" = "1" ]; then
    TRACE_FLAG="--benchmark_trace_inner_steps"
fi

echo ""
echo "=== Starting Language Table ${POLICY} env server on GPU ${ENV_SERVER_GPU}, port ${ENV_PORT} ==="
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}" \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port "${ENV_PORT}" \
    --num_envs "${NUM_ENVS}" --group_n "${GROUP_N}" \
    --max_inner_steps "${MAX_INNER_STEPS}" --num_attempts 1 \
    --max_turns "${MAX_TURNS}" \
    --block_mode "${BLOCK_MODE}" \
    --reward_type "${REWARD_TYPE}" \
    --split "${SPLIT}" \
    --policy "${POLICY}" \
    --vla_checkpoint "${VLA_CHECKPOINT}" \
    --preprocess_mode "${PREPROCESS_MODE}" \
    --chunk_size 10
    ${TRACE_FLAG} \
    > >(tee -a "${LOG_FILE}") 2>&1 \
    &
ENV_SERVER_PID=$!
echo "Env server PID: ${ENV_SERVER_PID}"

echo ""
echo "=== Waiting for env server to become ready ==="
max_attempts="${SERVER_READY_ATTEMPTS:-180}"
for i in $(seq 1 "${max_attempts}"); do
    if nc -z 127.0.0.1 "${ENV_PORT}" 2>/dev/null; then
        echo "  Env server ready on port ${ENV_PORT} (attempt ${i})"
        break
    fi
    if [ "${i}" -eq "${max_attempts}" ]; then
        echo "  ERROR: Env server on port ${ENV_PORT} did not start" >&2
        exit 1
    fi
    sleep 2
done

echo ""
echo "=== Running VLA rollout benchmark ==="
cd "${LAMER_DIR}"

${LAMER_PYTHON} examples/language_table/vla_rollout_benchmark.py \
    --remote_address "127.0.0.1:${ENV_PORT}" \
    --num_batches "${NUM_BATCHES}" \
    --max_turns "${MAX_TURNS}" \
    --goal_source "${GOAL_SOURCE}" \
    --expected_num_envs "$((NUM_ENVS * GROUP_N))" \
    --output "${OUTPUT_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "END TIME: $(date)"
echo "Results: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
