#!/bin/bash
set -euo pipefail
set -x

# Standalone Language Table evaluation using an external vLLM server.
# Starts the env server locally, then runs api_rollout.py pointing at a remote vLLM.
#
# Required env vars:
#   VLLM_URL             vLLM base URL (default: http://localhost:8000/v1)
#   MODEL                model name as served by vLLM (default: Qwen/Qwen3-4B)
#   NUM_EPISODES         number of episodes to run (default: 10)
#   NUM_ENVS             parallel environments per batch (default: 4)
#   MAX_TURNS            max turns per episode (default: 5)
#   ENV_PORT             env server port (default: 50053)
#   LANGTABLE_DIR        path to language-table repo (default: ~/projects/language-table)
#   LANGTABLE_PYTHON     python binary with language-table deps (default: ltvenv/bin/python)
#   SMOLVLA_CHECKPOINT   HF repo id or local path (default: Sidharth-R/langtable-smolvla-finetuned)
#   RUN_NAME             for output naming (default: api_rollout)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LANGTABLE_DIR="${LANGTABLE_DIR:-${HOME}/projects/language-table}"
LANGTABLE_PYTHON="${LANGTABLE_PYTHON:-/home/sidhraja/miniconda3/envs/ltvenv/bin/python}"
SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-Sidharth-R/langtable-smolvla-finetuned}"
RUN_NAME="${RUN_NAME:-api_rollout}"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-50}"
ENV_PORT="${ENV_PORT:-50063}"
ENV_SERVER_GPU="${ENV_SERVER_GPU:-1}"

VAL_DATA_PATH="${VAL_DATA_PATH:-${HOME}/data/verl-agent/text/test.parquet}"
OUTPUT_DIR="${LAMER_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/api_rollout_${RUN_NAME}.jsonl"
VIDEO_DIR="${OUTPUT_DIR}/api_rollout_${RUN_NAME}_videos"

export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1

LOG_FILE="${OUTPUT_DIR}/api_rollout_${RUN_NAME}.log"

mkdir -p "${OUTPUT_DIR}"

######################
### Server cleanup ###
######################
ENV_SERVER_PID=""

cleanup() {
    echo "Cleaning up env server..."
    [ -n "${ENV_SERVER_PID}" ] && kill "${ENV_SERVER_PID}" 2>/dev/null || true
    [ -n "${ENV_SERVER_PID}" ] && wait "${ENV_SERVER_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

########################################
### Step 1: Generate placeholder data ##
########################################
if [ ! -f "${VAL_DATA_PATH}" ]; then
    echo "=== Generating placeholder parquet ==="
    cd "${LAMER_DIR}"
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --train_data_size 1 \
        --val_data_size "${NUM_EPISODES}"
fi

########################################
### Step 2: Start env server ###########
########################################
echo ""
echo "=== Starting Language Table env server on GPU ${ENV_SERVER_GPU}, port ${ENV_PORT} (${NUM_ENVS} envs) ==="
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${ENV_PORT} \
    --num_envs ${NUM_ENVS} --group_n ${NUM_ENVS} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts 1 \
    --max_turns ${MAX_TURNS} \
    --reward_type tetris_shape \
    --split val \
    --policy smolvla \
    --vla_checkpoint "${SMOLVLA_CHECKPOINT}" \
    --chunk_size 10 \
    > >(tee -a "${LOG_FILE}") 2>&1 \
    &
ENV_SERVER_PID=$!
echo "Env server PID: ${ENV_SERVER_PID}"

########################################
### Step 3: Wait for server ############
########################################
echo ""
echo "=== Waiting for env server to become ready ==="
max_attempts=120  # 4 min max
for i in $(seq 1 ${max_attempts}); do
    if nc -z 127.0.0.1 ${ENV_PORT} 2>/dev/null; then
        echo "  Env server ready on port ${ENV_PORT} (attempt ${i})"
        break
    fi
    if [ "${i}" -eq "${max_attempts}" ]; then
        echo "  ERROR: Env server on port ${ENV_PORT} did not start"
        exit 1
    fi
    sleep 2
done

########################################
### Step 4: Run API rollout ############
########################################
echo ""
echo "=== Running API rollout ==="
cd "${LAMER_DIR}"

/home/sidhraja/miniconda3/envs/lamer/bin/python examples/language_table/api_rollout.py \
    --remote_address "localhost:${ENV_PORT}" \
    --vllm_url "${VLLM_URL}" \
    --model "${MODEL}" \
    --val_data "${VAL_DATA_PATH}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_envs "${NUM_ENVS}" \
    --max_turns "${MAX_TURNS}" \
    --output "${OUTPUT_FILE}" \
    --video_dir "${VIDEO_DIR}" \
    --human
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "END TIME: $(date)"
echo "Results: ${OUTPUT_FILE}"
