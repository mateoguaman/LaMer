#!/usr/bin/env bash
set -euo pipefail

######################
### Config ###########
######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAMER_DIR="${LAMER_DIR:-${DEFAULT_LAMER_DIR}}"

if [ -f "${LAMER_DIR}/.env.language_table_local" ]; then
    # shellcheck disable=SC1091
    source "${LAMER_DIR}/.env.language_table_local"
fi
if [ -f "${LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${LAMER_DIR}/.env.language_table.secrets"
fi

LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/../language-table" && pwd)}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-${LANGTABLE_DIR}/ltvenv}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$HOME/data/verl-agent/text/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$HOME/data/verl-agent/text/test.parquet}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b_local}"
REWARD_TYPE="${REWARD_TYPE:-block2block}"
VLA_POLICY="${VLA_POLICY:-smolvla}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gigpo}"
if [ -z "${CHECKPOINT_ROOT}" ]; then
    echo "ERROR: CHECKPOINT_ROOT is not set."
    echo "Set CHECKPOINT_ROOT in ${LAMER_DIR}/.env.language_table or export it before running."
    exit 1
fi

RUN_ID="${RUN_ID:-local-$(date +%Y%m%d-%H%M%S)}"
TRAINER_LOCAL_DIR="${TRAINER_LOCAL_DIR:-${CHECKPOINT_ROOT}/${RUN_NAME}/${RUN_ID}}"
RUN_LOG_PATH="${RUN_LOG_PATH:-${TRAINER_LOCAL_DIR}/train-${RUN_ID}.log}"

# Local GPU assignment on this workstation:
#   0 = NVIDIA GeForce RTX 2080 Ti  -> Language Table / SmolVLA env servers
#   1 = NVIDIA GeForce RTX 3080     -> LaMer / Qwen trainer
SMOLVLA_GPU="${SMOLVLA_GPU:-0}"
QWEN_GPU="${QWEN_GPU:-1}"

BASE_PORT="${BASE_PORT:-$((50000 + (RANDOM % 2000) * 2))}"
TRAIN_PORT="${TRAIN_PORT:-${BASE_PORT}}"
VAL_PORT="${VAL_PORT:-$((BASE_PORT + 1))}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4}"
VAL_NUM_ENVS="${VAL_NUM_ENVS:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BLOCK_MODE="${TRAIN_BLOCK_MODE:-${BLOCK_MODE:-BLOCK_4}}"
VAL_BLOCK_MODE="${VAL_BLOCK_MODE:-${BLOCK_MODE:-BLOCK_4}}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-5}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-2}"
MAX_TURNS="${MAX_TURNS:-2}"

# Multistep task configuration (only used when REWARD_TYPE=multistep)
TRAIN_TASK_LOCATIONS="${TRAIN_TASK_LOCATIONS:-}"
TRAIN_TASK_SHAPES="${TRAIN_TASK_SHAPES:-}"
TRAIN_TASK_COLORS="${TRAIN_TASK_COLORS:-}"
TRAIN_TASK_N_STEPS="${TRAIN_TASK_N_STEPS:-2}"
VAL_TASK_LOCATIONS="${VAL_TASK_LOCATIONS:-}"
VAL_TASK_SHAPES="${VAL_TASK_SHAPES:-}"
VAL_TASK_COLORS="${VAL_TASK_COLORS:-}"
VAL_TASK_N_STEPS="${VAL_TASK_N_STEPS:-3}"

# VLA (inner-loop policy). SmolVLA checkpoints may be local paths or HF repo IDs.
VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"
SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-Sidharth-R/langtable-smolvla-finetuned}"
if [ -z "${VLA_CHECKPOINT}" ] && [ "${VLA_POLICY}" = "smolvla" ]; then
    VLA_CHECKPOINT="${SMOLVLA_CHECKPOINT}"
elif [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi
if [ "${VLA_POLICY}" = "smolvla" ] && [[ "${VLA_CHECKPOINT}" == *bc_resnet_sim_checkpoint_* ]]; then
    echo "WARNING: VLA_CHECKPOINT looks like a LAVA checkpoint; using SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
    VLA_CHECKPOINT="${SMOLVLA_CHECKPOINT}"
fi

# LAVA-only preprocessing mode for _build_batch.
PREPROCESS_MODE="${PREPROCESS_MODE:-jax_gpu}"

######################
### Setup ############
######################
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX}")" != "${LAMER_CONDA_ENV}" ]; then
        set +u
        conda activate "${LAMER_CONDA_ENV}"
        set -u
    fi
else
    echo "WARNING: conda not found; using current Python environment for LaMer."
fi

if [ -n "${LANGTABLE_PYTHON:-}" ]; then
    :
elif [ -x "${LANGTABLE_CONDA_ENV}/bin/python" ]; then
    LANGTABLE_PYTHON="${LANGTABLE_CONDA_ENV}/bin/python"
elif command -v conda >/dev/null 2>&1; then
    if [[ "${LANGTABLE_CONDA_ENV}" == */* ]]; then
        LANGTABLE_PYTHON="$(conda run -p "${LANGTABLE_CONDA_ENV}" which python)"
    else
        LANGTABLE_PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python)"
    fi
else
    echo "ERROR: Could not resolve LANGTABLE_PYTHON."
    echo "Set LANGTABLE_PYTHON or LANGTABLE_CONDA_ENV to the language-table Python."
    exit 1
fi

echo "Environment activated!"
echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "LAMER_DIR: ${LAMER_DIR}"
echo "LANGTABLE_DIR: ${LANGTABLE_DIR}"
echo "LANGTABLE_PYTHON: ${LANGTABLE_PYTHON}"
echo "TRAINER_LOCAL_DIR: ${TRAINER_LOCAL_DIR}"
echo "REWARD_TYPE: ${REWARD_TYPE}"
echo "VLA_POLICY: ${VLA_POLICY}"
echo "VLA_CHECKPOINT: ${VLA_CHECKPOINT:-<none>}"
echo "SMOLVLA_GPU: ${SMOLVLA_GPU}"
echo "QWEN_GPU: ${QWEN_GPU}"
echo "TRAIN_PORT: ${TRAIN_PORT}"
echo "VAL_PORT: ${VAL_PORT}"

######################
### Environment ######
######################
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR
export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"
mkdir -p "${TRAINER_LOCAL_DIR}" "$(dirname "${RUN_LOG_PATH}")"
[ -n "${CONDA_PKGS_DIRS:-}" ] && mkdir -p "${CONDA_PKGS_DIRS}"
[ -n "${PIP_CACHE_DIR:-}" ] && mkdir -p "${PIP_CACHE_DIR}"
[ -n "${TMPDIR:-}" ] && mkdir -p "${TMPDIR}"
[ -n "${XDG_CACHE_HOME:-}" ] && mkdir -p "${XDG_CACHE_HOME}"
[ -n "${WANDB_DATA_DIR:-}" ] && mkdir -p "${WANDB_DATA_DIR}"

########################################
### Step 0: Download LAVA checkpoint ###
########################################
if [ "${VLA_POLICY}" = "lava" ] && [ -n "${VLA_CHECKPOINT}" ] && [ ! -f "${VLA_CHECKPOINT}" ]; then
    echo "=== Downloading LAVA checkpoint ==="
    mkdir -p "${VLA_CHECKPOINT_DIR}"
    wget -q --show-progress -O "${VLA_CHECKPOINT}" \
        "https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000"
    echo "Downloaded to ${VLA_CHECKPOINT}"
fi

########################################
### Step 1: Start env servers ##########
########################################
echo ""
echo "=== Starting Language Table env servers on 2080 GPU (${SMOLVLA_GPU}) ==="

VLA_FLAG=""
case "${VLA_POLICY}" in
    smolvla|lava)
        VLA_FLAG="--policy ${VLA_POLICY}"
        if [ -n "${VLA_CHECKPOINT}" ]; then
            VLA_FLAG+=" --vla_checkpoint ${VLA_CHECKPOINT}"
            echo "VLA enabled (${VLA_POLICY}): ${VLA_CHECKPOINT}"
        else
            echo "WARNING: No VLA checkpoint set; ${VLA_POLICY} manager will use random inner-loop actions"
        fi
        ;;
    gemini)
        VLA_FLAG="--policy gemini"
        echo "Using Gemini policy for environment actions"
        ;;
    *)
        echo "ERROR: unsupported VLA_POLICY='${VLA_POLICY}' (expected smolvla, lava, or gemini)"
        exit 1
        ;;
esac

PREPROCESS_FLAG=""
if [ "${VLA_POLICY}" = "lava" ]; then
    PREPROCESS_FLAG="--preprocess_mode ${PREPROCESS_MODE}"
fi

TRAIN_TASK_FLAGS=""
if [ "${REWARD_TYPE}" = "multistep" ]; then
    [ -n "${TRAIN_TASK_LOCATIONS}" ] && TRAIN_TASK_FLAGS+=" --task_locations ${TRAIN_TASK_LOCATIONS}"
    [ -n "${TRAIN_TASK_SHAPES}" ] && TRAIN_TASK_FLAGS+=" --task_shapes ${TRAIN_TASK_SHAPES}"
    [ -n "${TRAIN_TASK_COLORS}" ] && TRAIN_TASK_FLAGS+=" --task_colors ${TRAIN_TASK_COLORS}"
    TRAIN_TASK_FLAGS+=" --task_n_steps ${TRAIN_TASK_N_STEPS}"
fi
VAL_TASK_FLAGS=""
if [ "${REWARD_TYPE}" = "multistep" ]; then
    [ -n "${VAL_TASK_LOCATIONS}" ] && VAL_TASK_FLAGS+=" --task_locations ${VAL_TASK_LOCATIONS}"
    [ -n "${VAL_TASK_SHAPES}" ] && VAL_TASK_FLAGS+=" --task_shapes ${VAL_TASK_SHAPES}"
    [ -n "${VAL_TASK_COLORS}" ] && VAL_TASK_FLAGS+=" --task_colors ${VAL_TASK_COLORS}"
    VAL_TASK_FLAGS+=" --task_n_steps ${VAL_TASK_N_STEPS}"
fi

TRAIN_WORKERS=$((TRAIN_NUM_ENVS * GROUP_SIZE))
echo "Starting train server on port ${TRAIN_PORT} (${TRAIN_NUM_ENVS}x${GROUP_SIZE}=${TRAIN_WORKERS} workers, GPU ${SMOLVLA_GPU}, blocks ${TRAIN_BLOCK_MODE})..."
CUDA_VISIBLE_DEVICES=${SMOLVLA_GPU} \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
"${LANGTABLE_PYTHON}" -m language_table.lamer.server_main \
    --host 0.0.0.0 --port "${TRAIN_PORT}" \
    --num_envs "${TRAIN_NUM_ENVS}" --group_n "${GROUP_SIZE}" \
    --block_mode "${TRAIN_BLOCK_MODE}" \
    --max_inner_steps "${MAX_INNER_STEPS}" --num_attempts "${NUM_ATTEMPTS}" \
    --max_turns "${MAX_TURNS}" --do_reflection \
    --reward_type "${REWARD_TYPE}" \
    ${PREPROCESS_FLAG} \
    --split train \
    ${TRAIN_TASK_FLAGS} \
    ${VLA_FLAG} \
    &
TRAIN_SERVER_PID=$!
echo "Train server PID: ${TRAIN_SERVER_PID}"

echo "Starting validation server on port ${VAL_PORT} (${VAL_NUM_ENVS} envs, GPU ${SMOLVLA_GPU}, blocks ${VAL_BLOCK_MODE})..."
CUDA_VISIBLE_DEVICES=${SMOLVLA_GPU} \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
"${LANGTABLE_PYTHON}" -m language_table.lamer.server_main \
    --host 0.0.0.0 --port "${VAL_PORT}" \
    --num_envs "${VAL_NUM_ENVS}" --group_n 1 \
    --block_mode "${VAL_BLOCK_MODE}" \
    --max_inner_steps "${MAX_INNER_STEPS}" --num_attempts "${NUM_ATTEMPTS}" \
    --max_turns "${MAX_TURNS}" --do_reflection \
    --reward_type "${REWARD_TYPE}" \
    ${PREPROCESS_FLAG} \
    --split val \
    ${VAL_TASK_FLAGS} \
    ${VLA_FLAG} \
    &
VAL_SERVER_PID=$!
echo "Validation server PID: ${VAL_SERVER_PID}"

RECEIVED_SIGNAL=""
MEM_MONITOR_PID=""

handle_signal() {
    RECEIVED_SIGNAL="$1"
    echo "[$(date)] Received signal: ${RECEIVED_SIGNAL}"
    free -h 2>/dev/null || true
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
}

cleanup() {
    local exit_code=$?
    echo "[$(date)] cleanup() called; exit_code=${exit_code}, signal=${RECEIVED_SIGNAL:-none}"
    [ -n "${MEM_MONITOR_PID}" ] && kill "${MEM_MONITOR_PID}" 2>/dev/null || true
    echo "Cleaning up env servers..."
    kill "${TRAIN_SERVER_PID}" 2>/dev/null || true
    kill "${VAL_SERVER_PID}" 2>/dev/null || true
    wait "${TRAIN_SERVER_PID}" 2>/dev/null || true
    wait "${VAL_SERVER_PID}" 2>/dev/null || true
    echo "Servers stopped."
    if [ -n "${RECEIVED_SIGNAL}" ]; then
        exit 1
    fi
}

trap 'handle_signal SIGTERM' TERM
trap 'handle_signal SIGINT' INT
trap cleanup EXIT

########################################
### Step 2: Wait for servers ###########
########################################
echo ""
echo "=== Waiting for servers to become ready ==="

wait_for_port() {
    local port=$1
    local name=$2
    local max_attempts=120
    for i in $(seq 1 "${max_attempts}"); do
        if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
            echo "  ${name} server ready on port ${port} (attempt ${i})"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: ${name} server on port ${port} did not start after ${max_attempts} attempts"
    return 1
}

wait_for_port "${TRAIN_PORT}" "Training" || exit 1
wait_for_port "${VAL_PORT}" "Validation" || exit 1

########################################
### Step 3: Connection test ############
########################################
echo ""
echo "=== Running connection test ==="
"${LANGTABLE_PYTHON}" -m language_table.lamer.test_connection \
    --host 127.0.0.1 --port "${TRAIN_PORT}" --val_port "${VAL_PORT}" --full \
    --timeout 600
echo "Connection test passed!"

########################################
### Memory monitor (background) ########
########################################
monitor_memory() {
    while true; do
        echo "[$(date)] [mem-monitor] $(free -h 2>/dev/null | rg '^Mem' || echo 'N/A')"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null || true
        sleep 300
    done
}
monitor_memory &
MEM_MONITOR_PID=$!

########################################
### Step 4: LaMer training #############
########################################
echo ""
echo "=== Starting LaMer training on 3080 GPU (${QWEN_GPU}) ==="
export CUDA_VISIBLE_DEVICES="${QWEN_GPU}"
cd "${LAMER_DIR}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "${TRAIN_NUM_ENVS}" \
    --val_data_size "${VAL_NUM_ENVS}"

export TRAIN_DATA_PATH VAL_DATA_PATH
export TRAIN_NUM_ENVS VAL_NUM_ENVS GROUP_SIZE ADV_ESTIMATOR
export NUM_ATTEMPTS MAX_TURNS
export LEARNING_RATE BATCH_SIZE MICRO_BATCH_SIZE
export USE_KL_LOSS KL_LOSS_COEF KL_LOSS_TYPE
export USE_KL_IN_REWARD KL_REWARD_COEF
export RUN_NAME TRAINER_LOCAL_DIR RUN_LOG_PATH
export ENV_ADDRESS="127.0.0.1:${TRAIN_PORT}"
export VAL_ADDRESS="127.0.0.1:${VAL_PORT}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-1}"

set +e
bash "${LAMER_DIR}/examples/language_table/lamer_language_table_slurm.sh"
TRAIN_EXIT_CODE=$?
set -e

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] ERROR: Training exited with code ${TRAIN_EXIT_CODE}"
    free -h 2>/dev/null || true
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    exit ${TRAIN_EXIT_CODE}
fi

echo "END TIME: $(date)"
