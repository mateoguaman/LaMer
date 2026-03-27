#!/bin/bash
# SageMaker entrypoint for LaMer + language-table training.
# Adapted from scripts/slurm/lamer_language_table.slurm.
#
# Runs both env servers (language-table, GPU 0) and LaMer training (GPUs 1-4)
# on the same instance, communicating via localhost.
set -euo pipefail

######################
### Config ###########
######################
LAMER_DIR="/opt/ml/code"
LANGTABLE_DIR="/opt/language-table"
LANGTABLE_PYTHON="/opt/miniforge3/envs/ltvenv/bin/python"
LAMER_PYTHON="/opt/miniforge3/envs/lamer/bin/python"

RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

# SageMaker paths
TRAINER_LOCAL_DIR="/opt/ml/checkpoints/${RUN_NAME}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/train-${RUN_ID}.log"
# Must match examples/data_preprocess/prepare.py: files go to
# ${PREP_LOCAL_DIR}/${mode}/train.parquet (mode=text → .../text/train.parquet).
PREP_LOCAL_DIR="${TRAINER_LOCAL_DIR}/data"
TRAIN_DATA_PATH="${PREP_LOCAL_DIR}/text/train.parquet"
VAL_DATA_PATH="${PREP_LOCAL_DIR}/text/test.parquet"

# VLA checkpoint from S3 input channel
VLA_CHECKPOINT_DIR="${SM_CHANNEL_VLA:-/opt/ml/input/data/vla}"
VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"

# Env server config
TRAIN_PORT=50051
VAL_PORT=50052
TRAIN_NUM_ENVS=16
VAL_NUM_ENVS=128
GROUP_SIZE=8
BLOCK_MODE="BLOCK_4"
REWARD_TYPE="block2block"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gigpo}"
MAX_INNER_STEPS=100
NUM_ATTEMPTS=3
MAX_TURNS=1

# GPU isolation: env servers on GPU 0, training on GPUs 1-4
ENV_SERVER_GPU=0
TRAIN_GPUS="1,2,3,4"

######################
### Setup ############
######################
source /opt/miniforge3/etc/profile.d/conda.sh

echo "=== SageMaker LaMer Training ==="
echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "RUN_NAME: ${RUN_NAME}"
echo "TRAINER_LOCAL_DIR: ${TRAINER_LOCAL_DIR}"
echo "VLA_CHECKPOINT: ${VLA_CHECKPOINT}"
echo "SM_NUM_GPUS: ${SM_NUM_GPUS:-unknown}"

mkdir -p "${TRAINER_LOCAL_DIR}" "$(dirname "${RUN_LOG_PATH}")" "$(dirname "${TRAIN_DATA_PATH}")"

######################
### Environment ######
######################
# SageMaker training toolkit sets PYTHONPATH to the lamer env's Python 3.12
# stdlib, which breaks any process using a different Python (ltvenv=3.10,
# conda base=3.13).  Unset it globally; each subprocess sets its own below.
unset PYTHONPATH
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

########################################
### Step 1: Start env servers ##########
########################################
echo ""
echo "=== Starting Language Table env servers ==="

# Build VLA flag
VLA_FLAG=""
if [ -f "${VLA_CHECKPOINT}" ]; then
    VLA_FLAG="--vla_checkpoint ${VLA_CHECKPOINT}"
    echo "VLA enabled: ${VLA_CHECKPOINT}"
else
    echo "WARNING: No VLA checkpoint found at '${VLA_CHECKPOINT}' — using random actions"
fi

# Training server (background)
echo "Starting training server on port ${TRAIN_PORT} (${TRAIN_NUM_ENVS} envs, GPU ${ENV_SERVER_GPU})..."
PYTHONPATH="${LANGTABLE_DIR}" \
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${TRAIN_PORT} \
    --num_envs ${TRAIN_NUM_ENVS} --group_n ${GROUP_SIZE} \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    --reward_type ${REWARD_TYPE} \
    ${VLA_FLAG} \
    &
TRAIN_SERVER_PID=$!
echo "Training server PID: ${TRAIN_SERVER_PID}"

# Validation server (background)
echo "Starting validation server on port ${VAL_PORT} (${VAL_NUM_ENVS} envs, GPU ${ENV_SERVER_GPU})..."
PYTHONPATH="${LANGTABLE_DIR}" \
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${VAL_PORT} \
    --num_envs ${VAL_NUM_ENVS} --group_n 1 \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    --reward_type ${REWARD_TYPE} \
    ${VLA_FLAG} \
    &
VAL_SERVER_PID=$!
echo "Validation server PID: ${VAL_SERVER_PID}"

# Cleanup on exit
cleanup() {
    echo "Cleaning up env servers..."
    kill ${TRAIN_SERVER_PID} ${VAL_SERVER_PID} 2>/dev/null || true
    wait ${TRAIN_SERVER_PID} ${VAL_SERVER_PID} 2>/dev/null || true
    echo "Servers stopped."
}
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
    for i in $(seq 1 ${max_attempts}); do
        if nc -z 127.0.0.1 ${port} 2>/dev/null; then
            echo "  ${name} server ready on port ${port} (attempt ${i})"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: ${name} server on port ${port} did not start after ${max_attempts} attempts"
    return 1
}

wait_for_port ${TRAIN_PORT} "Training" || exit 1
wait_for_port ${VAL_PORT} "Validation" || exit 1

########################################
### Step 3: Connection test ############
########################################
if [ "${SKIP_CONNECTION_TEST:-0}" = "1" ]; then
    echo ""
    echo "=== Skipping connection test (SKIP_CONNECTION_TEST=1) ==="
else
    echo ""
    echo "=== Running connection test ==="
    PYTHONPATH="${LANGTABLE_DIR}" ${LANGTABLE_PYTHON} -m language_table.lamer.test_connection \
        --host 127.0.0.1 --port ${TRAIN_PORT} --val_port ${VAL_PORT} --full \
        --timeout 300
    echo "Connection test passed!"
fi

########################################
### Step 4: LaMer training #############
########################################
echo ""
echo "=== Starting LaMer training (GPUs: ${TRAIN_GPUS}) ==="
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
cd ${LAMER_DIR}

${LAMER_PYTHON} -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "${PREP_LOCAL_DIR}" \
    --train_data_size ${TRAIN_NUM_ENVS} \
    --val_data_size ${VAL_NUM_ENVS}

export TRAIN_DATA_PATH VAL_DATA_PATH
export TRAIN_NUM_ENVS VAL_NUM_ENVS GROUP_SIZE ADV_ESTIMATOR
export RUN_NAME TRAINER_LOCAL_DIR RUN_LOG_PATH
export ENV_ADDRESS="127.0.0.1:${TRAIN_PORT}"
export VAL_ADDRESS="127.0.0.1:${VAL_PORT}"

# Ensure lamer env's python is first on PATH for the training script.
# Avoid `conda activate` — it breaks under set -u (unbound variable errors
# in conda's shell function) and is redundant since the Dockerfile already
# sets PATH to include lamer/bin.
export PATH="/opt/miniforge3/envs/lamer/bin:${PATH}"

set +e
bash "${LAMER_DIR}/examples/language_table/lamer_language_table_slurm.sh"
TRAIN_EXIT_CODE=$?
set -e

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] ERROR: Training exited with code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

# Copy final model to SageMaker output path
if [ -d "${TRAINER_LOCAL_DIR}" ]; then
    echo "=== Copying final model to /opt/ml/model/ ==="
    cp -r "${TRAINER_LOCAL_DIR}"/* /opt/ml/model/ 2>/dev/null || true
fi

echo "END TIME: $(date)"
