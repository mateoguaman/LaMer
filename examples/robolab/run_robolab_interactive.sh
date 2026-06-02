#!/bin/bash
# Run all RoboLab + LaMer components in the current interactive session.
# Assumes the session already has 3 GPUs allocated (e.g. via salloc).
#
# GPU layout:
#   GPU 0  — Pi0-5 policy server + RoboLab train/val env servers
#   GPU 1  — LaMer training (actor / rollout)
#   GPU 2  — LaMer training (ref / critic)
#
# Usage:
#   bash examples/robolab/run_robolab_interactive.sh
#
# Override any default with env vars, e.g.:
#   TASK=BananaInBowlTask NUM_ENVS=2 bash examples/robolab/run_robolab_interactive.sh

set -euo pipefail
set -x

######################
### Config ###########
######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROBOLAB_DIR="${ROBOLAB_DIR:-/gscratch/weirdlab/sidhraja/projects/RoboLab}"
OPENPI_DIR="${OPENPI_DIR:-/gscratch/weirdlab/sidhraja/projects/openpi}"
LAMER_PYTHON="${LAMER_PYTHON:-/gscratch/weirdlab/sidhraja/miniconda3/envs/lamer}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"

TASK="${TASK:-BananaInBowlTask}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_TURNS="${MAX_TURNS:-5}"
NUM_INNER_STEPS="${NUM_INNER_STEPS:-50}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"
# GRPO/GiGPO: one group containing all RoboLab trajectories by default.
NUM_GROUPS="${NUM_GROUPS:-1}"
GROUP_SIZE="${GROUP_SIZE:-${NUM_ENVS}}"
ENABLE_VALIDATION="${ENABLE_VALIDATION:-False}"
case "${ENABLE_VALIDATION,,}" in
    1|true|yes|y) ENABLE_VALIDATION=True ;;
    *) ENABLE_VALIDATION=False ;;
esac

ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
RUN_NAME="${RUN_NAME:-robolab_lamer_qwen3vl_4b}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${LAMER_DIR}/checkpoints/lamer}"

POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-4598}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-50051}"
VAL_SERVER_PORT="${VAL_SERVER_PORT:-50052}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
TRAINER_LOCAL_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}/${RUN_ID}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/train-${RUN_ID}.log"

# GPU assignment
SERVERS_GPU=0
TRAIN_GPUS="1,2"

######################
### Environment ######
######################
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR

mkdir -p "${TRAINER_LOCAL_DIR}"

if [ "$((NUM_GROUPS * GROUP_SIZE))" -ne "${NUM_ENVS}" ]; then
    echo "ERROR: NUM_GROUPS * GROUP_SIZE must equal NUM_ENVS (${NUM_GROUPS} * ${GROUP_SIZE} != ${NUM_ENVS})"
    exit 1
fi

echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "LAMER_DIR: ${LAMER_DIR}"
echo "TASK: ${TASK}"
echo "NUM_ENVS: ${NUM_ENVS}"
echo "NUM_GROUPS: ${NUM_GROUPS}"
echo "GROUP_SIZE: ${GROUP_SIZE}"
echo "ENABLE_VALIDATION: ${ENABLE_VALIDATION}"
echo "TRAINER_LOCAL_DIR: ${TRAINER_LOCAL_DIR}"
echo "SERVERS_GPU: ${SERVERS_GPU}"
echo "TRAIN_GPUS: ${TRAIN_GPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

######################
### Cleanup ##########
######################
POLICY_SERVER_PID=""
ENV_TRAIN_SERVER_PID=""
ENV_VAL_SERVER_PID=""
MEM_MONITOR_PID=""

cleanup() {
    set +e
    echo "[$(date)] Shutting down all servers..."
    [ -n "${MEM_MONITOR_PID}" ]      && kill "${MEM_MONITOR_PID}"      2>/dev/null
    [ -n "${ENV_TRAIN_SERVER_PID}" ] && kill "${ENV_TRAIN_SERVER_PID}" 2>/dev/null
    [ -n "${ENV_VAL_SERVER_PID}" ]   && kill "${ENV_VAL_SERVER_PID}"   2>/dev/null
    [ -n "${POLICY_SERVER_PID}" ]    && kill "${POLICY_SERVER_PID}"    2>/dev/null
    [ -n "${ENV_TRAIN_SERVER_PID}" ] && wait "${ENV_TRAIN_SERVER_PID}" 2>/dev/null
    [ -n "${ENV_VAL_SERVER_PID}" ]   && wait "${ENV_VAL_SERVER_PID}"   2>/dev/null
    [ -n "${POLICY_SERVER_PID}" ]    && wait "${POLICY_SERVER_PID}"    2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

########################################
### Step 1: Start Pi0-5 policy server ##
########################################
echo ""
echo "=== Starting Pi0-5 policy server on GPU ${SERVERS_GPU}, port ${POLICY_SERVER_PORT} ==="
"${OPENPI_DIR}/run_openpi_server.sh" -c "
    export VIRTUAL_ENV=/workspace/openpi/.venv
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    CUDA_VISIBLE_DEVICES=${SERVERS_GPU} \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
    uv run scripts/serve_policy.py \
        --port ${POLICY_SERVER_PORT} \
        policy:checkpoint \
        --policy.config=pi05_droid_jointpos_polaris \
        --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
" &
POLICY_SERVER_PID=$!
echo "Policy server PID: ${POLICY_SERVER_PID}"

########################################
### Step 2: Start RoboLab env servers ##
########################################
echo ""
echo "=== Starting RoboLab train env server on GPU ${SERVERS_GPU}, port ${ENV_SERVER_PORT} ==="
"${ROBOLAB_DIR}/run_apptainer.sh" bash -c "
    export VIRTUAL_ENV=/workspace/robolab/.venv
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    CUDA_VISIBLE_DEVICES=${SERVERS_GPU} \
    uv run examples/policy/server_main.py \
        --policy pi05 \
        --task ${TASK} \
        --num-envs ${NUM_ENVS} \
        --headless \
        --remote-host localhost \
        --remote-port ${POLICY_SERVER_PORT} \
        --server-host 0.0.0.0 \
        --server-port ${ENV_SERVER_PORT} \
        --num-inner-steps ${NUM_INNER_STEPS} \
        --max-turns ${MAX_TURNS}
" &
ENV_TRAIN_SERVER_PID=$!
echo "Train env server PID: ${ENV_TRAIN_SERVER_PID}"

if [ "${ENABLE_VALIDATION}" = "True" ]; then
    echo ""
    echo "=== Starting RoboLab val env server on GPU ${SERVERS_GPU}, port ${VAL_SERVER_PORT} ==="
    "${ROBOLAB_DIR}/run_apptainer.sh" bash -c "
        export VIRTUAL_ENV=/workspace/robolab/.venv
        export PATH=\$VIRTUAL_ENV/bin:\$PATH
        CUDA_VISIBLE_DEVICES=${SERVERS_GPU} \
        uv run examples/policy/server_main.py \
            --policy pi05 \
            --task ${TASK} \
            --num-envs ${NUM_ENVS} \
            --headless \
            --remote-host localhost \
            --remote-port ${POLICY_SERVER_PORT} \
            --server-host 0.0.0.0 \
            --server-port ${VAL_SERVER_PORT} \
            --num-inner-steps ${NUM_INNER_STEPS} \
            --max-turns ${MAX_TURNS}
    " &
    ENV_VAL_SERVER_PID=$!
    echo "Val env server PID: ${ENV_VAL_SERVER_PID}"
else
    echo ""
    echo "=== Skipping RoboLab val env server (ENABLE_VALIDATION=False) ==="
fi

########################################
### Step 3: Wait for servers ###########
########################################
echo ""
echo "=== Waiting for servers ==="

wait_for_port() {
    local port=$1
    local name=$2
    local max_attempts=180  # 6 min max
    for i in $(seq 1 "${max_attempts}"); do
        if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
            echo "  ${name} ready on port ${port} (attempt ${i})"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: ${name} on port ${port} did not start after ${max_attempts} attempts"
    return 1
}

wait_for_port "${POLICY_SERVER_PORT}" "Policy server"    || exit 1
wait_for_port "${ENV_SERVER_PORT}"    "Train env server" || exit 1
if [ "${ENABLE_VALIDATION}" = "True" ]; then
    wait_for_port "${VAL_SERVER_PORT}" "Val env server" || exit 1
fi

########################################
### Memory monitor #####################
########################################
monitor_memory() {
    while true; do
        echo "[$(date)] [mem] $(free -h 2>/dev/null | grep Mem || echo 'N/A')"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null || true
        sleep 300
    done
}
monitor_memory &
MEM_MONITOR_PID=$!

########################################
### Step 4: Prepare data ###############
########################################
echo ""
echo "=== Preparing data ==="
cd "${LAMER_DIR}"

TRAIN_NUM_ENVS="${NUM_GROUPS}"
VAL_NUM_ENVS="${NUM_GROUPS}"
DATA_LOCAL_DIR="${HOME}/data/verl-agent"
TRAIN_DATA_PATH="${DATA_LOCAL_DIR}/visual/train.parquet"
VAL_DATA_PATH="${DATA_LOCAL_DIR}/visual/test.parquet"

"${LAMER_PYTHON}/bin/python" -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --local_dir "${DATA_LOCAL_DIR}" \
    --train_data_size "${TRAIN_NUM_ENVS}" \
    --val_data_size "${VAL_NUM_ENVS}"

########################################
### Step 5: LaMer training #############
########################################
echo ""
echo "=== Starting LaMer training on GPUs ${TRAIN_GPUS} ==="
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

export TRAIN_DATA_PATH VAL_DATA_PATH
export TRAIN_NUM_ENVS VAL_NUM_ENVS
export GROUP_SIZE ADV_ESTIMATOR
export ENABLE_VALIDATION
export NUM_ATTEMPTS MAX_TURNS
export RUN_NAME TRAINER_LOCAL_DIR RUN_LOG_PATH
export ENV_ADDRESS="127.0.0.1:${ENV_SERVER_PORT}"
export VAL_ADDRESS="127.0.0.1:${VAL_SERVER_PORT}"
export BATCH_SIZE="${NUM_ENVS}"
export MICRO_BATCH_SIZE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u

set +e
bash "${LAMER_DIR}/examples/robolab/lamer_robolab_slurm.sh"
TRAIN_EXIT_CODE=$?
set -e

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
    echo "[$(date)] ERROR: Training exited with code ${TRAIN_EXIT_CODE}"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    exit "${TRAIN_EXIT_CODE}"
fi

echo "END TIME: $(date)"
