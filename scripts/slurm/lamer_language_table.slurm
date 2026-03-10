#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=lamer-langtable
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

######################
### Config ###########
######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAMER_DIR="${LAMER_DIR:-${DEFAULT_LAMER_DIR}}"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/../language-table" && pwd)}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
TRAIN_DATA_PATH="$HOME/data/verl-agent/text/train.parquet"
VAL_DATA_PATH="$HOME/data/verl-agent/text/test.parquet"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
if [ -z "${CHECKPOINT_ROOT}" ]; then
    echo "ERROR: CHECKPOINT_ROOT is not set."
    echo "Set CHECKPOINT_ROOT in .env.language_table before submitting."
    exit 1
fi
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
TRAINER_LOCAL_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}/${RUN_ID}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/train-${RUN_ID}.log"

TRAIN_PORT=50051
VAL_PORT=50052
TRAIN_NUM_ENVS=16
VAL_NUM_ENVS=128
GROUP_SIZE=8
BLOCK_MODE="BLOCK_4"
MAX_INNER_STEPS=100
NUM_ATTEMPTS=3
MAX_TURNS=1

# VLA (inner-loop policy) — set to "" to use random actions
VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"
if [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi

######################
### Setup ############
######################
module load conda/Miniforge3-25.3.1-3
module load gcc/13.4.0
module load cmake/3.31.8

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
LANGTABLE_PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python)"
echo "Environment activated!"
echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "LAMER_DIR: ${LAMER_DIR}"
echo "LANGTABLE_DIR: ${LANGTABLE_DIR}"
echo "LANGTABLE_CONDA_ENV: ${LANGTABLE_CONDA_ENV}"
echo "TRAINER_LOCAL_DIR: ${TRAINER_LOCAL_DIR}"
echo "CONDA_PKGS_DIRS: ${CONDA_PKGS_DIRS:-<default>}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR:-<default>}"
echo "TMPDIR: ${TMPDIR:-<default>}"
echo "WANDB_DATA_DIR: ${WANDB_DATA_DIR:-<default>}"
echo "XDG_CACHE_HOME: ${XDG_CACHE_HOME:-<default>}"

######################
### Environment ######
######################
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK:-32}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
if [ -n "${WANDB_DATA_DIR:-}" ]; then
    export WANDB_DATA_DIR
fi
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "${TRAINER_LOCAL_DIR}" "$(dirname "${RUN_LOG_PATH}")"
[ -n "${CONDA_PKGS_DIRS:-}" ] && mkdir -p "${CONDA_PKGS_DIRS}"
[ -n "${PIP_CACHE_DIR:-}" ] && mkdir -p "${PIP_CACHE_DIR}"
[ -n "${TMPDIR:-}" ] && mkdir -p "${TMPDIR}"
[ -n "${XDG_CACHE_HOME:-}" ] && mkdir -p "${XDG_CACHE_HOME}"
[ -n "${WANDB_DATA_DIR:-}" ] && mkdir -p "${WANDB_DATA_DIR}"

# Language-table needs its repo on PYTHONPATH
export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"

# GPU isolation: env servers (JAX/LAVA) on GPU 0, LaMer training on GPUs 1-4.
# Without this, JAX preallocates most of GPU memory before training starts → OOM.
ENV_SERVER_GPU=0
TRAIN_GPUS="1,2,3,4"

########################################
### Step 0: Download VLA checkpoint ####
########################################
if [ -n "${VLA_CHECKPOINT}" ] && [ ! -f "${VLA_CHECKPOINT}" ]; then
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
echo "=== Starting Language Table env servers ==="

# Build VLA flag (empty string if no checkpoint)
VLA_FLAG=""
if [ -n "${VLA_CHECKPOINT}" ] && [ -f "${VLA_CHECKPOINT}" ]; then
    VLA_FLAG="--vla_checkpoint ${VLA_CHECKPOINT}"
    echo "VLA enabled: ${VLA_CHECKPOINT}"
else
    echo "WARNING: No VLA checkpoint found at '${VLA_CHECKPOINT}' — using random actions"
fi

# Training server (background)
# Pin to ENV_SERVER_GPU; disable JAX preallocation so it only uses what LAVA needs (~2-3 GB)
echo "Starting training server on port ${TRAIN_PORT} (${TRAIN_NUM_ENVS} envs, GPU ${ENV_SERVER_GPU})..."
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${TRAIN_PORT} \
    --num_envs ${TRAIN_NUM_ENVS} --group_n ${GROUP_SIZE} \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    ${VLA_FLAG} \
    &
TRAIN_SERVER_PID=$!
echo "Training server PID: ${TRAIN_SERVER_PID} (${TRAIN_NUM_ENVS}x${GROUP_SIZE}=$((TRAIN_NUM_ENVS * GROUP_SIZE)) workers)"

# Validation server (background, group_n=1 — no grouping for eval)
echo "Starting validation server on port ${VAL_PORT} (${VAL_NUM_ENVS} envs, GPU ${ENV_SERVER_GPU})..."
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${VAL_PORT} \
    --num_envs ${VAL_NUM_ENVS} --group_n 1 \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    ${VLA_FLAG} \
    &
VAL_SERVER_PID=$!
echo "Validation server PID: ${VAL_SERVER_PID}"

# Cleanup servers on exit (SIGTERM, SIGINT, or script end)
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
    local max_attempts=120  # 4 min max (128+ PyBullet workers take time to init)
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
echo ""
echo "=== Running connection test ==="
${LANGTABLE_PYTHON} -m language_table.lamer.test_connection \
    --host 127.0.0.1 --port ${TRAIN_PORT} --val_port ${VAL_PORT} --full \
    --timeout 300
echo "Connection test passed!"

########################################
### Step 4: LaMer training #############
########################################
echo ""
echo "=== Starting LaMer training (GPUs: ${TRAIN_GPUS}) ==="
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
cd ${LAMER_DIR}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size ${TRAIN_NUM_ENVS} \
    --val_data_size ${VAL_NUM_ENVS}

export TRAIN_DATA_PATH VAL_DATA_PATH
export TRAIN_NUM_ENVS VAL_NUM_ENVS GROUP_SIZE
export RUN_NAME TRAINER_LOCAL_DIR RUN_LOG_PATH
export ENV_ADDRESS="127.0.0.1:${TRAIN_PORT}"
export VAL_ADDRESS="127.0.0.1:${VAL_PORT}"

bash "${LAMER_DIR}/examples/language_table/lamer_language_table_slurm.sh"

echo "END TIME: $(date)"
