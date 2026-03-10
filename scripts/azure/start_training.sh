#!/usr/bin/env bash
# start_training.sh — Start multi-node Ray cluster and run LaMer training.
#
# Run this ON vm-train-0 (the Ray head node).
# It SSHes into vm-train-1 to join the Ray cluster, then launches training.
#
# Usage:
#   bash scripts/azure/start_training.sh <env_vm_private_ip> <train1_private_ip>
#
# Example:
#   bash scripts/azure/start_training.sh 10.0.0.6 10.0.0.5
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <env_vm_private_ip> <train1_private_ip>"
    echo ""
    echo "  env_vm_private_ip:  Private IP of the vm-env running env servers"
    echo "  train1_private_ip:  Private IP of vm-train-1 (Ray worker node)"
    echo ""
    echo "Get these IPs from your local machine with:"
    echo "  scripts/azure/provision_vms.sh --info"
    exit 1
fi

ENV_VM_IP="$1"
TRAIN1_IP="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${BASE_LAMER_DIR}/.env.azure"
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env.azure not found at ${ENV_FILE}"
    echo "Copy .env.azure.example to .env.azure and fill in your values."
    exit 1
fi
# shellcheck disable=SC1091
source "${ENV_FILE}"
if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi

LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/.." && pwd)/language-table}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:?'Set LAMER_CONDA_ENV in .env.azure'}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:?'Set LANGTABLE_CONDA_ENV in .env.azure'}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?'Set CHECKPOINT_ROOT in .env.azure'}"
RUN_NAME="${RUN_NAME:?'Set RUN_NAME in .env.azure'}"
HF_HOME="${HF_HOME:?'Set HF_HOME in .env.azure'}"
TRAIN_PORT="${TRAIN_PORT:-50051}"
VAL_PORT="${VAL_PORT:-50052}"
SSH_KEY="${AZURE_SSH_KEY:?'Set AZURE_SSH_KEY in .env.azure'}"
ADMIN="${AZURE_ADMIN_USER:?'Set AZURE_ADMIN_USER in .env.azure'}"

TRAIN_DATA_PATH="${HOME}/data/verl-agent/text/train.parquet"
VAL_DATA_PATH="${HOME}/data/verl-agent/text/test.parquet"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16}"
VAL_NUM_ENVS="${VAL_NUM_ENVS:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"
MAX_TURNS="${MAX_TURNS:-1}"

# Training config
N_GPUS_PER_NODE=2
NNODES=2
TOTAL_GPUS=$((N_GPUS_PER_NODE * NNODES))

TRAINER_LOCAL_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/train-$(date +%Y%m%d-%H%M%S).log"

echo "=== LaMer Multi-Node Training ==="
echo "  Head node (this):    $(hostname) / $(hostname -I | awk '{print $1}')"
echo "  Worker node:         ${TRAIN1_IP}"
echo "  Env server:          ${ENV_VM_IP}:${TRAIN_PORT}/${VAL_PORT}"
echo "  GPUs per node:       ${N_GPUS_PER_NODE}"
echo "  Total training GPUs: ${TOTAL_GPUS}"
echo "  Checkpoint dir:      ${TRAINER_LOCAL_DIR}"
echo ""

######################
### Activate conda ###
######################
# Miniforge may not be on PATH in a fresh shell; source it from known location.
if ! command -v conda &>/dev/null; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u

######################
### Environment ######
######################
export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
# Azure NC-series VMs use Ethernet, not InfiniBand — force NCCL to use TCP sockets.
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
if [ -n "${HF_HOME}" ]; then
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
fi

mkdir -p "${TRAINER_LOCAL_DIR}" "$(dirname "${RUN_LOG_PATH}")"

########################################
### Step 1: Test env server reachability
########################################
echo "=== Testing env server connectivity ==="
if ! nc -z -w5 "${ENV_VM_IP}" "${TRAIN_PORT}" 2>/dev/null; then
    echo "ERROR: Cannot reach env server at ${ENV_VM_IP}:${TRAIN_PORT}"
    echo "Make sure start_env_servers.sh is running on vm-env."
    exit 1
fi
if ! nc -z -w5 "${ENV_VM_IP}" "${VAL_PORT}" 2>/dev/null; then
    echo "ERROR: Cannot reach env server at ${ENV_VM_IP}:${VAL_PORT}"
    exit 1
fi
echo "  Env servers reachable."

########################################
### Step 2: Start Ray cluster ##########
########################################
echo ""
echo "=== Starting Ray cluster ==="

HEAD_IP="$(hostname -I | awk '{print $1}')"
RAY_PORT=6379

# Stop any existing Ray instances
ray stop --force 2>/dev/null || true

# Start Ray head on this node
echo "Starting Ray head on ${HEAD_IP}:${RAY_PORT}..."
ray start --head \
    --port=${RAY_PORT} \
    --num-gpus=${N_GPUS_PER_NODE} \
    --num-cpus=32 \
    --dashboard-host=0.0.0.0
echo "  Ray head started."

# SSH into vm-train-1 and join the Ray cluster
echo "Starting Ray worker on ${TRAIN1_IP}..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${ADMIN}@${TRAIN1_IP}" bash -l <<REMOTE_EOF
set -euo pipefail
if ! command -v conda &>/dev/null; then
    source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    source "\$(conda info --base)/etc/profile.d/conda.sh"
fi
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
ray stop --force 2>/dev/null || true
ray start --address=${HEAD_IP}:${RAY_PORT} \
    --num-gpus=${N_GPUS_PER_NODE} \
    --num-cpus=32
echo "Ray worker joined cluster."
REMOTE_EOF

# Verify cluster
echo ""
echo "=== Ray cluster status ==="
ray status
echo ""

# Wait for all nodes to register
echo "Waiting for ${NNODES} nodes to join..."
for attempt in $(seq 1 30); do
    node_count=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo 0)
    if [ "${node_count}" -ge "${NNODES}" ]; then
        echo "  ${node_count} nodes registered."
        break
    fi
    if [ "${attempt}" -eq 30 ]; then
        echo "  ERROR: Only ${node_count}/${NNODES} nodes after 60s. Check Ray worker logs."
        exit 1
    fi
    sleep 2
done

########################################
### Step 3: Run connection test ########
########################################
echo ""
echo "=== Running env connection test ==="
LANGTABLE_PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python)"
${LANGTABLE_PYTHON} -m language_table.lamer.test_connection \
    --host "${ENV_VM_IP}" --port "${TRAIN_PORT}" --val_port "${VAL_PORT}" --full \
    --timeout 300
echo "Connection test passed!"

########################################
### Step 4: Data preparation ###########
########################################
echo ""
echo "=== Preparing data ==="
cd "${LAMER_DIR}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size ${TRAIN_NUM_ENVS} \
    --val_data_size ${VAL_NUM_ENVS}

########################################
### Step 5: LaMer training #############
########################################
echo ""
echo "=== Starting LaMer training ==="
echo "  Nodes: ${NNODES}, GPUs/node: ${N_GPUS_PER_NODE}, Total: ${TOTAL_GPUS}"
echo "  Script: examples/language_table/lamer_language_table_azure_2x2.sh"
echo "  Log: ${RUN_LOG_PATH}"
echo ""

# Export vars that the example script reads
export TRAIN_DATA_PATH VAL_DATA_PATH
export ENV_VM_IP TRAIN_PORT VAL_PORT
export TRAIN_NUM_ENVS VAL_NUM_ENVS GROUP_SIZE
export RUN_NAME TRAINER_LOCAL_DIR RUN_LOG_PATH

bash "${LAMER_DIR}/examples/language_table/lamer_language_table_azure_2x2.sh"

echo ""
echo "END TIME: $(date)"
