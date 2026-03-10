#!/usr/bin/env bash
# start_env_servers.sh — Start language-table env servers on the env VM.
#
# Run this ON the vm-env Azure VM.
# Starts training and validation env servers, then waits for them.
#
# Usage:
#   bash scripts/azure/start_env_servers.sh
#   bash scripts/azure/start_env_servers.sh --fg   # run in foreground (don't background servers)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${BASE_LAMER_DIR}/.env.azure" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.azure"
fi
if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi

LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/.." && pwd)/language-table}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

TRAIN_PORT="${TRAIN_PORT:-50051}"
VAL_PORT="${VAL_PORT:-50052}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16}"
VAL_NUM_ENVS="${VAL_NUM_ENVS:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
BLOCK_MODE="${BLOCK_MODE:-BLOCK_4}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-100}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"
MAX_TURNS="${MAX_TURNS:-1}"

# Resolve VLA checkpoint
if [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi

# Find language-table python
# Miniforge may not be on PATH in a fresh shell; source it from known location.
if ! command -v conda &>/dev/null; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
LANGTABLE_PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python)"
echo "Using language-table python: ${LANGTABLE_PYTHON}"

# Add language-table to PYTHONPATH
export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"

# Use all available GPUs for env serving
# A100 VM has 1 GPU; if dual-GPU, JAX can use both
echo ""
echo "=== GPU available for env servers ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Limit JAX memory — don't preallocate, use at most 40%
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# VLA flag
VLA_FLAG=""
if [ -n "${VLA_CHECKPOINT}" ] && [ -f "${VLA_CHECKPOINT}" ]; then
    VLA_FLAG="--vla_checkpoint ${VLA_CHECKPOINT}"
    echo "VLA enabled: ${VLA_CHECKPOINT}"
else
    echo "WARNING: No VLA checkpoint at '${VLA_CHECKPOINT:-<unset>}' — using random actions"
fi

echo ""
echo "=== Starting env servers ==="
echo "  Training:   port ${TRAIN_PORT} (${TRAIN_NUM_ENVS} envs x ${GROUP_SIZE} group)"
echo "  Validation: port ${VAL_PORT} (${VAL_NUM_ENVS} envs)"
echo "  Bind:       0.0.0.0 (accessible from training VMs)"
echo ""

LOG_DIR="${LAMER_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Training server
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${TRAIN_PORT} \
    --num_envs ${TRAIN_NUM_ENVS} --group_n ${GROUP_SIZE} \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    ${VLA_FLAG} \
    > "${LOG_DIR}/env_train.log" 2>&1 &
TRAIN_PID=$!
echo "Training server PID: ${TRAIN_PID}"

# Validation server (group_n=1 for eval)
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${VAL_PORT} \
    --num_envs ${VAL_NUM_ENVS} --group_n 1 \
    --block_mode ${BLOCK_MODE} \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    ${VLA_FLAG} \
    > "${LOG_DIR}/env_val.log" 2>&1 &
VAL_PID=$!
echo "Validation server PID: ${VAL_PID}"

# Cleanup on exit
cleanup() {
    echo "Stopping env servers..."
    kill ${TRAIN_PID} ${VAL_PID} 2>/dev/null || true
    wait ${TRAIN_PID} ${VAL_PID} 2>/dev/null || true
    echo "Servers stopped."
}
trap cleanup EXIT

# Wait for servers to become ready
echo ""
echo "=== Waiting for servers ==="
wait_for_port() {
    local port=$1
    local name=$2
    local max_attempts=120
    for i in $(seq 1 ${max_attempts}); do
        if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
            echo "  ${name} ready on port ${port} (attempt ${i})"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: ${name} on port ${port} did not start after ${max_attempts} attempts"
    echo "  Check logs: ${LOG_DIR}/env_${name,,}.log"
    return 1
}

wait_for_port "${TRAIN_PORT}" "Training" || exit 1
wait_for_port "${VAL_PORT}" "Validation" || exit 1

echo ""
echo "=== Env servers ready ==="
echo "  Training:   0.0.0.0:${TRAIN_PORT} (PID ${TRAIN_PID})"
echo "  Validation: 0.0.0.0:${VAL_PORT} (PID ${VAL_PID})"
echo "  Logs:       ${LOG_DIR}/env_train.log, ${LOG_DIR}/env_val.log"
echo ""
echo "  Servers will keep running until you Ctrl-C or this shell exits."
echo "  Training VMs should connect to this VM's private IP on these ports."
echo ""

# Keep the script alive so the trap works and servers stay up
wait
