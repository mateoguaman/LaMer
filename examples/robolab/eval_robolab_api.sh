#!/bin/bash
set -euo pipefail
set -x

# Standalone RoboLab evaluation in human mode.
# Starts the RoboLab env server (via Apptainer) and the Pi0-5 policy server
# (via OpenPI Apptainer), then runs api_rollout.py in --human mode.
#
# Required env vars (all have defaults):
#   TASK                 RoboLab task name (default: BananaInBowlTask)
#   NUM_ENVS             parallel environments (default: 1)
#   MAX_TURNS            max outer turns per episode (default: 10)
#   NUM_INNER_STEPS      inner sim steps per outer step() call (default: 50)
#   NUM_EPISODES         number of episodes to run (default: 1)
#   ENV_SERVER_PORT      port for the RoboLab env server (default: 50051)
#   POLICY_SERVER_PORT   port for the Pi0-5 policy server (default: 4598)
#   POLICY_SERVER_GPU    GPU index for the policy server (default: 0)
#   ENV_SERVER_GPU       GPU index for the env server (default: 0)
#   ROBOLAB_DIR          path to RoboLab repo (default: ~/projects/RoboLab)
#   OPENPI_DIR           path to OpenPI repo (default: ~/projects/openpi)
#   RUN_NAME             for output naming (default: robolab_human)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROBOLAB_DIR="${ROBOLAB_DIR:-${GHOME}/projects/RoboLab}"
OPENPI_DIR="${OPENPI_DIR:-${GHOME}/projects/openpi}"
LAMER_PYTHON="${LAMER_PYTHON:-/gscratch/weirdlab/sidhraja/miniconda3/envs/lamer}"

TASK="${TASK:-BananaInBowlTask}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_TURNS="${MAX_TURNS:-10}"
NUM_INNER_STEPS="${NUM_INNER_STEPS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-50051}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-4598}"
POLICY_SERVER_GPU="${POLICY_SERVER_GPU:-0}"
ENV_SERVER_GPU="${ENV_SERVER_GPU:-0}"
RUN_NAME="${RUN_NAME:-robolab_human}"

OUTPUT_DIR="${LAMER_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_NAME}.jsonl"
VIDEO_DIR="${OUTPUT_DIR}/${RUN_NAME}_videos"
IMAGE_DIR="${OUTPUT_DIR}/${RUN_NAME}_images"

mkdir -p "${OUTPUT_DIR}"

######################
### Server cleanup ###
######################
POLICY_SERVER_PID=""
ENV_SERVER_PID=""

cleanup() {
    echo "Cleaning up servers..."
    [ -n "${ENV_SERVER_PID}" ]    && kill "${ENV_SERVER_PID}"    2>/dev/null || true
    [ -n "${POLICY_SERVER_PID}" ] && kill "${POLICY_SERVER_PID}" 2>/dev/null || true
    [ -n "${ENV_SERVER_PID}" ]    && wait "${ENV_SERVER_PID}"    2>/dev/null || true
    [ -n "${POLICY_SERVER_PID}" ] && wait "${POLICY_SERVER_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

########################################
### Step 1: Start Pi0-5 policy server ##
########################################
echo ""
echo "=== Starting Pi0-5 policy server on GPU ${POLICY_SERVER_GPU}, port ${POLICY_SERVER_PORT} ==="
"${OPENPI_DIR}/run_openpi_server.sh" -c "
    export VIRTUAL_ENV=/workspace/openpi/.venv
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    CUDA_VISIBLE_DEVICES=${POLICY_SERVER_GPU} \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
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
### Step 2: Start RoboLab env server ###
########################################
echo ""
echo "=== Starting RoboLab env server on GPU ${ENV_SERVER_GPU}, port ${ENV_SERVER_PORT} (${NUM_ENVS} envs, task: ${TASK}) ==="
"${ROBOLAB_DIR}/run_apptainer.sh" bash -c "
    export VIRTUAL_ENV=/workspace/robolab/.venv
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
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
ENV_SERVER_PID=$!
echo "Env server PID: ${ENV_SERVER_PID}"

########################################
### Step 3: Wait for policy server #####
########################################
echo ""
echo "=== Waiting for policy server on port ${POLICY_SERVER_PORT} ==="
max_attempts=180  # 6 min max
for i in $(seq 1 ${max_attempts}); do
    if nc -z 127.0.0.1 ${POLICY_SERVER_PORT} 2>/dev/null; then
        echo "  Policy server ready (attempt ${i})"
        break
    fi
    if [ "${i}" -eq "${max_attempts}" ]; then
        echo "  ERROR: Policy server on port ${POLICY_SERVER_PORT} did not start"
        exit 1
    fi
    sleep 2
done

########################################
### Step 4: Wait for env server ########
########################################
echo ""
echo "=== Waiting for env server on port ${ENV_SERVER_PORT} ==="
for i in $(seq 1 ${max_attempts}); do
    if nc -z 127.0.0.1 ${ENV_SERVER_PORT} 2>/dev/null; then
        echo "  Env server ready (attempt ${i})"
        break
    fi
    if [ "${i}" -eq "${max_attempts}" ]; then
        echo "  ERROR: Env server on port ${ENV_SERVER_PORT} did not start"
        exit 1
    fi
    sleep 2
done

########################################
### Step 5: Run human rollout ##########
########################################
echo ""
echo "=== Running RoboLab rollout in human mode ==="
cd "${LAMER_DIR}"

"${LAMER_PYTHON}/bin/python" examples/robolab/api_rollout.py \
    --remote_address "localhost:${ENV_SERVER_PORT}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_envs "${NUM_ENVS}" \
    --max_turns "${MAX_TURNS}" \
    --output "${OUTPUT_FILE}" \
    --video_dir "${VIDEO_DIR}" \
    --image_dir "${IMAGE_DIR}" \
    --human

echo ""
echo "END TIME: $(date)"
echo "Results: ${OUTPUT_FILE}"

########################################
### Step 6: Shut down servers ##########
########################################
echo ""
echo "=== Shutting down servers ==="
[ -n "${ENV_SERVER_PID}" ]    && kill "${ENV_SERVER_PID}"    2>/dev/null || true
[ -n "${POLICY_SERVER_PID}" ] && kill "${POLICY_SERVER_PID}" 2>/dev/null || true
[ -n "${ENV_SERVER_PID}" ]    && wait "${ENV_SERVER_PID}"    2>/dev/null || true
[ -n "${POLICY_SERVER_PID}" ] && wait "${POLICY_SERVER_PID}" 2>/dev/null || true
echo "Servers stopped."
