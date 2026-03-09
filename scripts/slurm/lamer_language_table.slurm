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
TRAINER_LOCAL_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/train-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}.log"

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

######################
### Environment ######
######################
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK:-32}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "${TRAINER_LOCAL_DIR}" "$(dirname "${RUN_LOG_PATH}")"
[ -n "${CONDA_PKGS_DIRS:-}" ] && mkdir -p "${CONDA_PKGS_DIRS}"
[ -n "${PIP_CACHE_DIR:-}" ] && mkdir -p "${PIP_CACHE_DIR}"
[ -n "${TMPDIR:-}" ] && mkdir -p "${TMPDIR}"
[ -n "${XDG_CACHE_HOME:-}" ] && mkdir -p "${XDG_CACHE_HOME}"

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

train_data_size=${TRAIN_NUM_ENVS}
val_data_size=${VAL_NUM_ENVS}
group_size=${GROUP_SIZE}
mode="mean_norm"
reflection_type="reflection_only"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size ${train_data_size} \
    --val_data_size ${val_data_size}

# NOTE: ppo_mini_batch_size must be <= train_data_size * group_size (currently 16*8=128).
# For a smoke test with smaller sizes, lower ppo_mini_batch_size accordingly.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.9 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=${mode} \
    reward_model.reward_manager=episode \
    env.env_name=language_table \
    env.seed=0 \
    +env.remote=True \
    +env.remote_address=127.0.0.1:${TRAIN_PORT} \
    +env.remote_val_address=127.0.0.1:${VAL_PORT} \
    env.num_attempts=${NUM_ATTEMPTS} \
    env.max_turns=${MAX_TURNS} \
    +env.reflection_type=${reflection_type} \
    env.rollout.n=${group_size} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lamer' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=300 \
    trainer.val_before_train=True \
    trainer.log_val_generations=1 \
    trainer.default_local_dir=${TRAINER_LOCAL_DIR} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    2>&1 | tee -a "${RUN_LOG_PATH}"

echo "END TIME: $(date)"
