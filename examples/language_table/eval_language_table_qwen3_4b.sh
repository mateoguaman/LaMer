#!/bin/bash
set -euo pipefail
set -x

# Inference-only evaluation of the LaMer MetaRL loop on Language Table.
# No PPO updates — trainer exits after a single validation pass.
#
# GPU layout (2-GPU local machine):
#   GPU 1 (RTX 3080, 10GB)   → language_table env server + SmolVLA
#   GPU 0 (RTX 2080 Ti, 11GB) → Qwen3.5-4B via vllm
#
# Required env vars:
#   LANGTABLE_DIR        path to language-table repo (default: ~/projects/language-table)
#   LANGTABLE_PYTHON     python binary with language-table deps (default: ltvenv/bin/python)
#   SMOLVLA_CHECKPOINT   HF repo id or local path (default: Sidharth-R/langtable-smolvla-finetuned)
#   RUN_NAME             for logging (default: eval_language_table_qwen3_4b)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LANGTABLE_DIR="${LANGTABLE_DIR:-${HOME}/projects/language-table}"
LANGTABLE_PYTHON="${LANGTABLE_PYTHON:-ltvenv/bin/python}"
SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-Sidharth-R/langtable-smolvla-finetuned}"
RUN_NAME="${RUN_NAME:-eval_language_table_qwen3_4b}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${HOME}/data/verl-agent/text/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${HOME}/data/verl-agent/text/test.parquet}"
TRAINER_LOCAL_DIR="${TRAINER_LOCAL_DIR:-${LAMER_DIR}/checkpoints/lamer/${RUN_NAME}}"
RUN_LOG_PATH="${TRAINER_LOCAL_DIR}/${RUN_NAME}.log"

VAL_NUM_ENVS="${VAL_NUM_ENVS:-8}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-5}"
VAL_PORT=50052

ENV_SERVER_GPU=1
TRAIN_GPU=0

export PYTHONPATH="${LANGTABLE_DIR}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "${TRAINER_LOCAL_DIR}"

######################
### Server cleanup ###
######################
VAL_SERVER_PID=""

cleanup() {
    echo "Cleaning up env server..."
    [ -n "${VAL_SERVER_PID}" ] && kill "${VAL_SERVER_PID}" 2>/dev/null || true
    [ -n "${VAL_SERVER_PID}" ] && wait "${VAL_SERVER_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

########################################
### Step 1: Generate placeholder data ##
########################################
# The language_table env generates its own tasks at reset(); the parquet only
# needs to supply the correct schema for the dataloader.
if [ ! -f "${VAL_DATA_PATH}" ] || [ ! -f "${TRAIN_DATA_PATH}" ]; then
    echo "=== Generating placeholder parquet files ==="
    cd "${LAMER_DIR}"
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --train_data_size "${VAL_NUM_ENVS}" \
        --val_data_size "${VAL_NUM_ENVS}"
fi

########################################
### Step 2: Start env server ###########
########################################
echo ""
echo "=== Starting Language Table validation server on GPU ${ENV_SERVER_GPU} ==="
CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
${LANGTABLE_PYTHON} -m language_table.lamer.server_main \
    --host 0.0.0.0 --port ${VAL_PORT} \
    --num_envs ${VAL_NUM_ENVS} --group_n 1 \
    --max_inner_steps ${MAX_INNER_STEPS} --num_attempts ${NUM_ATTEMPTS} \
    --max_turns ${MAX_TURNS} --do_reflection \
    --reward_type block2block \
    --split val \
    --policy smolvla \
    --vla_checkpoint "${SMOLVLA_CHECKPOINT}" \
    &
VAL_SERVER_PID=$!
echo "Validation server PID: ${VAL_SERVER_PID}"

########################################
### Step 3: Wait for server ############
########################################
echo ""
echo "=== Waiting for server to become ready ==="
max_attempts=120  # 4 min max
for i in $(seq 1 ${max_attempts}); do
    if nc -z 127.0.0.1 ${VAL_PORT} 2>/dev/null; then
        echo "  Validation server ready on port ${VAL_PORT} (attempt ${i})"
        break
    fi
    if [ "${i}" -eq "${max_attempts}" ]; then
        echo "  ERROR: Server on port ${VAL_PORT} did not start after ${max_attempts} attempts"
        exit 1
    fi
    sleep 2
done

########################################
### Step 4: Connection test ############
########################################
echo ""
echo "=== Running connection test ==="
${LANGTABLE_PYTHON} -m language_table.lamer.test_connection \
    --host 127.0.0.1 --val_port ${VAL_PORT} --full \
    --timeout 300
echo "Connection test passed!"

########################################
### Step 5: Val-only evaluation ########
########################################
echo ""
echo "=== Starting val-only evaluation on GPU ${TRAIN_GPU} ==="
cd "${LAMER_DIR}"

CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files="${TRAIN_DATA_PATH}" \
    data.val_files="${VAL_DATA_PATH}" \
    data.train_batch_size="${VAL_NUM_ENVS}" \
    data.val_batch_size="${VAL_NUM_ENVS}" \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${VAL_NUM_ENVS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${VAL_NUM_ENVS}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${VAL_NUM_ENVS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${VAL_NUM_ENVS}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.9 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    reward_model.reward_manager=episode \
    env.env_name=language_table \
    env.seed=0 \
    +env.remote=True \
    +env.remote_address=localhost:${VAL_PORT} \
    +env.remote_val_address=localhost:${VAL_PORT} \
    env.rollout.n=1 \
    env.num_attempts="${NUM_ATTEMPTS}" \
    env.max_turns="${MAX_TURNS}" \
    +env.reflection_type=history_and_reflection \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='lamer' \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${TRAINER_LOCAL_DIR}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations="${VAL_NUM_ENVS}" \
    trainer.log_val_videos=4 \
    trainer.resume_mode=disable \
    2>&1 | tee -a "${RUN_LOG_PATH}"

echo "END TIME: $(date)"
echo "Check console output above for val/success_rate[0..${NUM_ATTEMPTS}] metrics."
