set -xo pipefail

# Tillicum/SLURM: 1 node x 4 training GPUs (GPU 0 reserved for env servers).
# Env servers run on localhost (same node).
#
# Required env vars (set by lamer_language_table.slurm or export manually):
#   TRAIN_DATA_PATH, VAL_DATA_PATH

ENGINE=${ENGINE:-vllm}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gigpo}

train_data_size=${TRAIN_NUM_ENVS:-16}
val_data_size=${VAL_NUM_ENVS:-128}
group_size=${GROUP_SIZE:-8}
num_attempts=${NUM_ATTEMPTS:-3}
max_turns=${MAX_TURNS:-5}
learning_rate=${LEARNING_RATE:-1e-6}
batch_size=${BATCH_SIZE:-64}
micro_batch_size=${MICRO_BATCH_SIZE:-16}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_reward_coef=${KL_REWARD_COEF:-0.001}
mode="mean_norm"
reflection_type="history_and_reflection"

env_address="${ENV_ADDRESS:-127.0.0.1:50051}"
val_address="${VAL_ADDRESS:-127.0.0.1:50052}"

# Algorithm-specific overrides
ALGO_ARGS=()
if [ "$ADV_ESTIMATOR" = "gigpo" ]; then
    ALGO_ARGS+=(
        "+algorithm.step_gamma=0.95"
        "+algorithm.traj_gamma=0.9"
        "algorithm.gigpo.step_advantage_w=1.0"
        "algorithm.gigpo.mode=$mode"
    )
fi

if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    echo "flash-attn not found; installing..."
    pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=${TRAIN_DATA_PATH:?'Set TRAIN_DATA_PATH'} \
    data.val_files=${VAL_DATA_PATH:?'Set VAL_DATA_PATH'} \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
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
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_reward_coef \
    algorithm.gamma=0.95 \
    "${ALGO_ARGS[@]}" \
    reward_model.reward_manager=episode \
    env.env_name=language_table \
    env.seed=0 \
    +env.remote=True \
    +env.remote_address=$env_address \
    +env.remote_val_address=$val_address \
    env.rollout.n=$group_size \
    env.num_attempts=$num_attempts \
    env.max_turns=$max_turns \
    +env.reflection_type=$reflection_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lamer' \
    trainer.experiment_name=${RUN_NAME:-language_table_lamer_qwen3_4b} \
    trainer.default_local_dir=${TRAINER_LOCAL_DIR:-checkpoints/lamer/${RUN_NAME:-language_table_lamer_qwen3_4b}} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1000 \
    trainer.val_before_train=True \
    trainer.log_val_generations=$val_data_size \
    trainer.log_train_generations=$train_data_size \
    trainer.log_train_videos=8 \
    trainer.log_val_videos=8 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    2>&1 | tee -a ${RUN_LOG_PATH:-language_table_slurm.log}
