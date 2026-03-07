set -x
ENGINE=${1:-vllm}

train_data_size=16
val_data_size=128
group_size=8
mode="mean_norm"                  # "mean_norm" or "mean_std_norm"
reflection_type="history_and_reflection" # "reflection_only" or "history_and_reflection" or "history_only" --> paper shows "reflection_only" works best

# Navigation path length.  Any integer >= 1.
# n=4  → 4-step optimal path, 9×9 open grid    (easy, start here)
# n=6  → 6-step path,         13×13 grid        (medium)
# n=8  → 8-step path,         17×17 grid        (hard)
nav_n=4

experiment_name=nav_lamer_meta_4_step_history_and_reflection
save_dir=/gpfs/scrubbed/memmelma/projects/LaMer

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir $save_dir/data/$experiment_name/verl-agent/ \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$save_dir/checkpoints/$experiment_name \
    trainer.experiment_name=$experiment_name \
    algorithm.adv_estimator=gigpo \
    data.train_files=$save_dir/data/$experiment_name/verl-agent/text/train.parquet \
    data.val_files=$save_dir/data/$experiment_name/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
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
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    reward_model.reward_manager=episode \
    env.env_name=Navigation \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.navigation.n=$nav_n \
    "+env.navigation.train_disturbances=[FlipLeftRight,FlipUpDown]" \
    "+env.navigation.val_disturbances=[FlipBoth]" \
    env.num_attempts=3 \
    env.max_steps=1 \
    env.max_turns=1 \
    +env.reflection_type=$reflection_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lamer' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=500 \
    trainer.val_before_train=True \
    trainer.log_val_generations=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    2>&1 | tee -a $save_dir/slurm/$experiment_name.log
