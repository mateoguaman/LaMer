import imageio.v2 as imageio

import numpy as np
from language_table.environments import blocks, language_table
from language_table.environments.rewards import block2absolutelocation
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step as ts
from language_table.lamer.lava_policy import LAVAPolicy
from agent_system.environments.language_table.smolvla_policy import SmolVLAPolicy

num_steps = 100
instruction = "push the red moon to the top right corner"

block_mode = blocks.LanguageTableBlockVariants.BLOCK_4
reward_factory = block2absolutelocation.BlockToAbsoluteLocationReward

env = language_table.LanguageTable(
    block_mode=block_mode,
    reward_factory=reward_factory,
)

env = gym_wrapper.GymWrapper(env)

if not hasattr(env, "get_control_frequency"):
    env.get_control_frequency = lambda: env._control_frequency

# langtable policies
# policy_checkpoint_dir = "/home/sidhraja/projects/LaMer/checkpoints"
# policy_checkpoint_prefix = "bc_resnet_sim_checkpoint_955000"
# policy = LAVAPolicy(checkpoint_dir=policy_checkpoint_dir, checkpoint_prefix=policy_checkpoint_prefix)

# smolvla policy
policy_checkpoint_path = "/home/sidhraja/projects/language-table/outputs/smolvla_padded/checkpoints/last/pretrained_model"
policy = SmolVLAPolicy(checkpoint_path=policy_checkpoint_path)

policy.reset(num_envs=1)
time_step = env.reset(); obs = time_step.observation
frames = []

for _ in range(num_steps):
    action = policy.predict(
        goals=[instruction],
        obs_list=[obs],
        active_mask=np.array([True], dtype=bool),
    )[0]

    time_step = env.step(action)
    obs = time_step.observation

    frame = env.render(mode="rgb_array")
    frames.append(frame)
    if time_step.is_last():
        break

imageio.mimwrite("language_table.mp4", frames, fps=10)