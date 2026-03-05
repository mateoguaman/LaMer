from omegaconf import DictConfig, OmegaConf

import os
import numpy as np
import random
import ray
import json
import warnings
warnings.filterwarnings('ignore')

"""python -m examples.test_env
"""

env_name = 'language_table'  # 'minesweeper' or 'sokoban' or 'webshop' or 'alfworld' or 'language_table'
os.environ['ALFWORLD_DATA']='/your/alfworld/path' # only needed for alfworld

def create_envs(config):
    if env_name == 'sokoban':
        from agent_system.environments.sokoban import make_envs
    elif env_name == 'minesweeper':
        from agent_system.environments.minesweeper import make_envs
    elif env_name == 'alfworld':
        from agent_system.environments.alfworld import make_envs
    elif env_name == 'maze':
        from agent_system.environments.maze import make_envs
    elif env_name == 'webshop':
        from agent_system.environments.webshop import make_envs
    elif env_name == 'language_table':
        from agent_system.environments.language_table import make_envs
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return make_envs(config)

def init_config() -> DictConfig:
    config = OmegaConf.load(f"verl/trainer/config/eval/{env_name}.yaml")
    return config

def random_action(obs_list, info_list):
    if env_name == 'sokoban':
        num_actions_per_turn = 3
        actions = []
        for _ in range(len(info_list)):
            action = ','.join([np.random.choice(["up", "down", "left", "right"]) for _ in range(num_actions_per_turn)])
            actions.append(f"<action>{action}</action>")
    elif env_name == 'minesweeper':
        actions = []
        for anchor_str in obs_list['anchor']:
            anchor_str = anchor_str.replace('Row 1: ', '').replace('Row 2: ', '').replace('Row 3: ', '').replace('Row 4: ', '').replace('Row 5: ', '').replace('Row 6: ', '')
            arr = [row.split() for row in anchor_str.split("\n")]
            question_marks = [(r + 1, c + 1)          # convert to 1–6 indexing
                          for r in range(6)
                          for c in range(6)
                          if arr[r][c] == '?']
            x, y = random.choice(question_marks)
            actions.append(f"<action>({x}, {y})</action>")
    elif env_name == 'alfworld':
        actions = ['<action>'+np.random.choice(_info['admissible_commands'])+'</action>' for _info in info_list]
    elif env_name == 'maze':
        plan_length = 5
        actions = []
        for _ in range(len(info_list)):
            action = ','.join([np.random.choice(["up", "down", "left", "right"]) for _ in range(plan_length)])
            actions.append(f"<action>{action}</action>")
    elif env_name == 'webshop':
        from agent_system.environments.webshop.webshop.web_agent_site.models.models import RandomPolicy
        policy = RandomPolicy()
        actions = []
        for _info in info_list:
            available_actions = _info['available_actions']
            action = policy.forward('', available_actions)
            actions.append(f"<action>{action}</action>")
    elif env_name == 'language_table':
        actions = [f"<action>push the red star to the blue cube</action>"] * len(info_list)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return actions

def run_meta_rl_episode(envs, N, num_attempts=3, max_turns=7, label=""):
    """Run a full meta-RL episode: num_attempts × (max_turns steps + reflect + restart).

    This mirrors what TrajectoryCollector.multi_turn_loop does in real training.
    """
    prompts = []

    obs_list, info_list = envs.reset()
    prompts.append(f'[{label} Attempt 0]\n' + obs_list['text'][0])

    for traj_idx in range(num_attempts):
        if traj_idx >= 1:
            obs_list, info_list = envs.reflect()
            prompts.append(f'[{label} Reflection]\n' + obs_list['text'][0])

            reflections = ['<remark>In my previous trial, I did ... I should have ...</remark>'] * N
            obs_list, reward_list, done_list, info_list = envs.step(reflections, phase='reflect')

            obs_list, info_list = envs.restart()
            prompts.append(f'[{label} Attempt {traj_idx}]\n' + obs_list['text'][0])

        for _ in range(max_turns):
            actions = random_action(obs_list, info_list)
            obs_list, reward_list, done_list, info_list = envs.step(actions, phase='play')
            prompts.append(f'[{label} Attempt {traj_idx}]\n' + obs_list['text'][0])

            if np.all(done_list):
                break

    return prompts


def main():
    ray.init(log_to_driver=False)
    config = init_config()
    config.data.train_batch_size = 1
    config.data.val_batch_size = 1
    print(config.env)
    N = config.data.val_batch_size

    train_envs, val_envs = create_envs(config)

    num_epochs = 6
    test_freq = 3  # validate every 3 epochs

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        # -- Training rollout (every epoch) --
        print(f"\n--- Train rollout (epoch {epoch}) ---")
        train_prompts = run_meta_rl_episode(
            train_envs, N, label=f"Train E{epoch}")
        for p in train_prompts:
            print(p)

        # -- Validation (periodic) --
        if epoch % test_freq == 0:
            print(f"\n--- Validation (epoch {epoch}) ---")
            val_prompts = run_meta_rl_episode(
                val_envs, N, label=f"Val E{epoch}")
            for p in val_prompts:
                print(p)

    # ---- Cleanup ----
    print(f"\n{'='*60}")
    print("Closing connections...")
    train_envs.close()
    val_envs.close()
    print("Done.")


if __name__ == '__main__':
    main()
