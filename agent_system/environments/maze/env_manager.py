from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os

from .prompt import get_maze_prompt
from .memory import SimpleMemoryMaze as SimpleMemory
from .envs import build_maze_envs
from .projection import maze_projection
from ..base import EnvironmentManagerBase


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class MazeEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        self.maze_size = config.env.maze.maze_size
        self.plan_length = config.env.maze.get('plan_length', 5)

        ## MetaRL: num_attempts >= 2
        self.num_attempts = num_attempts
        self.num_processes = envs.num_processes
        ## init states
        self.init_states = [None for _ in range(self.num_processes)]
        ## memories of previous state and actions
        self.memories = [SimpleMemory() for _ in range(self.num_attempts)]
        ## reflections
        self.reflections = [{} for _ in range(self.num_processes)]
        self.do_reflection = do_reflection
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in ['history_and_reflection', 'reflection_only', 'history_only']
        ## curr_traj_idx is used to track the current trajectory index for MetaRL
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        self.max_turns = config.env.get('max_turns', 10)
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        ## reset init states
        self.init_states = obs

        ## reset memories, reflections and curr_traj_idx
        for memory in self.memories:
            memory.reset(self.num_processes)

        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs
        }
        return observations, infos

    def restart(self):
        ''' Used for 2nd or N-th attempts '''
        obs, infos = self.envs.restart()
        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs
        }

        return observations, infos

    def reflect(self):
        '''Get prompts for reflect phase.'''
        infos = [{
            "action_is_valid": True,
            "won": False
        } for _ in range(self.num_processes)]

        observations = {
            'text': self.build_text_obs(phase='reflect'),
            'image': None,
            'anchor': ['reflection' for _ in self.build_text_obs(phase='reflect')]
        }

        return observations, infos

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'reflect']
        if phase == 'reflect':
            # extract reflection from text_actions
            reflections, valids = self.projection_f(text_actions, phase='reflect')

            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

            infos = [{
                "action_is_valid": False,
                "won": False
            } for _ in range(self.num_processes)]

            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            next_observations = {
                'text': '',
                'image': None,
                'anchor': ''
            }
            rewards = np.array(valids)
            dones = np.array([False] * len(text_actions))
            return next_observations, rewards, dones, infos

        else:
            # Parse the multi-step plan from text
            thoughts, actions, valids = self.projection_f(text_actions, phase='play')

            # Execute the full plan on the environment
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # add action_valid to infos
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            # Format actions for memory: show planned vs executed
            planned_strs = []
            executed_strs = []
            for i in range(self.num_processes):
                if not valids[i]:
                    planned_strs.append('no operation')
                    executed_strs.append('no operation')
                else:
                    planned_strs.append(','.join(actions[i]))
                    exec_actions = infos[i].get('executed_actions', actions[i])
                    executed_strs.append(','.join(exec_actions))

            self.memories[self.curr_traj_idx].store({
                'text_obs': next_obs,
                'planned_actions': planned_strs,
                'executed_actions': executed_strs,
                'reward': rewards,
                'dones': dones,
                'won': [info['won'] for info in infos]
            })
            self.curr_turn_idx += 1
            next_observations = {
                'text': self.build_text_obs(phase='play'),
                'image': None,
                'anchor': next_obs
            }

            rewards = to_numpy(rewards)
            dones = to_numpy(dones)

            return next_observations, rewards, dones, infos

    def build_text_obs(self, phase: str = 'play') -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        assert phase in ['play', 'reflect']

        obs_length = 2 if phase == 'play' else 7
        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(obs_length=obs_length)

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajectories, _ = self.memories[traj_idx].fetch()
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]

        for i in range(self.num_processes):
            obs = get_maze_prompt(
                maze_size=self.maze_size,
                plan_length=self.plan_length,
                phase=phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self.init_states[i],
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                reflection=self.reflections[i],
                reflection_type=self.reflection_type
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


def make_envs(config):
    """
    Create environments
    """
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    if "maze" in config.env.env_name.lower():
        env_kwargs = {
            "maze_size": config.env.maze.maze_size,
            "max_episode_steps": config.env.get('max_steps', 50),
            "plan_length": config.env.maze.get('plan_length', 5),
            "sticky_prob": config.env.maze.get('sticky_prob', 0.0),
            "double_prob": config.env.maze.get('double_prob', 0.0),
            "noise_prob": config.env.maze.get('noise_prob', 0.0),
            "unavail_actions": list(config.env.maze.get('unavail_actions', [])),
        }
        _envs = build_maze_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            env_kwargs=env_kwargs
        )
        _val_envs = build_maze_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs
        )

        num_attempts = config.env.get('num_attempts', 1)
        do_reflection = config.env.get('do_reflection', True)
        val_num_attempts = config.env.get('val_num_attempts', num_attempts)
        val_do_reflection = config.env.get('val_do_reflection', do_reflection)

        projection_f = partial(maze_projection, plan_length=config.env.maze.get('plan_length', 5))

        envs = MazeEnvironmentManager(_envs, projection_f, num_attempts, do_reflection, config)
        val_envs = MazeEnvironmentManager(_val_envs, projection_f, val_num_attempts, val_do_reflection, config)
        return envs, val_envs

    else:
        print("Environment not supported")
        exit(1)
