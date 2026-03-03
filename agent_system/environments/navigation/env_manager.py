from typing import List, Dict, Any
from functools import partial

import torch
import numpy as np

from .prompt import get_navigation_prompt
from .memory import SimpleMemoryNavigation as SimpleMemory
from .envs import build_navigation_envs
from .projection import navigation_projection
from .game.disturbances import (
    CyclicRotation,
    FlipAntiDiagonal,
    FlipBoth,
    FlipDiagonal,
    FlipLeftRight,
    FlipUpDown,
    NoDisturbance,
    RandomChoice,
    RandomPermutation,
    RotateActions,
)
from ..base import EnvironmentManagerBase

_DISTURBANCE_MAP = {
    'FlipLeftRight':     FlipLeftRight,
    'FlipUpDown':        FlipUpDown,
    'FlipBoth':          FlipBoth,
    'FlipDiagonal':      FlipDiagonal,
    'FlipAntiDiagonal':  FlipAntiDiagonal,
    'CyclicRotation':    CyclicRotation,
    'RotateActions':     RotateActions,
    'RandomPermutation': RandomPermutation,
    'NoDisturbance':     NoDisturbance,
}


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, bool, tuple, list)):
        return np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)}")


class NavigationEnvironmentManager(EnvironmentManagerBase):
    """
    Orchestrates the Navigation environment for the LaMer/GiGPO training loop.

    Architecture
    ------------
    - ``envs``         : ``NavigationMultiProcessEnv`` — Ray-based parallel workers.
    - ``projection_f`` : ``navigation_projection`` — parses LLM text -> action string.
    - ``memories``     : one ``SimpleMemoryNavigation`` per attempt (for MetaRL).
    - ``reflections``  : stores per-env reflection strings across attempts.

    The navigation env is *episodic*: each ``step`` call accepts one action
    string of length ``n`` (the path length) and executes all moves at once.
    MetaRL allows multiple attempts on the same grid instance via ``restart``.
    """

    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        self.n = config.env.navigation.n
        self.grid_size = 2 * self.n + 1
        self.num_attempts = num_attempts
        self.num_processes = envs.num_processes

        self.init_states = [None] * self.num_processes
        self.memories = [SimpleMemory() for _ in range(self.num_attempts)]
        self.reflections = [{} for _ in range(self.num_processes)]
        self.do_reflection = do_reflection
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in [
            'history_and_reflection', 'reflection_only', 'history_only'
        ]

        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        self.max_turns = config.env.get('max_turns', 1)
        super().__init__(envs, projection_f, config)

    # ---------------------------------------------------------------------- #
    # Episode lifecycle                                                        #
    # ---------------------------------------------------------------------- #

    def reset(self):
        obs, infos = self.envs.reset()
        self.init_states = obs

        for memory in self.memories:
            memory.reset(self.num_processes)

        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs,
        }
        return observations, infos

    def restart(self):
        """Used for the 2nd / Nth MetaRL attempt on the same grid instance."""
        obs, infos = self.envs.restart()
        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs,
        }
        return observations, infos

    def reflect(self):
        """Return prompts for the reflect phase (called between MetaRL attempts)."""
        infos = [{'action_is_valid': True, 'won': False}
                 for _ in range(self.num_processes)]
        text_obs = self.build_text_obs(phase='reflect')
        observations = {
            'text': text_obs,
            'image': None,
            'anchor': ['reflection'] * len(text_obs),
        }
        return observations, infos

    # ---------------------------------------------------------------------- #
    # Stepping                                                                 #
    # ---------------------------------------------------------------------- #

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'reflect']

        if phase == 'reflect':
            reflections, valids = self.projection_f(text_actions, phase='reflect')
            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

            infos = [{'action_is_valid': False, 'won': False}
                     for _ in range(self.num_processes)]
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            next_observations = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.array(valids)
            dones = np.array([False] * len(text_actions))
            return next_observations, rewards, dones, infos

        # ------------------------------------------------------------------ #
        # Play phase                                                           #
        # ------------------------------------------------------------------ #
        thoughts, actions, valids = self.projection_f(text_actions, phase='play')
        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        stored_actions = [
            a if valids[i] else 'no operation'
            for i, a in enumerate(actions)
        ]

        self.memories[self.curr_traj_idx].store({
            'text_obs': next_obs,
            'thought': thoughts,
            'action': stored_actions,
            'reward': rewards,
            'dones': dones,
            'won': [info['won'] for info in infos],
        })
        self.curr_turn_idx += 1

        next_observations = {
            'text': self.build_text_obs(phase='play'),
            'image': None,
            'anchor': next_obs,
        }
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    # ---------------------------------------------------------------------- #
    # Observation builder                                                      #
    # ---------------------------------------------------------------------- #

    def build_text_obs(self, phase: str = 'play') -> List[str]:
        assert phase in ['play', 'reflect']

        obs_length = 2 if phase == 'play' else 7

        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(
                obs_length=obs_length
            )

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajectories, _ = self.memories[traj_idx].fetch()
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]

        postprocess_text_obs = []
        for i in range(self.num_processes):
            obs = get_navigation_prompt(
                n_remaining=self.n,
                grid_size=self.grid_size,
                phase=phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self.init_states[i],
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                reflection=self.reflections[i],
                reflection_type=self.reflection_type,
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def make_envs(config):
    """Build train and validation ``NavigationEnvironmentManager`` instances."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n must be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    if "navigation" not in config.env.env_name.lower():
        print("Environment not supported")
        exit(1)

    legacy_names            = list(config.env.navigation.get('disturbances', []))
    train_disturbance_names = list(config.env.navigation.get('train_disturbances', legacy_names))
    val_disturbance_names   = list(config.env.navigation.get('val_disturbances',   legacy_names))

    def _build_disturbance(names: list):
        for name in names:
            if name not in _DISTURBANCE_MAP:
                raise ValueError(
                    f"Unknown disturbance '{name}'. "
                    f"Valid options: {list(_DISTURBANCE_MAP.keys())}"
                )
        if len(names) == 0:
            return NoDisturbance()
        if len(names) == 1:
            return _DISTURBANCE_MAP[names[0]]()
        return RandomChoice([_DISTURBANCE_MAP[n]() for n in names])

    train_env_kwargs = {
        "n": config.env.navigation.n,
        "disturbances": [_build_disturbance(train_disturbance_names)],
    }
    val_env_kwargs = {
        "n": config.env.navigation.n,
        "disturbances": [_build_disturbance(val_disturbance_names)],
    }

    _envs = build_navigation_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_kwargs=train_env_kwargs,
    )
    _val_envs = build_navigation_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=val_env_kwargs,
    )

    num_attempts = config.env.get('num_attempts', 1)
    do_reflection = config.env.get('do_reflection', True)
    val_num_attempts = config.env.get('val_num_attempts', num_attempts)
    val_do_reflection = config.env.get('val_do_reflection', do_reflection)

    projection_f = partial(navigation_projection, n_remaining=config.env.navigation.n)

    envs = NavigationEnvironmentManager(
        _envs, projection_f, num_attempts, do_reflection, config
    )
    val_envs = NavigationEnvironmentManager(
        _val_envs, projection_f, val_num_attempts, val_do_reflection, config
    )
    return envs, val_envs
