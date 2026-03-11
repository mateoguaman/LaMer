from typing import List
from functools import partial

from agent_system.environments.navigation.env_manager import (
    NavigationEnvironmentManager,
    _DISTURBANCE_MAP,
)
from agent_system.environments.navigation.game.disturbances import (
    NoDisturbance,
    RandomChoice,
)
from agent_system.environments.navigation.envs import build_navigation_envs
from agent_system.environments.navigation.projection import navigation_projection

from .prompt import get_navigation_single_step_prompt


class NavigationSingleStepEnvironmentManager(NavigationEnvironmentManager):
    """
    Single-step variant of NavigationEnvironmentManager.

    Instead of predicting the full action sequence (e.g., "LUDR") in one shot,
    the model predicts one action character per turn.  The outer training loop
    runs up to ``env.max_steps`` turns, one LLM call per move.

    Differences from the parent:
    - ``projection_f`` uses ``n_remaining=1`` (extract 1 char per call).
    - ``build_text_obs`` calls the single-step prompt template, which asks
      for exactly 1 character rather than the full path length.
    """

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
            obs = get_navigation_single_step_prompt(
                n=self.n,
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

    def get_traj_log(self):
        return []


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def make_envs(config):
    """Build train and val NavigationSingleStepEnvironmentManager instances."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n must be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

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

    num_attempts    = config.env.get('num_attempts', 1)
    do_reflection   = config.env.get('do_reflection', True)
    val_num_attempts  = config.env.get('val_num_attempts', num_attempts)
    val_do_reflection = config.env.get('val_do_reflection', do_reflection)

    # Single-step: extract exactly 1 action character per LLM call.
    projection_f = partial(navigation_projection, n_remaining=1)

    envs = NavigationSingleStepEnvironmentManager(
        _envs, projection_f, num_attempts, do_reflection, config
    )
    val_envs = NavigationSingleStepEnvironmentManager(
        _val_envs, projection_f, val_num_attempts, val_do_reflection, config
    )
    return envs, val_envs
