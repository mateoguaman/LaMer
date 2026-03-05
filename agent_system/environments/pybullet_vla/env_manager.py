"""
PyBulletVLAEnvironmentManager — environment manager where each outer
``step()`` triggers an entire VLA inner-loop episode.

From the outer LLM's perspective:
    1. ``reset()``  → text description of initial pybullet state
    2. ``step(goal_strings)`` → VLA runs full episode → (text_obs, rewards, dones, infos)
    3. ``reflect()`` / ``restart()`` → meta-RL attempt loop (same as Sokoban)

Internally, each ``step()`` does:
    for inner_step in range(max_inner_steps):
        vla_actions = vla.predict(goals, observations, active_mask)
        pybullet_obs, rewards, dones, infos = pybullet_envs.step(vla_actions)
        accumulate rewards, update active_mask
    convert final states to text → return to LLM
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import EnvironmentManagerBase
from .state_to_text import batch_state_to_text
from .vla_policy import VLAPolicy

logger = logging.getLogger(__name__)


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    return np.array(data)


class PyBulletVLAEnvironmentManager(EnvironmentManagerBase):
    """Environment manager that wraps pybullet envs + a frozen VLA policy.

    The VLA is part of the *environment* from the outer LLM's perspective.
    Each ``step(goal_strings)`` runs the VLA for a full inner episode and
    returns the outcome as text.

    Parameters
    ----------
    envs
        Vectorised pybullet environment (gym-like interface with
        ``reset()``, ``step(actions)``, ``restart()``, ``close()``).
        Must expose ``num_processes`` and return state dicts as observations.
    vla_policy : VLAPolicy
        Frozen VLA model used for inner-loop action generation.
    config
        OmegaConf configuration object.
    """

    def __init__(self, envs, vla_policy: VLAPolicy, config):
        # We don't use a projection_f — the VLA *is* the action mapping.
        super().__init__(envs, projection_f=None, config=config)

        self.vla = vla_policy
        self.num_processes = envs.num_processes
        self.num_attempts = config.env.get("num_attempts", 1)
        self.max_turns = config.env.get("max_turns", 1)
        self.do_reflection = config.env.get("do_reflection", False)
        self.reflection_type = config.env.get("reflection_type", "reflection_only")

        # VLA inner-loop config
        self.max_inner_steps = config.env.get("max_inner_steps", 100)
        self.reward_shaping = config.env.get("reward_shaping", "sum")

        # Meta-RL state
        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

        # Cached initial text observations (set on reset)
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_infos: List[Dict] = [{} for _ in range(self.num_processes)]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all pybullet envs and return text observations."""
        obs, infos = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]

        text_obs = batch_state_to_text(obs)
        self._init_text_obs = text_obs
        self._last_text_obs = text_obs
        self._last_infos = infos

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": text_obs,
        }
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        """Execute a step in the meta-RL loop.

        When ``phase="play"``, ``text_actions`` are goal strings for the VLA.
        The VLA runs a full inner episode and we return the outcome.

        When ``phase="reflect"``, ``text_actions`` are reflections from the LLM.
        We store them and return dummy observations (same as Sokoban).
        """
        assert phase in ("play", "reflect")

        if phase == "reflect":
            return self._handle_reflect_step(text_actions)

        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt."""
        obs, infos = self.envs.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        text_obs = batch_state_to_text(obs)
        self._last_text_obs = text_obs
        self._last_infos = infos

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": text_obs,
        }
        return observations, infos

    def reflect(self):
        """Return prompts for the reflection phase."""
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]

        # Build reflection prompt: include last episode outcome.
        reflect_obs = self._build_reflect_prompt()

        observations = {
            "text": reflect_obs,
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success (same logic as base class)."""
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False for _ in range(self.num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

        return {key: np.array(value) for key, value in success.items()}

    # ------------------------------------------------------------------
    # VLA inner loop
    # ------------------------------------------------------------------

    def _handle_play_step(self, goal_strings: List[str]):
        """Run the VLA inner loop for a full episode.

        Parameters
        ----------
        goal_strings : list[str]
            One goal per env — the outer LLM's action.

        Returns
        -------
        tuple
            (obs_dict, rewards, dones, infos) — same shape as any env step.
        """
        batch = self.num_processes
        active_mask = np.ones(batch, dtype=bool)
        total_rewards = np.zeros(batch, dtype=np.float32)
        final_dones = np.zeros(batch, dtype=bool)
        last_obs = None
        last_infos = [{} for _ in range(batch)]

        for inner_step in range(self.max_inner_steps):
            # Get current observations from pybullet.
            # (After reset/restart, envs already hold the current state.)
            if inner_step == 0:
                # Use whatever state the envs are currently in.
                current_obs = self._get_current_observations()
            else:
                current_obs = last_obs

            # VLA predicts actions for active envs.
            vla_actions = self.vla.predict(goal_strings, current_obs, active_mask)

            # Step pybullet envs.
            obs, rewards, dones, infos = self.envs.step(vla_actions)

            rewards = to_numpy(rewards)
            dones = to_numpy(dones)

            # Accumulate rewards for active envs.
            total_rewards += rewards * active_mask

            # Update state for envs that just finished.
            newly_done = dones & active_mask
            final_dones |= newly_done
            active_mask &= ~dones

            last_obs = obs
            for i in range(batch):
                if newly_done[i] or inner_step == self.max_inner_steps - 1:
                    last_infos[i] = infos[i]

            # All envs done — early exit.
            if not active_mask.any():
                break

        # Mark envs that hit the step limit as done.
        final_dones |= active_mask  # timed-out envs
        active_mask[:] = False

        # Convert final pybullet state to text for the LLM.
        text_obs = batch_state_to_text(last_obs if last_obs is not None else [{}] * batch)
        self._last_text_obs = text_obs
        self._last_infos = last_infos
        self.curr_turn_idx += 1

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": text_obs,
        }

        # Add validity info (VLA actions are always "valid").
        for info in last_infos:
            info["is_action_valid"] = np.array(1.0)

        return observations, total_rewards, final_dones, last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Store reflections from the LLM (no env interaction)."""
        for i, reflection in enumerate(text_actions):
            self.reflections[i][self.curr_traj_idx] = reflection

        infos = [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": "",
            "image": None,
            "anchor": "",
        }
        rewards = np.zeros(self.num_processes, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_observations(self):
        """Get current pybullet state without stepping.

        Falls back to returning empty dicts if the env doesn't support
        a ``get_obs()`` method (the VLA will still run, just with no obs
        on the first inner step).
        """
        if hasattr(self.envs, "get_obs"):
            return self.envs.get_obs()
        return [{}] * self.num_processes

    def _build_reflect_prompt(self) -> List[str]:
        """Build text prompts for the reflection phase.

        Includes the initial state, the last episode outcome, and
        any previous reflections.
        """
        prompts = []
        for i in range(self.num_processes):
            parts = [f"Initial state:\n{self._init_text_obs[i]}"]
            parts.append(f"\nEpisode outcome:\n{self._last_text_obs[i]}")

            # Include previous reflections.
            for traj_idx in sorted(self.reflections[i]):
                parts.append(
                    f"\nReflection (attempt {traj_idx}):\n"
                    f"{self.reflections[i][traj_idx]}"
                )

            parts.append(
                "\nBased on the episode outcome, reflect on what went wrong "
                "and propose a new goal for the next attempt."
            )
            prompts.append("\n".join(parts))
        return prompts

    def build_text_obs(self, phase: str = "play") -> List[str]:
        """Build text observations (required by base class interface)."""
        if phase == "reflect":
            return self._build_reflect_prompt()
        return self._last_text_obs


def make_envs(config):
    """Factory function matching the pattern used by other env managers.

    Expected config keys under ``config.env.pybullet_vla``:
        - ``vla_checkpoint``: path to VLA checkpoint
        - ``vla_device``: torch device (default ``"cuda"``)
        - ``vla_batch_size``: max VLA batch size (default 128)
        - ``env_kwargs``: dict of pybullet env constructor args

    Returns
    -------
    tuple
        ``(train_env_manager, val_env_manager)``

    Raises
    ------
    NotImplementedError
        If pybullet env construction is not configured. This factory
        is a template — concrete pybullet env setup must be provided
        by the user.
    """
    vla_cfg = config.env.get("pybullet_vla", {})

    # --- VLA policy ---
    checkpoint = vla_cfg.get("vla_checkpoint", "")
    device = vla_cfg.get("vla_device", "cuda")
    vla_batch_size = vla_cfg.get("vla_batch_size", 128)

    # Use DummyVLAPolicy if no checkpoint is provided (testing mode).
    if not checkpoint:
        from .vla_policy import DummyVLAPolicy
        vla = DummyVLAPolicy(
            device=device,
            batch_size=vla_batch_size,
            action_dim=vla_cfg.get("action_dim", 7),
        )
    else:
        raise NotImplementedError(
            "Concrete VLA model loading not yet implemented. "
            "Subclass VLAPolicy and register it, or provide a "
            "checkpoint-specific loader in this factory."
        )

    # --- Pybullet environments ---
    # This section must be customised for the specific pybullet task.
    # The pattern mirrors build_sokoban_envs / build_minesweeper_envs.
    env_factory = vla_cfg.get("env_factory", None)
    if env_factory is not None:
        import importlib
        module_path, func_name = env_factory.rsplit(".", 1)
        module = importlib.import_module(module_path)
        build_envs = getattr(module, func_name)

        env_kwargs = dict(vla_cfg.get("env_kwargs", {}))
        train_envs = build_envs(
            num_envs=config.data.train_batch_size,
            is_train=True,
            **env_kwargs,
        )
        val_envs = build_envs(
            num_envs=config.data.val_batch_size,
            is_train=False,
            **env_kwargs,
        )
    else:
        raise NotImplementedError(
            "No pybullet env factory configured. Set "
            "config.env.pybullet_vla.env_factory to a dotted path like "
            "'my_module.build_pybullet_envs'."
        )

    train_manager = PyBulletVLAEnvironmentManager(train_envs, vla, config)
    val_manager = PyBulletVLAEnvironmentManager(val_envs, vla, config)
    return train_manager, val_manager
