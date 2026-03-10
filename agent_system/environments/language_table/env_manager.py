"""
LaMer-side wrapper for Language Table with prompt construction and projection.

The actual environment runs in a separate process (language-table venv)
via the remote env server protocol. This module wraps RemoteEnvironmentManager
to add structured prompts and parse LLM output before forwarding to the server.
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from agent_system.environments.remote import RemoteEnvironmentManager
from .prompt import get_language_table_prompt
from .projection import language_table_projection


class LanguageTableEnvironmentManager:
    """Wraps a RemoteEnvironmentManager with prompt construction and projection.

    The remote server handles the VLA inner loop and env simulation.
    This wrapper handles:
    - Building structured prompts with history/reflection context
    - Extracting goal strings from <action> tags in LLM output
    - Tracking per-attempt state for meta-RL
    """

    def __init__(self, remote_env: RemoteEnvironmentManager, config):
        self._remote = remote_env
        self.config = config

        self.num_processes = remote_env.num_processes
        self.num_attempts = remote_env.num_attempts
        self.max_turns = remote_env.max_turns
        self.do_reflection = remote_env.do_reflection
        self.reflection_type = config.env.get("reflection_type", "reflection_only")

        # Meta-RL state
        self.curr_traj_idx = 0
        self.curr_turn_idx = 0

        # Per-env tracking
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_goals: List[Dict[int, str]] = [
            {} for _ in range(self.num_processes)
        ]
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

    # ------------------------------------------------------------------
    # Core interface (mirrors EnvironmentManagerBase)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs and return structured prompt observations."""
        obs, infos = self._remote.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]
        self._last_goals = [{} for _ in range(self.num_processes)]

        text_obs = obs.get("text", [""] * self.num_processes)
        self._init_text_obs = list(text_obs)
        self._last_text_obs = list(text_obs)

        observations = {
            "text": self._build_play_prompts(),
            "image": obs.get("image"),
            "anchor": text_obs,
        }
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        """Process LLM output through projection, then forward to remote."""
        assert phase in ("play", "reflect")

        if phase == "reflect":
            return self._handle_reflect_step(text_actions)
        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt."""
        obs, infos = self._remote.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        text_obs = obs.get("text", [""] * self.num_processes)
        self._last_text_obs = list(text_obs)

        observations = {
            "text": self._build_play_prompts(),
            "image": obs.get("image"),
            "anchor": text_obs,
        }
        return observations, infos

    def reflect(self):
        """Return prompts for the reflection phase."""
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]

        observations = {
            "text": self._build_reflect_prompts(),
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success from trajectory info dicts."""
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
                        wons[traj_idx] = wons[traj_idx] or info.get(
                            "won", False
                        )

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

        return {key: np.array(value) for key, value in success.items()}

    def close(self):
        """Close the remote connection."""
        self._remote.close()

    # ------------------------------------------------------------------
    # Play phase
    # ------------------------------------------------------------------

    def _handle_play_step(self, text_actions: List[str]):
        """Extract goal from LLM output, forward to remote server."""
        goals, valids = language_table_projection(text_actions, phase="play")

        # Track goals for history
        for i, goal in enumerate(goals):
            self._last_goals[i][self.curr_turn_idx] = goal

        # Forward extracted goals (not raw LLM output) to the remote server
        obs, rewards, dones, infos = self._remote.step(goals, phase="play")

        for i, info in enumerate(infos):
            info["is_action_valid"] = np.array(valids[i], dtype=np.float32)

        text_obs = obs.get("text", [""] * self.num_processes)
        self._last_text_obs = (
            list(text_obs) if isinstance(text_obs, list) else [text_obs] * self.num_processes
        )
        self.curr_turn_idx += 1

        observations = {
            "text": self._build_play_prompts(),
            "image": obs.get("image"),
            "anchor": text_obs,
        }
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Reflect phase
    # ------------------------------------------------------------------

    def _handle_reflect_step(self, text_actions: List[str]):
        """Extract reflection from LLM output, store for next attempt."""
        reflections, valids = language_table_projection(
            text_actions, phase="reflect"
        )

        for i, reflection in enumerate(reflections):
            self.reflections[i][self.curr_traj_idx] = reflection

        infos = [
            {
                "action_is_valid": True,
                "won": False,
                "is_action_valid": np.array(valids[i], dtype=np.float32),
            }
            for i in range(self.num_processes)
        ]
        observations = {"text": "", "image": None, "anchor": ""}
        rewards = np.array(valids, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_play_prompts(self) -> List[str]:
        """Build structured play prompts for each environment."""
        prompts = []
        for i in range(self.num_processes):
            # Build current trajectory string from tracked goals
            if self.curr_turn_idx == 0:
                curr_traj = ""
            else:
                goal_parts = []
                for t in range(self.curr_turn_idx):
                    goal = self._last_goals[i].get(t, "")
                    if goal:
                        goal_parts.append(goal)
                curr_traj = "; ".join(goal_parts) if goal_parts else ""

            # Build past trajectory dict
            past_traj = {}
            for traj_idx in range(self.curr_traj_idx):
                goals_in_traj = self._last_goals[i].get(traj_idx)
                if isinstance(goals_in_traj, str):
                    past_traj[traj_idx] = goals_in_traj
                else:
                    past_traj[traj_idx] = ""

            prompt = get_language_table_prompt(
                phase="play",
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self._init_text_obs[i],
                curr_traj=curr_traj,
                past_traj=past_traj,
                reflection=self.reflections[i],
                reflection_type=self.reflection_type,
            )
            prompts.append(prompt)
        return prompts

    def _build_reflect_prompts(self) -> List[str]:
        """Build structured reflect prompts for each environment."""
        prompts = []
        for i in range(self.num_processes):
            # Build current trajectory from tracked goals
            goal_parts = []
            for t in range(self.curr_turn_idx):
                goal = self._last_goals[i].get(t, "")
                if goal:
                    goal_parts.append(goal)
            curr_traj = "; ".join(goal_parts) if goal_parts else ""

            prompt = get_language_table_prompt(
                phase="reflect",
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self._init_text_obs[i],
                curr_traj=curr_traj,
                reflection_type=self.reflection_type,
            )
            prompts.append(prompt)
        return prompts

    def __repr__(self):
        return (
            f"LanguageTableEnvironmentManager("
            f"remote={self._remote!r}, "
            f"num_processes={self.num_processes})"
        )


def make_envs(config):
    """Return (train_env_manager, val_env_manager) for Language Table.

    Expects config.env to have:
        - remote_address: "host:port" for training server
        - remote_val_address: "host:port" for validation server
    """
    train_remote = RemoteEnvironmentManager(config.env.remote_address)
    val_remote = RemoteEnvironmentManager(config.env.remote_val_address)

    envs = LanguageTableEnvironmentManager(train_remote, config)
    val_envs = LanguageTableEnvironmentManager(val_remote, config)
    return envs, val_envs
