from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from agent_system.environments.remote import RemoteEnvironmentManager
from .prompt import get_robolab_prompt
from .projection import robolab_projection


class RobolabEnvironmentManager:
    """Wraps a RemoteEnvironmentManager with prompt construction and projection.

    The remote server (RoboLab EnvServer) handles simulation and policy inference.
    This wrapper handles:
    - Building structured prompts incorporating the task instruction and image
    - Extracting language commands from <action> tags in LLM output
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

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0

        # Per-env tracking: language instruction (from obs["text"])
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_commands: List[Dict[int, Dict[int, str]]] = [
            {} for _ in range(self.num_processes)
        ]
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self):
        obs, infos = self._remote.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]
        self._last_commands = [{} for _ in range(self.num_processes)]

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
        assert phase in ("play", "reflect")
        if phase == "reflect":
            return self._handle_reflect_step(text_actions)
        return self._handle_play_step(text_actions)

    def restart(self):
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

    def close(self):
        self._remote.close()

    # ------------------------------------------------------------------
    # Play phase
    # ------------------------------------------------------------------

    def _handle_play_step(self, text_actions: List[str]):
        commands, valids = robolab_projection(text_actions, phase="play")

        for i, command in enumerate(commands):
            if self.curr_traj_idx not in self._last_commands[i]:
                self._last_commands[i][self.curr_traj_idx] = {}
            self._last_commands[i][self.curr_traj_idx][self.curr_turn_idx] = command

        obs, rewards, dones, infos = self._remote.step(commands, phase="play")

        for i, info in enumerate(infos):
            info["is_action_valid"] = np.array(valids[i], dtype=np.float32)

        text_obs = obs.get("text", [""] * self.num_processes)
        self._last_text_obs = (
            list(text_obs) if isinstance(text_obs, list) else [text_obs] * self.num_processes
        )
        self.curr_turn_idx += 1

        observations = {
            "text": self._build_play_prompts() if self.curr_turn_idx < self.max_turns else [""] * self.num_processes,
            "image": obs.get("image"),
            "anchor": text_obs,
        }
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Reflect phase
    # ------------------------------------------------------------------

    def _handle_reflect_step(self, text_actions: List[str]):
        reflections, valids = robolab_projection(text_actions, phase="reflect")

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
        prompts = []
        for i in range(self.num_processes):
            if self.curr_turn_idx == 0:
                curr_traj = ""
            else:
                current_commands = self._last_commands[i].get(self.curr_traj_idx, {})
                parts = [current_commands[t] for t in range(self.curr_turn_idx) if current_commands.get(t)]
                curr_traj = "; ".join(parts) if parts else ""

            past_traj = {}
            for traj_idx in range(self.curr_traj_idx):
                past_commands = self._last_commands[i].get(traj_idx, {})
                parts = [past_commands[t] for t in sorted(past_commands) if past_commands[t]]
                past_traj[traj_idx] = "; ".join(parts) if parts else ""

            prompt = get_robolab_prompt(
                phase="play",
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                language_instruction=self._init_text_obs[i],
                curr_traj=curr_traj,
                past_traj=past_traj,
                reflection=self.reflections[i],
                reflection_type=self.reflection_type,
            )
            prompts.append(prompt)
        return prompts

    def _build_reflect_prompts(self) -> List[str]:
        prompts = []
        for i in range(self.num_processes):
            current_commands = self._last_commands[i].get(self.curr_traj_idx, {})
            parts = [current_commands[t] for t in range(self.curr_turn_idx) if current_commands.get(t)]
            curr_traj = "; ".join(parts) if parts else ""

            prompt = get_robolab_prompt(
                phase="reflect",
                turn_idx=min(self.curr_turn_idx, self.max_turns - 1),
                traj_idx=self.curr_traj_idx,
                language_instruction=self._init_text_obs[i],
                curr_traj=curr_traj,
                reflection_type=self.reflection_type,
            )
            prompts.append(prompt)
        return prompts

    def __repr__(self):
        return (
            f"RobolabEnvironmentManager("
            f"remote={self._remote!r}, "
            f"num_processes={self.num_processes})"
        )


def make_envs(config, prompt_state=None):
    """Return (train_env_manager, val_env_manager) for RoboLab.

    Expects config.env to have:
        - remote_address: "host:port" for training server
        - remote_val_address: "host:port" for validation server
    """
    train_remote = RemoteEnvironmentManager(config.env.remote_address)
    val_remote = RemoteEnvironmentManager(config.env.remote_val_address)

    envs = RobolabEnvironmentManager(train_remote, config)
    val_envs = RobolabEnvironmentManager(val_remote, config)
    return envs, val_envs
