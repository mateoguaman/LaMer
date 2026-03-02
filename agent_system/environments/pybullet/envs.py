# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray-parallel vectorized wrapper for PyBullet (or any gym-compatible) environments.

Why this works with PyBullet
-----------------------------
PyBullet maintains one physics server **per process** (identified by a
``physicsClientId``).  Ray actors run in separate OS processes, so each
worker gets its own isolated physics simulation with no shared-state
conflicts — the same reason this pattern already works for MazeEnv and
SokobanEnv in this codebase.

Usage
-----
    from agent_system.environments.pybullet.envs import build_pybullet_envs

    envs = build_pybullet_envs(
        env_id="AntBulletEnv-v0",
        seed=0,
        env_num=16,   # number of *distinct* environments
        group_n=4,    # GRPO/GiGPO repetitions per environment
    )
    obs_list, info_list = envs.reset()
    obs_list, rew_list, done_list, info_list = envs.step(actions)
"""

import ray
import gym
import numpy as np
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Helper: import pybullet_envs to register gym env IDs, ignore if absent
# ---------------------------------------------------------------------------
def _try_import_pybullet_envs():
    try:
        import pybullet_envs  # noqa: F401  registers AntBulletEnv-v0 etc.
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Ray remote worker — one per sub-environment
# ---------------------------------------------------------------------------
@ray.remote(num_cpus=0.5)
class PyBulletWorker:
    """Ray remote actor that owns a single gym-compatible environment.

    ``num_cpus=0.5`` reflects that PyBullet physics is CPU-bound; adjust
    upward (e.g. 1.0) if you see CPU contention or downward (0.1) if your
    envs are lightweight and you want higher packing density.

    The worker handles both the gym <=0.25 API
    ``(obs, reward, done, info)`` and the gym 0.26+ API
    ``(obs, reward, terminated, truncated, info)``.
    """

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        _try_import_pybullet_envs()
        self.env = gym.make(env_id, **(env_kwargs or {}))

    # ------------------------------------------------------------------
    def step(self, action: List[float]):
        """Execute one environment step.

        Parameters
        ----------
        action:
            Flat list of floats (Box) or a single int (Discrete).

        Returns
        -------
        obs, reward, done, info
        """
        action_arr = np.array(action, dtype=np.float32)
        result = self.env.step(action_arr)

        if len(result) == 5:
            # gym 0.26+
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            # gym <=0.25
            obs, reward, done, info = result
            done = bool(done)

        info.setdefault("won", False)
        return obs.tolist(), float(reward), done, info

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        """Reset the environment.

        Returns
        -------
        obs (list), info (dict)
        """
        kwargs = {"seed": seed} if seed is not None else {}
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        return obs.tolist(), info

    # ------------------------------------------------------------------
    def action_space_info(self) -> Dict[str, Any]:
        """Return serialisable action-space metadata (for random-action generation)."""
        import gym.spaces as spaces
        sp = self.env.action_space
        if isinstance(sp, spaces.Box):
            return {
                "type": "Box",
                "shape": list(sp.shape),
                "low": sp.low.tolist(),
                "high": sp.high.tolist(),
            }
        if isinstance(sp, spaces.Discrete):
            return {"type": "Discrete", "n": int(sp.n)}
        return {"type": type(sp).__name__}

    # ------------------------------------------------------------------
    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Vectorised multi-process environment (mirrors Maze/Sokoban pattern)
# ---------------------------------------------------------------------------
class PyBulletMultiProcessEnv(gym.Env):
    """Vectorised wrapper over N Ray-parallel PyBullet workers.

    Each worker runs in a separate OS process with its own pybullet physics
    server.  All ``step`` / ``reset`` calls are dispatched concurrently via
    ``ray.remote`` and collected with ``ray.get``.

    Parameters
    ----------
    env_id:
        A gym-registered environment ID (e.g. ``"AntBulletEnv-v0"``).
    seed:
        Base random seed.
    env_num:
        Number of *distinct* environments (different seeds / tasks).
    group_n:
        Number of copies of each environment (for GRPO / GiGPO grouping).
        Total workers = ``env_num * group_n``.
    is_train:
        If True, seeds are sampled from ``[0, 2**16)``;
        otherwise from ``[2**16, 2**32)`` to avoid train/val overlap.
    env_kwargs:
        Extra keyword arguments forwarded to ``gym.make``.
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.env_id = env_id
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        _try_import_pybullet_envs()

        self.workers: List[PyBulletWorker] = [
            PyBulletWorker.remote(env_id, env_kwargs or {})
            for _ in range(self.num_processes)
        ]

        # Cache action-space info (fetched once from the first worker)
        self._action_space_info: Optional[Dict] = None

    # ------------------------------------------------------------------
    def _get_action_space_info(self) -> Dict[str, Any]:
        if self._action_space_info is None:
            self._action_space_info = ray.get(
                self.workers[0].action_space_info.remote()
            )
        return self._action_space_info

    # ------------------------------------------------------------------
    def sample_random_actions(self) -> List:
        """Generate one random action per worker (useful for testing / benchmarks)."""
        info = self._get_action_space_info()
        if info["type"] == "Box":
            low = np.array(info["low"], dtype=np.float32)
            high = np.array(info["high"], dtype=np.float32)
            shape = tuple(info["shape"])
            return [
                np.random.uniform(low, high, shape).tolist()
                for _ in range(self.num_processes)
            ]
        if info["type"] == "Discrete":
            return [int(np.random.randint(info["n"])) for _ in range(self.num_processes)]
        raise NotImplementedError(f"Unsupported action space: {info['type']}")

    # ------------------------------------------------------------------
    def step(self, actions: List) -> tuple:
        """Parallel step across all workers.

        Parameters
        ----------
        actions:
            List of length ``num_processes``.  Each element is the action for
            the corresponding worker (list of floats for Box, int for Discrete).

        Returns
        -------
        obs_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes, (
            f"Expected {self.num_processes} actions, got {len(actions)}"
        )
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    # ------------------------------------------------------------------
    def reset(self) -> tuple:
        """Parallel reset across all workers with fresh seeds.

        Returns
        -------
        obs_list, info_list
        """
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Each group copy shares the same seed (same starting state)
        seeds = np.repeat(seeds, self.group_n).tolist()

        futures = [w.reset.remote(s) for w, s in zip(self.workers, seeds)]
        results = ray.get(futures)

        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    # ------------------------------------------------------------------
    def render(self, mode: str = "rgb_array", env_idx: Optional[int] = None):
        """Render one or all environments.  Note: PyBullet workers run
        headless (``pybullet.DIRECT``), so this returns pixel arrays only
        when the underlying env supports software rendering."""
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote(mode))
        futures = [w.render.remote(mode) for w in self.workers]
        return ray.get(futures)

    # ------------------------------------------------------------------
    def close(self):
        for w in self.workers:
            ray.kill(w)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory function (mirrors build_maze_envs / build_sokoban_envs)
# ---------------------------------------------------------------------------
def build_pybullet_envs(
    env_id: str,
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> PyBulletMultiProcessEnv:
    """Convenience factory that matches the signature of other env builders."""
    return PyBulletMultiProcessEnv(
        env_id=env_id,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
