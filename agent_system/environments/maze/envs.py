import ray
import gym
import numpy as np
from typing import Dict, Any, List, Tuple

from .game.episodic_env import EpisodicMazeEnv


@ray.remote(num_cpus=0.1)
class MazeWorker:
    """
    Ray remote actor that owns one ``EpisodicMazeEnv`` instance.

    Each worker runs in its own process so that many environments can be
    stepped in parallel without the GIL.
    """

    def __init__(self, env_kwargs: Dict[str, Any] = None):
        env_kwargs = env_kwargs or {}
        self.env = EpisodicMazeEnv(**env_kwargs)
        self._env_copy = None

    def step(self, action: str) -> Tuple:
        """Execute the full action string and return (obs, reward, done, info)."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, seed_for_reset: int) -> Tuple:
        """Reset to a new episode; save a copy for MetaRL restart."""
        obs, info = self.env.reset(seed=seed_for_reset)
        self._env_copy = self.env.copy()
        return obs, info

    def restart(self) -> Tuple:
        """Restore the environment to the state saved at the last reset (MetaRL)."""
        self.env = self._env_copy.copy()
        obs = self.env.render()
        info = {'won': False}
        return obs, info


class MazeMultiProcessEnv(gym.Env):
    """
    Ray-based vectorised wrapper for ``EpisodicMazeEnv``.

    Launches ``env_num * group_n`` Ray workers.  Identical seeds are shared
    within each group of ``group_n`` workers (for GRPO / GiGPO).

    Parameters
    ----------
    seed : int
        Base RNG seed.
    env_num : int
        Number of *distinct* environments (unique seeds).
    group_n : int
        Number of copies of each environment (same seed, different rollouts).
    is_train : bool
        Training envs draw seeds from [0, 2¹⁶); validation from [2¹⁶, 2³²).
    env_kwargs : dict
        Forwarded verbatim to ``EpisodicMazeEnv.__init__``.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        env_kwargs = env_kwargs or {}
        self.workers: List = [
            MazeWorker.remote(env_kwargs) for _ in range(self.num_processes)
        ]

    # ---------------------------------------------------------------------- #
    # Core interface                                                           #
    # ---------------------------------------------------------------------- #

    def step(self, actions: List[str]):
        """
        Step all workers in parallel.

        Parameters
        ----------
        actions : List[str]
            One action string per worker (e.g. ``["LRUD", "ULDR", ...]``).

        Returns
        -------
        obs_list, reward_list, done_list, info_list : each a list of length
        ``num_processes``.
        """
        assert len(actions) == self.num_processes
        futures = [
            w.step.remote(a) for w, a in zip(self.workers, actions)
        ]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def reset(self):
        """
        Reset all workers in parallel, assigning seeds by group.

        Returns
        -------
        obs_list, info_list : each a list of length ``num_processes``.
        """
        rng = np.random.randint
        if self.is_train:
            seeds = rng(0, 2 ** 16 - 1, size=self.env_num)
        else:
            seeds = rng(2 ** 16, 2 ** 32 - 1, size=self.env_num)

        seeds = np.repeat(seeds, self.group_n).tolist()
        futures = [w.reset.remote(s) for w, s in zip(self.workers, seeds)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def restart(self):
        """Restore all workers to their last-reset state (MetaRL)."""
        futures = [w.restart.remote() for w in self.workers]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def close(self):
        for w in self.workers:
            ray.kill(w)

    def __del__(self):
        self.close()


def build_maze_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: Dict[str, Any] = None,
) -> MazeMultiProcessEnv:
    return MazeMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
