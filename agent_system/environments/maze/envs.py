import ray
import gym
import numpy as np
from typing import Dict, Any, Tuple, List

from .game.env import MazeEnv


@ray.remote(num_cpus=0.1)
class MazeWorker:
    """
    Ray remote actor for Maze environments.
    Each worker holds its own MazeEnv instance.
    """

    def __init__(self, env_kwargs: Dict[str, Any] = None):
        """Initialize the MazeEnv in this worker."""
        if env_kwargs is None:
            env_kwargs = {}
        self.env = MazeEnv(**env_kwargs)
        ## Code for MetaRL ## To allow restart for MetaRL, we add an env_copy to allow returning to init_state
        self.env_copy = self.env.copy()

    def step(self, action):
        """Execute a step in the environment. action is a list of direction strings."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, seed_for_reset):
        """Reset the environment with a new episode."""
        obs, info = self.env.reset(seed=seed_for_reset)
        ## Code for MetaRL ##
        self.env_copy = self.env.copy()
        return obs, info

    def render(self, mode_for_render='ascii'):
        """Render the environment."""
        return self.env.render()

    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        self.env = self.env_copy.copy()
        obs = self.env.render()
        info = {
            'won': False,
        }
        return obs, info


class MazeMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Maze environment.
    Each Ray actor creates an independent MazeEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing MazeEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors
        self.workers = []
        for i in range(self.num_processes):
            worker = MazeWorker.remote(env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list of action lists, length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='ascii', env_idx=None):
        """
        Request rendering from Ray actor environments.
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote(mode)
                futures.append(future)
            results = ray.get(futures)
            return results

    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        futures = [worker.restart.remote() for worker in self.workers]
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        """Close all Ray actors."""
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_maze_envs(
        seed=0,
        env_num=1,
        group_n=1,
        is_train=True,
        env_kwargs=None):
    return MazeMultiProcessEnv(seed, env_num, group_n, is_train, env_kwargs=env_kwargs)
