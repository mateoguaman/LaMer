"""
maze_env — Gym maze environment with exact shortest-path length.

A "step" is moving one cell in the display grid.  n must be an even integer
>= 2 (passage-based maze corners always have even-length shortest paths).

Quick start
-----------
    import gym
    import maze_env  # registers "Maze-v0"

    env = gym.make("Maze-v0", n=8)  # n must be even
    obs = env.reset(seed=42)
    print(obs)

    obs, reward, done, info = env.step(env.action_space.sample())

Custom disturbances
-------------------
    from maze_env import ActionDisturbance, MazeEnv

    class MyDisturbance(ActionDisturbance):
        def __call__(self, action: int) -> int:
            return (action + 2) % 4  # flip both axes

    env = MazeEnv(n=10, disturbances=[MyDisturbance()])

Action encoding
---------------
    UP=0  DOWN=1  LEFT=2  RIGHT=3
    String names: "up", "down", "left", "right" (ACTION_NAMES tuple).
    env.step() accepts either integers or strings.
"""

import gym

from .env import ACTION_NAMES, DOWN, LEFT, RIGHT, UP, MazeEnv
from .episodic_env import EpisodicMazeEnv
from .disturbances import (
    ActionDisturbance,
    FlipBoth,
    FlipLeftRight,
    FlipUpDown,
    NoDisturbance,
    Probabilistic,
    RandomPermutation,
    RotateActions,
)

__all__ = [
    # Environments
    "MazeEnv",
    "EpisodicMazeEnv",
    # Action constants
    "ACTION_NAMES",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    # Disturbance base
    "ActionDisturbance",
    # Built-in disturbances
    "NoDisturbance",
    "FlipLeftRight",
    "FlipUpDown",
    "FlipBoth",
    "RotateActions",
    "RandomPermutation",
    "Probabilistic",
]

gym.register(
    id="Maze-v0",
    entry_point="agent_system.environments.maze.game.env:MazeEnv",
)
