"""
navigation_env — Open 2D grid navigation environment.

The agent starts at the center of a (2n+1) x (2n+1) grid and must reach
a goal placed exactly n Manhattan-distance steps away.  The grid has no
walls; every cell is passable.  Optional action disturbances make the
environment adversarial.

Quick start
-----------
    import gym
    import agent_system.environments.navigation.game  # registers "Navigation-v0"

    env = gym.make("Navigation-v0", n=5)
    obs = env.reset(seed=42)
    print(obs)

    obs, reward, done, info = env.step(env.action_space.sample())

Custom disturbances
-------------------
    from agent_system.environments.navigation.game import ActionDisturbance, NavigationEnv

    class MyDisturbance(ActionDisturbance):
        def __call__(self, action: int) -> int:
            return (action + 2) % 4  # flip both axes

    env = NavigationEnv(n=5, disturbances=[MyDisturbance()])

Action encoding
---------------
    UP=0  DOWN=1  LEFT=2  RIGHT=3
    String names: "up", "down", "left", "right" (ACTION_NAMES tuple).
    env.step() accepts either integers or strings.
"""

import gym

from .env import ACTION_NAMES, DOWN, LEFT, RIGHT, UP, NavigationEnv
from .episodic_env import EpisodicNavigationEnv
from .disturbances import (
    ActionDisturbance,
    CyclicRotation,
    FlipAntiDiagonal,
    FlipBoth,
    FlipDiagonal,
    FlipLeftRight,
    FlipUpDown,
    NoDisturbance,
    Probabilistic,
    RandomPermutation,
    RandomChoice,
    RotateActions,
)

__all__ = [
    # Environments
    "NavigationEnv",
    "EpisodicNavigationEnv",
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
    "FlipDiagonal",
    "FlipAntiDiagonal",
    "CyclicRotation",
    "RotateActions",
    "RandomPermutation",
    "RandomChoice",
    "Probabilistic",
]

gym.register(
    id="Navigation-v0",
    entry_point="agent_system.environments.navigation.game.env:NavigationEnv",
)
