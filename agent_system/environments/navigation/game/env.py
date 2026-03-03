from __future__ import annotations

import random

import numpy as np
import gym
from gym import spaces

from .disturbances import ActionDisturbance

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

ACTION_NAMES = ("up", "down", "left", "right")
_ACTION_TO_INT = {name: i for i, name in enumerate(ACTION_NAMES)}

_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _render_text(G, agent, goal, remaining):
    """ASCII string used by render()."""
    lines = []
    for r in range(G):
        chars = []
        for c in range(G):
            if (r, c) == agent:
                chars.append("A")
            elif (r, c) == goal:
                chars.append("G")
            else:
                chars.append(".")
        lines.append("".join(chars))
    hdr = "Agent: ({},{})  Goal: ({},{})  Steps remaining: {}".format(
        agent[0], agent[1], goal[0], goal[1], remaining)
    return hdr + "\n" + "\n".join(lines)


def _make_obs(G, agent, goal, remaining):
    """Build the dict observation returned by reset() and step()."""
    grid_arr = np.zeros((G, G), dtype=np.uint8)
    grid_arr[agent[0], agent[1]] = 2   # agent
    grid_arr[goal[0], goal[1]] = 3     # goal
    return {
        "agent_pos":       np.array(agent,       dtype=np.int32),
        "goal_pos":        np.array(goal,         dtype=np.int32),
        "steps_remaining": np.array([remaining], dtype=np.int32),
        "grid":            grid_arr,
    }


def _sample_goal(n, center, rng):
    """
    Sample a goal position uniformly from all grid cells that are exactly
    Manhattan distance ``n`` from ``center``.  These form a diamond of 4*n
    points (for n >= 1).
    """
    G = 2 * n + 1
    candidates = [
        (r, c)
        for r in range(G)
        for c in range(G)
        if abs(r - center[0]) + abs(c - center[1]) == n
    ]
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


class NavigationEnv(gym.Env):
    """
    Open 2D grid navigation environment with exact shortest-path length n.

    The grid has no walls — every cell is passable.  The agent always starts
    at the center cell ``(n, n)`` of a ``(2n+1) x (2n+1)`` grid.  On each
    reset the goal is placed uniformly at random among the ``4n`` cells that
    are exactly ``n`` Manhattan-distance steps from the center, guaranteeing
    that the shortest path (without disturbances) is exactly ``n`` steps.

    Moving into a boundary would move the agent out of the grid; in that case
    the agent stays in place.

    Observation space: Dict
        "agent_pos"      : Box(2,)    int32  — (row, col)
        "goal_pos"       : Box(2,)    int32  — (row, col)
        "steps_remaining": Box(1,)    int32  — Manhattan distance to goal
        "grid"           : Box(G, G)  uint8  — 0=empty 2=agent 3=goal
                           G = 2*n + 1

    Action space: Discrete(4) — step() accepts integers OR strings.
        0="up"  1="down"  2="left"  3="right"

    Reward:
        sparse=False (default): 1 - dist/n  in [0, 1], where dist is the
            Manhattan distance to the goal.  Equals 0 at the start, 1 at goal.
        sparse=True: 0 everywhere, +1 only when the agent reaches the goal.

    Terminates when agent reaches goal cell.

    Parameters
    ----------
    n : int                      Shortest-path length in steps (>= 1).
    disturbances : list          ActionDisturbance instances applied in order.
    max_steps : int              Episode length limit (default 4*n).
    sparse : bool                If True, use sparse {0, 1} reward.
    seed : int                   RNG seed.
    """

    metadata = dict(render_modes=["ansi"])

    def __init__(self, n, disturbances=None, max_steps=None, sparse=False, seed=None):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.sparse = sparse
        self.disturbances = list(disturbances) if disturbances else []
        self.max_steps = max_steps if max_steps is not None else 4 * n
        self._seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._rng = random.Random(seed)

        self.action_names = ACTION_NAMES
        self.action_space = spaces.Discrete(4)

        G = 2 * n + 1
        self.G = G
        self._center = (n, n)
        self.observation_space = spaces.Dict({
            "agent_pos":       spaces.Box(0, G - 1, shape=(2,),    dtype=np.int32),
            "goal_pos":        spaces.Box(0, G - 1, shape=(2,),    dtype=np.int32),
            "steps_remaining": spaces.Box(0, n,     shape=(1,),    dtype=np.int32),
            "grid":            spaces.Box(0, 3,     shape=(G, G),  dtype=np.uint8),
        })

        self._agent = self._center
        self._goal = self._center
        self._remaining = 0
        self._step_count = 0
        self._last_info: dict = {}

    def reset(self, seed=None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
            self._rng = random.Random(seed)
        for d in self.disturbances:
            d.reset(self._np_rng)
        self._agent = self._center
        self._goal = _sample_goal(self.n, self._center, self._np_rng)
        self._remaining = _manhattan(self._agent, self._goal)
        self._step_count = 0
        self._last_info = dict(
            start=self._agent,
            goal=self._goal,
            path_length=self._remaining,
            disturbances=[repr(d) for d in self.disturbances],
        )
        return _make_obs(self.G, self._agent, self._goal, self._remaining)

    def step(self, action):
        if isinstance(action, str):
            action = _ACTION_TO_INT[action.lower()]
        disturbed = int(action)
        for d in self.disturbances:
            disturbed = d(disturbed)
        dr, dc = _DELTA[disturbed]
        nr, nc = self._agent[0] + dr, self._agent[1] + dc
        if 0 <= nr < self.G and 0 <= nc < self.G:
            self._agent = (nr, nc)
        dist = _manhattan(self._agent, self._goal)
        self._remaining = dist
        self._step_count += 1
        terminated = (self._agent == self._goal)
        truncated = not terminated and self._step_count >= self.max_steps
        done = terminated or truncated
        obs = _make_obs(self.G, self._agent, self._goal, self._remaining)
        info = dict(
            agent=self._agent,
            goal=self._goal,
            disturbed_action=disturbed,
            step=self._step_count,
            terminated=terminated,
            truncated=truncated,
        )
        if self.sparse:
            reward = 1.0 if terminated else 0.0
        else:
            reward = 1.0 - self._remaining / self.n
        return obs, reward, done, info

    def render(self):
        return _render_text(self.G, self._agent, self._goal, self._remaining)

    def close(self):
        pass
