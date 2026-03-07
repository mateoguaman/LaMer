from __future__ import annotations
import random
from collections import deque
import numpy as np
import gym
from gym import spaces
from .disturbances import ActionDisturbance
from .generator import FREE, generate_maze, grid_size_for_n

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Ordered to match the integer action encoding above.
ACTION_NAMES = ("up", "down", "left", "right")
_ACTION_TO_INT = {name: i for i, name in enumerate(ACTION_NAMES)}

_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Maze cell encoding used in the "maze" observation array.
_CELL_WALL  = np.uint8(0)   # '#'
_CELL_FREE  = np.uint8(1)   # '.'
_CELL_AGENT = np.uint8(2)   # agent position
_CELL_GOAL  = np.uint8(3)   # goal position


def _bfs_grid_distance(grid, start, goal):
    """BFS counting individual cell steps through FREE cells."""
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in _DELTA.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited and grid[nr][nc] == FREE):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def _render_text(grid, agent, goal, remaining):
    """ASCII string used by render()."""
    lines = []
    for r, row in enumerate(grid):
        chars = []
        for c, cell in enumerate(row):
            if (r, c) == agent:
                chars.append("A")
            elif (r, c) == goal:
                chars.append("G")
            else:
                chars.append(cell)
        lines.append("".join(chars))
    hdr = "Agent: ({},{})  Goal: ({},{})  Steps remaining: {}".format(
        agent[0], agent[1], goal[0], goal[1], remaining)
    return hdr + "\n" + "\n".join(lines)


def _make_obs(grid, agent, goal, remaining):
    """Build the dict observation returned by reset() and step()."""
    rows, cols = len(grid), len(grid[0])
    maze_arr = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            maze_arr[r, c] = _CELL_FREE if grid[r][c] == FREE else _CELL_WALL
    maze_arr[agent[0], agent[1]] = _CELL_AGENT
    maze_arr[goal[0], goal[1]] = _CELL_GOAL
    return {
        "agent_pos":       np.array(agent,       dtype=np.int32),
        "goal_pos":        np.array(goal,         dtype=np.int32),
        "steps_remaining": np.array([remaining], dtype=np.int32),
        "maze":            maze_arr,
    }


class MazeEnv(gym.Env):
    """
    Passage-based maze env with exact shortest-path length n.

    A "step" moves the agent one cell in the display grid.  n must be an even
    integer >= 2 (room corners always sit at odd grid positions, so the
    shortest path between any two corners has even length).

    Observation space: Dict
        "agent_pos"      : Box(2,)    int32  — (row, col) in display grid
        "goal_pos"       : Box(2,)    int32  — (row, col) in display grid
        "steps_remaining": Box(1,)    int32  — BFS distance remaining to goal
        "maze"           : Box(G, G)  uint8  — grid encoding
                           0=wall  1=free  2=agent  3=goal
                           G = grid_size_for_n(n)

    Action space: Discrete(4) — step() accepts integers OR strings.
        0="up"  1="down"  2="left"  3="right"
        action_space.sample() returns an integer;
        map with ACTION_NAMES[i] to get the string name.

    Reward:
        sparse=False (default): 1 - dist/n  ∈ [0, 1], where dist is the
            BFS distance to the goal.  Equals 0 at the start, 1 at the goal.
        sparse=True: 0 everywhere, +1 only when the agent reaches the goal.
    Terminates when agent reaches goal cell.

    Parameters
    ----------
    n : int                      Shortest-path length in cell steps (even, >= 2).
    disturbances : list          ActionDisturbance instances applied in order.
    max_steps : int              Episode length limit (default 4*n).
    sparse : bool                If True, use sparse {0, 1} reward (default False).
    seed : int                   RNG seed.
    """

    metadata = dict(render_modes=["ansi"])

    def __init__(self, n, disturbances=None, max_steps=None, sparse=False, seed=None):
        super().__init__()
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 != 0:
            raise ValueError(
                f"n must be even (got {n}). Cell-step paths between corners "
                "always have even length in a passage-based maze."
            )
        self.n = n
        self.sparse = sparse
        self.disturbances = list(disturbances) if disturbances else []
        self.max_steps = max_steps if max_steps is not None else 4 * n
        self._seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._rng = random.Random(seed)

        self.action_names = ACTION_NAMES
        self.action_space = spaces.Discrete(4)

        G = grid_size_for_n(n)
        self.observation_space = spaces.Dict({
            "agent_pos":       spaces.Box(0, G - 1, shape=(2,),    dtype=np.int32),
            "goal_pos":        spaces.Box(0, G - 1, shape=(2,),    dtype=np.int32),
            "steps_remaining": spaces.Box(0, n,     shape=(1,),    dtype=np.int32),
            "maze":            spaces.Box(0, 3,     shape=(G, G),  dtype=np.uint8),
        })

        self._maze = None
        self._agent = (0, 0)
        self._remaining = 0
        self._step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
            self._rng = random.Random(seed)
        for d in self.disturbances:
            d.reset(self._np_rng)
        self._maze = generate_maze(self.n, self._rng)
        self._agent = self._maze.start
        self._remaining = self._maze.path_length
        self._step_count = 0
        obs = _make_obs(self._maze.grid, self._agent, self._maze.goal, self._remaining)
        self._last_info = dict(
            start=self._maze.start,
            goal=self._maze.goal,
            path_length=self._maze.path_length,
            disturbances=[repr(d) for d in self.disturbances],
        )
        return obs

    def step(self, action):
        if self._maze is None:
            raise RuntimeError("Call reset() first.")
        if isinstance(action, str):
            action = _ACTION_TO_INT[action.lower()]
        disturbed = int(action)
        for d in self.disturbances:
            disturbed = d(disturbed)
        dr, dc = _DELTA[disturbed]
        nr, nc = self._agent[0] + dr, self._agent[1] + dc
        grid = self._maze.grid
        rows, cols = len(grid), len(grid[0])
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == FREE:
            self._agent = (nr, nc)
        dist = _bfs_grid_distance(grid, self._agent, self._maze.goal)
        self._remaining = dist if dist is not None else self._remaining
        self._step_count += 1
        terminated = (self._agent == self._maze.goal)
        truncated = not terminated and self._step_count >= self.max_steps
        done = terminated or truncated
        obs = _make_obs(grid, self._agent, self._maze.goal, self._remaining)
        info = dict(agent=self._agent, goal=self._maze.goal,
                    disturbed_action=disturbed, step=self._step_count,
                    terminated=terminated, truncated=truncated)
        if self.sparse:
            reward = 1.0 if terminated else 0.0
        else:
            reward = 1.0 - self._remaining / self.n
        return obs, reward, done, info

    def render(self):
        if self._maze is None:
            return None
        return _render_text(self._maze.grid, self._agent, self._maze.goal, self._remaining)

    def close(self):
        pass
