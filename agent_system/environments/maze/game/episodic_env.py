"""
Episodic wrapper around MazeEnv.

Instead of stepping one action at a time, ``EpisodicMazeEnv.step`` accepts a
string of ``remaining_actions`` characters — one per move — executes the full
sequence inside the underlying environment, and returns only the final
observation and done signal.

Action character mapping
------------------------
    'U' → up
    'D' → down
    'L' → left
    'R' → right

Example
-------
    env = EpisodicMazeEnv(n=4)
    obs, info = env.reset(seed=42)

    # Execute 4 moves in one call; action string length == remaining_actions
    obs, reward, done, info = env.step("LRUD")
"""

from __future__ import annotations

import copy

from .env import MazeEnv

_CHAR_TO_ACTION: dict[str, str] = {
    "U": "up",
    "D": "down",
    "L": "left",
    "R": "right",
}


class EpisodicMazeEnv:
    """
    Episodic wrapper around MazeEnv.

    ``step`` consumes a string of action characters (U/D/L/R), executes them
    one by one inside the underlying ``MazeEnv``, and returns the state after
    the last action (or after the episode terminates / truncates early).

    Parameters
    ----------
    n : int
        Shortest-path length for the maze (even, >= 2).
    disturbances : list, optional
        ``ActionDisturbance`` instances forwarded to ``MazeEnv``.
    max_steps : int, optional
        Maximum number of individual cell-steps per episode (default 4*n).
    sparse : bool
        If True, use sparse {0, 1} reward (default False).
    """

    def __init__(
        self,
        n: int,
        disturbances=None,
        max_steps: int | None = None,
        sparse: bool = False,
    ) -> None:
        self._env = MazeEnv(
            n=n,
            disturbances=disturbances,
            max_steps=max_steps,
            sparse=sparse,
        )
        self.n = n

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """
        Reset the environment and return an initial text observation.

        Parameters
        ----------
        seed : int, optional

        Returns
        -------
        obs : str   ASCII text representation of the initial maze state.
        info : dict Episode metadata (start, goal, path_length, …).
        """
        self._env.reset(seed=seed)
        obs = self._env.render()
        return obs, self._env._last_info

    def step(self, action_str: str) -> tuple[str, float, bool, dict]:
        """
        Execute a sequence of actions encoded as a single string.

        Each character in ``action_str`` is one move:
            'U' → up  |  'D' → down  |  'L' → left  |  'R' → right

        Execution stops early if the episode terminates or is truncated.

        Parameters
        ----------
        action_str : str
            Sequence of action characters, e.g. ``"LRUD"`` for a 4-step
            episode (``remaining_actions == 4``).

        Returns
        -------
        obs : str
            ASCII text representation of the final maze state.
        reward : float
            Cumulative reward over all executed steps.
        done : bool
            True if the episode terminated (goal reached) or was truncated
            (step limit hit) before all actions were consumed.
        info : dict
            Info dict from the last executed step.
        """
        if self._env._maze is None:
            raise RuntimeError("Call reset() before step().")

        # total_reward = 0.0
        reward = 0.0
        terminated = False
        done = False
        info: dict = {}

        for char in action_str.upper():
            if char not in _CHAR_TO_ACTION:
                # skip invalid characters silently so empty/malformed strings
                # don't crash — the env manager already tracks validity
                continue
            _, reward, done, info = self._env.step(_CHAR_TO_ACTION[char])
            # total_reward += reward
            terminated = info.get("terminated", False)
            if done:
                break

        info["won"] = terminated
        obs = self._env.render()
        # return the reward of the last step
        return obs, reward, done, info
        # return obs, total_reward, done, info

    def copy(self) -> "EpisodicMazeEnv":
        """Return a deep copy of this environment (state included)."""
        new = EpisodicMazeEnv.__new__(EpisodicMazeEnv)
        new.n = self.n
        new._env = copy.deepcopy(self._env)
        return new

    # ---------------------------------------------------------------------- #
    # Convenience pass-throughs                                                #
    # ---------------------------------------------------------------------- #

    @property
    def action_names(self):
        return self._env.action_names

    def render(self) -> str | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
