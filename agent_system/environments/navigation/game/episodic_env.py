"""
Episodic wrapper around NavigationEnv.

Instead of stepping one action at a time, ``EpisodicNavigationEnv.step``
accepts a string of action characters — one per move — executes the full
sequence inside the underlying environment, and returns only the final
observation and done signal.

Action character mapping
------------------------
    'U' -> up
    'D' -> down
    'L' -> left
    'R' -> right

Example
-------
    env = EpisodicNavigationEnv(n=4)
    obs, info = env.reset(seed=42)

    # Execute up to 4 moves in one call
    obs, reward, done, info = env.step("LLUR")
"""

from __future__ import annotations

import copy

from .env import NavigationEnv

_CHAR_TO_ACTION: dict[str, str] = {
    "U": "up",
    "D": "down",
    "L": "left",
    "R": "right",
}


class EpisodicNavigationEnv:
    """
    Episodic wrapper around NavigationEnv.

    ``step`` consumes a string of action characters (U/D/L/R), executes them
    one by one inside the underlying ``NavigationEnv``, and returns the state
    after the last action (or after the episode terminates / truncates early).

    Parameters
    ----------
    n : int
        Shortest-path length (>= 1).
    disturbances : list, optional
        ``ActionDisturbance`` instances forwarded to ``NavigationEnv``.
    max_steps : int, optional
        Maximum number of individual steps per episode (default 4*n).
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
        self._env = NavigationEnv(
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

        Returns
        -------
        obs : str   ASCII text representation of the initial grid state.
        info : dict Episode metadata (start, goal, path_length, …).
        """
        self._env.reset(seed=seed)
        obs = self._env.render()
        return obs, self._env._last_info

    def step(self, action_str: str) -> tuple[str, float, bool, dict]:
        """
        Execute a sequence of actions encoded as a single string.

        Each character in ``action_str`` is one move:
            'U' -> up  |  'D' -> down  |  'L' -> left  |  'R' -> right

        Execution stops early if the episode terminates or is truncated.

        Returns
        -------
        obs : str
            ASCII text representation of the final grid state.
        reward : float
            Reward of the last executed step.
        done : bool
            True if the episode terminated (goal reached) or was truncated.
        info : dict
            Info dict from the last executed step.
        """
        if self._env._last_info == {}:
            raise RuntimeError("Call reset() before step().")

        reward = 0.0
        terminated = False
        done = False
        info: dict = {}

        for char in action_str.upper():
            if char not in _CHAR_TO_ACTION:
                continue
            _, reward, done, info = self._env.step(_CHAR_TO_ACTION[char])
            terminated = info.get("terminated", False)
            if done:
                break

        info["won"] = terminated
        obs = self._env.render()
        return obs, reward, done, info

    def copy(self) -> "EpisodicNavigationEnv":
        """Return a deep copy of this environment (state included)."""
        new = EpisodicNavigationEnv.__new__(EpisodicNavigationEnv)
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
