"""
Modular action-disturbance system for NavigationEnv.

Usage
-----
Subclass ``ActionDisturbance`` and implement ``__call__``.  Optionally
override ``reset(rng)`` to draw fresh randomness at the start of each episode.

    from navigation.disturbances import ActionDisturbance

    class MyDisturbance(ActionDisturbance):
        def __call__(self, action: int) -> int:
            return (action + 1) % 4

Pass a list of disturbances to ``NavigationEnv``:

    env = NavigationEnv(n=5, disturbances=[FlipLeftRight(), RotateActions()])

Disturbances are applied in list order on every ``step`` call, and each
disturbance's ``reset`` is called at the start of every episode.

Action encoding (matches NavigationEnv)
---------------------------------------
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

_N_ACTIONS = 4


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #


class ActionDisturbance(ABC):
    """
    Abstract base for action-space disturbances.

    Subclasses **must** implement ``__call__``.  They **may** override
    ``reset`` to re-draw randomness at the start of each episode.
    """

    @abstractmethod
    def __call__(self, action: int) -> int:
        """Map a raw action index to a (possibly disturbed) action index."""

    def reset(self, rng: np.random.Generator) -> None:
        """
        Called by ``NavigationEnv.reset()`` before each new episode.

        Override this to re-sample any stochastic state (e.g. random
        permutations).  The default implementation does nothing.

        Parameters
        ----------
        rng:
            The environment's shared ``numpy.random.Generator`` so that the
            disturbance is reproducible when the env is seeded.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# --------------------------------------------------------------------------- #
# Built-in disturbances                                                        #
# --------------------------------------------------------------------------- #


class NoDisturbance(ActionDisturbance):
    """Identity — passes actions through unchanged."""

    def __call__(self, action: int) -> int:
        return action


class FlipLeftRight(ActionDisturbance):
    """Swap LEFT <-> RIGHT; UP and DOWN are unaffected."""

    _TABLE = {UP: UP, DOWN: DOWN, LEFT: RIGHT, RIGHT: LEFT}

    def __call__(self, action: int) -> int:
        return self._TABLE[action]


class FlipUpDown(ActionDisturbance):
    """Swap UP <-> DOWN; LEFT and RIGHT are unaffected."""

    _TABLE = {UP: DOWN, DOWN: UP, LEFT: LEFT, RIGHT: RIGHT}

    def __call__(self, action: int) -> int:
        return self._TABLE[action]


class FlipBoth(ActionDisturbance):
    """Swap LEFT <-> RIGHT *and* UP <-> DOWN simultaneously."""

    _TABLE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    def __call__(self, action: int) -> int:
        return self._TABLE[action]


class FlipDiagonal(ActionDisturbance):
    """
    Reflect actions across the main diagonal: swap UP <-> RIGHT and DOWN <-> LEFT.

    Geometrically this mirrors movement across the top-right / bottom-left axis,
    so moving right becomes moving up, moving up becomes moving right, etc.
    """

    _TABLE = {UP: RIGHT, RIGHT: UP, DOWN: LEFT, LEFT: DOWN}

    def __call__(self, action: int) -> int:
        return self._TABLE[action]


class FlipAntiDiagonal(ActionDisturbance):
    """
    Reflect actions across the anti-diagonal: swap UP <-> LEFT and DOWN <-> RIGHT.

    Geometrically this mirrors movement across the top-left / bottom-right axis,
    so moving left becomes moving up, moving up becomes moving left, etc.
    """

    _TABLE = {UP: LEFT, LEFT: UP, DOWN: RIGHT, RIGHT: DOWN}

    def __call__(self, action: int) -> int:
        return self._TABLE[action]


class CyclicRotation(ActionDisturbance):
    """
    Cyclically shift all actions by a fixed offset of 1 (UP->DOWN->LEFT->RIGHT->UP).

    Unlike ``RotateActions``, the offset is always exactly 1 and never changes
    across resets, making this a deterministic disturbance.

    Action mapping: UP->DOWN, DOWN->LEFT, LEFT->RIGHT, RIGHT->UP
    """

    def __call__(self, action: int) -> int:
        return (action + 1) % _N_ACTIONS


class RotateActions(ActionDisturbance):
    """
    Cyclically shift all actions by a random offset drawn at each reset.

    With the default action ordering [UP=0, DOWN=1, LEFT=2, RIGHT=3], a
    rotation of k maps action a -> (a + k) % 4.  The offset k is drawn
    uniformly from {1, 2, 3} so the identity is excluded.
    """

    def __init__(self) -> None:
        self._k: int = 1

    def reset(self, rng: np.random.Generator) -> None:
        self._k = int(rng.integers(1, _N_ACTIONS))  # k in {1, 2, 3}

    def __call__(self, action: int) -> int:
        return (action + self._k) % _N_ACTIONS

    def __repr__(self) -> str:
        return f"RotateActions(k={self._k})"


class RandomPermutation(ActionDisturbance):
    """
    Apply a uniformly random permutation of all four actions, re-drawn at
    each reset.

    The permutation is guaranteed to differ from the identity (i.e. at least
    one action is remapped).
    """

    def __init__(self) -> None:
        self._perm: list[int] = list(range(_N_ACTIONS))

    def reset(self, rng: np.random.Generator) -> None:
        perm = list(range(_N_ACTIONS))
        while perm == list(range(_N_ACTIONS)):
            rng.shuffle(perm)  # type: ignore[arg-type]
        self._perm = perm

    def __call__(self, action: int) -> int:
        return self._perm[action]

    def __repr__(self) -> str:
        names = ["UP", "DOWN", "LEFT", "RIGHT"]
        mapping = ", ".join(f"{names[i]}->{names[self._perm[i]]}" for i in range(_N_ACTIONS))
        return f"RandomPermutation({mapping})"


class RandomChoice(ActionDisturbance):
    """
    Meta-level disturbance selector: on each ``reset``, uniformly draw one
    disturbance from the pool and apply it for the entire episode.

    Because ``NavigationWorker.restart()`` restores the environment from a
    saved copy rather than calling ``reset()`` again, the chosen disturbance
    is automatically preserved across all attempts within a meta-episode.

    Include ``NoDisturbance`` in the pool to give the agent a chance of
    encountering an undisturbed grid.

    Parameters
    ----------
    disturbances : list[ActionDisturbance]
        Pool of disturbances to sample from (must be non-empty).
    """

    def __init__(self, disturbances: list) -> None:
        if not disturbances:
            raise ValueError("disturbances list must be non-empty")
        self._pool = list(disturbances)
        self._active: ActionDisturbance = self._pool[0]

    def reset(self, rng: np.random.Generator) -> None:
        idx = int(rng.integers(0, len(self._pool)))
        self._active = self._pool[idx]
        self._active.reset(rng)

    def __call__(self, action: int) -> int:
        return self._active(action)

    def __repr__(self) -> str:
        return f"RandomChoice(active={self._active!r})"


class Probabilistic(ActionDisturbance):
    """
    Domain-randomization wrapper: activates ``disturbance`` with probability
    ``p`` at the start of each episode, otherwise acts as identity.

    Both the activation coin-flip and the inner disturbance's own reset are
    tied to the environment's shared RNG, so everything is reproducible when
    the env is seeded.

    Parameters
    ----------
    disturbance : ActionDisturbance
        Any disturbance to wrap.
    p : float
        Probability of activating the disturbance each episode (default 0.5).
    """

    def __init__(self, disturbance: ActionDisturbance, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self._inner = disturbance
        self.p = p
        self._active = False

    def reset(self, rng: np.random.Generator) -> None:
        self._active = bool(rng.random() < self.p)
        if self._active:
            self._inner.reset(rng)

    def __call__(self, action: int) -> int:
        return self._inner(action) if self._active else action

    def __repr__(self) -> str:
        return (
            f"Probabilistic({self._inner!r}, p={self.p}, active={self._active})"
        )
