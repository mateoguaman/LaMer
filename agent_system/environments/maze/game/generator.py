"""
Passage-based maze generation with an exact guaranteed shortest-path length.

Representation
--------------
A "room maze" of R×R rooms is encoded in a (2R+1)×(2R+1) grid:

  - Room cells sit at odd grid positions (2r+1, 2c+1) — always FREE.
  - Passage cells between adjacent rooms (r,c) and (r,c+1) sit at
    (2r+1, 2c+2) — FREE if the passage is open, WALL otherwise.
  - Passage cells between (r,c) and (r+1,c) sit at (2r+2, 2c+1).
  - All other cells (border + interior corners at even/even positions) are
    WALL.

This prevents BFS shortcuts by construction: two rooms are connected *only*
through an explicitly opened passage cell.

Steps
-----
A "step" is moving one cell in the display grid.  Because room corners sit at
odd grid positions, any path between two corners always has even length.
Therefore ``n`` (the desired shortest-path length) **must be even**.

Algorithm
---------
1. Compute n_rooms = n // 2 (room hops equivalent to n grid steps).
2. Pick the smallest R such that:
   - R² ≥ n_rooms+1      (path visits n_rooms+1 rooms; must fit)
   - Some corner pair has room-Manhattan distance d ≤ n_rooms with
     (n_rooms−d) even.
3. Draw a random valid corner pair as start/goal (room coordinates).
4. Carve a simple path of exactly n_rooms room hops via constrained DFS.
5. Run a recursive-backtracker DFS from every path room, only visiting
   rooms *not* on the solution path, to add dead-end branches.
6. Cell-by-cell BFS verifies grid-step distance == n.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

WALL = "#"
FREE = "."

# Room-space directions (row-delta, col-delta)
ROOM_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class MazeResult(NamedTuple):
    grid: list[list[str]]   # (2R+1)×(2R+1) display grid
    start: tuple[int, int]  # agent start in grid coordinates
    goal: tuple[int, int]   # goal in grid coordinates
    path_length: int        # guaranteed BFS grid-step path length (even)


# --------------------------------------------------------------------------- #
# Grid ↔ room coordinate helpers                                              #
# --------------------------------------------------------------------------- #

def room_to_grid(r: int, c: int) -> tuple[int, int]:
    """Room (r,c) → grid position of its cell."""
    return 2 * r + 1, 2 * c + 1


def _passage_cell(
    r1: int, c1: int, r2: int, c2: int
) -> tuple[int, int]:
    """Grid position of the passage between adjacent rooms (r1,c1) and (r2,c2)."""
    return r1 + r2 + 1, c1 + c2 + 1


# --------------------------------------------------------------------------- #
# Grid sizing                                                                  #
# --------------------------------------------------------------------------- #

def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _corner_pairs(
    R: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All distinct corner-room pairs for an R×R room grid."""
    corners = [(0, 0), (0, R - 1), (R - 1, 0), (R - 1, R - 1)]
    pairs = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            pairs.append((corners[i], corners[j]))
    return pairs


def _find_grid_params(
    n: int,
) -> tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Return ``(R, valid_pairs)`` — the smallest room-grid side R and the
    corner pairs whose Manhattan distance d satisfies d ≤ n and (n−d) even,
    with R² ≥ n+1.
    """
    for R in range(2, n + 3):
        if R * R < n + 1:
            continue
        valid = [
            (a, b)
            for a, b in _corner_pairs(R)
            if _manhattan(a, b) <= n and (n - _manhattan(a, b)) % 2 == 0
        ]
        if valid:
            return R, valid
    raise ValueError(f"Cannot determine grid parameters for n={n}")


# --------------------------------------------------------------------------- #
# BFS (cell-by-cell over the display grid)                                    #
# --------------------------------------------------------------------------- #

def _bfs_distance(
    grid: list[list[str]],
    start_grid: tuple[int, int],
    goal_grid: tuple[int, int],
) -> int | None:
    """
    BFS from ``start_grid`` to ``goal_grid``, moving one cell at a time.
    Only FREE cells are traversable.  Returns the number of single-cell
    steps, or None if the goal is unreachable.
    """
    rows = len(grid)
    cols = len(grid[0])
    visited: set[tuple[int, int]] = {start_grid}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start_grid, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal_grid:
            return dist
        for dr, dc in ROOM_DIRS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and (nr, nc) not in visited
                and grid[nr][nc] == FREE
            ):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


# --------------------------------------------------------------------------- #
# Exact-length path (DFS + backtracking in room space)                        #
# --------------------------------------------------------------------------- #

def _carve_exact_path(
    R: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    n: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    """
    DFS with backtracking to find a simple path of exactly n steps (room
    hops) from ``start`` to ``goal``.

    Prunes when:
    - Manhattan distance to goal > remaining budget, OR
    - (remaining budget − distance) is odd (parity mismatch).
    """
    path: list[tuple[int, int]] = [start]
    visited: set[tuple[int, int]] = {start}

    def dfs(pos: tuple[int, int], budget: int) -> bool:
        if budget == 0:
            return pos == goal
        dist = _manhattan(pos, goal)
        if dist > budget or (budget - dist) % 2 != 0:
            return False
        dirs = ROOM_DIRS[:]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < R and 0 <= nc < R and (nr, nc) not in visited:
                visited.add((nr, nc))
                path.append((nr, nc))
                if dfs((nr, nc), budget - 1):
                    return True
                path.pop()
                visited.remove((nr, nc))
        return False

    return path if dfs(start, n) else None


# --------------------------------------------------------------------------- #
# Branch growth (recursive backtracker in room space)                         #
# --------------------------------------------------------------------------- #

def _carve_branches(
    grid: list[list[str]],
    R: int,
    path_rooms: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    """
    Extend the maze with dead-end branches via a randomised iterative
    backtracker, starting from each path room.

    Only visits rooms NOT in ``path_rooms`` — this guarantees no new
    passage is created between two path rooms, preserving BFS distance = n.
    """
    visited = set(path_rooms)

    seeds = list(path_rooms)
    rng.shuffle(seeds)
    for seed in seeds:
        stack = [seed]
        while stack:
            r, c = stack[-1]
            dirs = ROOM_DIRS[:]
            rng.shuffle(dirs)
            moved = False
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < R
                    and 0 <= nc < R
                    and (nr, nc) not in visited
                ):
                    # Open the passage between (r,c) and (nr,nc)
                    pr, pc = _passage_cell(r, c, nr, nc)
                    grid[pr][pc] = FREE
                    visited.add((nr, nc))
                    stack.append((nr, nc))
                    moved = True
                    break
            if not moved:
                stack.pop()


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def grid_size_for_n(n: int) -> int:
    """Return the display-grid side length for a maze with shortest path ``n``.

    Useful for pre-computing observation space shapes before calling
    ``generate_maze``.  Raises ``ValueError`` for invalid ``n``.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 2")
    R, _ = _find_grid_params(n // 2)
    return 2 * R + 1


def generate_maze(n: int, rng: random.Random | None = None) -> MazeResult:
    """
    Generate a passage-based maze whose BFS shortest path from start to goal
    is exactly ``n`` single-cell steps.

    Parameters
    ----------
    n:
        Desired shortest-path length in single-cell steps.  Must be an even
        integer ≥ 2.  (Room corners sit at odd grid positions, so any path
        between two corners always has even length.)
    rng:
        Optional seeded ``random.Random`` instance for reproducibility.

    Returns
    -------
    MazeResult with ``grid``, ``start``/``goal`` in grid coords, and
    ``path_length == n``.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n % 2 != 0:
        raise ValueError(
            f"n must be even (got {n}). In a passage-based maze the shortest "
            "path between any two corner cells always has even length."
        )

    if rng is None:
        rng = random.Random()

    # Internally generate a maze with n//2 room hops; that equals n grid steps.
    n_rooms = n // 2
    R, valid_pairs = _find_grid_params(n_rooms)
    grid_size = 2 * R + 1

    max_attempts = 200
    for _ in range(max_attempts):
        # All-walls grid; room cells always FREE
        grid: list[list[str]] = [[WALL] * grid_size for _ in range(grid_size)]
        for r in range(R):
            for c in range(R):
                gr, gc = room_to_grid(r, c)
                grid[gr][gc] = FREE

        # Pick a random valid corner pair
        start_room, goal_room = rng.choice(valid_pairs)

        # Carve solution path of exactly n_rooms room hops
        path = _carve_exact_path(R, start_room, goal_room, n_rooms, rng)
        if path is None:
            continue

        # Open passages along the solution path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            pr, pc = _passage_cell(r1, c1, r2, c2)
            grid[pr][pc] = FREE

        # Grow dead-end branches from path rooms
        _carve_branches(grid, R, set(path), rng)

        # Verify cell-by-cell BFS distance == n
        sg = room_to_grid(*start_room)
        gg = room_to_grid(*goal_room)
        dist = _bfs_distance(grid, sg, gg)
        if dist == n:
            return MazeResult(grid=grid, start=sg, goal=gg, path_length=n)

    raise RuntimeError(
        f"Could not generate a maze with shortest path length {n} "
        f"after {max_attempts} attempts."
    )
