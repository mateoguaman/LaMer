import random
import copy
from typing import List, Tuple, Optional


def generate_maze(width: int = 9, height: int = 9, seed: Optional[int] = None) -> List[List[str]]:
    """
    Generate a random maze using randomized DFS (recursive backtracker).

    The maze uses odd-sized grids where:
    - Cells are at odd coordinates (1,1), (1,3), (3,1), etc.
    - Walls are at even coordinates

    Args:
        width: Width of the maze (will be forced to odd)
        height: Height of the maze (will be forced to odd)
        seed: Random seed for reproducibility

    Returns:
        2D grid where '#' = wall, '.' = path, 'S' = start, 'G' = goal
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Ensure odd dimensions for proper maze structure
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    # Initialize grid with all walls
    grid = [['#' for _ in range(width)] for _ in range(height)]

    # Cell coordinates are at odd positions
    cell_rows = height // 2
    cell_cols = width // 2

    visited = [[False] * cell_cols for _ in range(cell_rows)]

    def cell_to_grid(cr, cc):
        return (cr * 2 + 1, cc * 2 + 1)

    def get_neighbors(cr, cc):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not visited[nr][nc]:
                neighbors.append((nr, nc))
        return neighbors

    # Randomized DFS
    start_cr, start_cc = 0, 0
    visited[start_cr][start_cc] = True
    gr, gc = cell_to_grid(start_cr, start_cc)
    grid[gr][gc] = '.'

    stack = [(start_cr, start_cc)]
    while stack:
        cr, cc = stack[-1]
        neighbors = get_neighbors(cr, cc)
        if neighbors:
            nr, nc = rng.choice(neighbors)
            visited[nr][nc] = True
            # Remove wall between current and neighbor
            wall_r = cr * 2 + 1 + (nr - cr)
            wall_c = cc * 2 + 1 + (nc - cc)
            grid[wall_r][wall_c] = '.'
            ngr, ngc = cell_to_grid(nr, nc)
            grid[ngr][ngc] = '.'
            stack.append((nr, nc))
        else:
            stack.pop()

    # Place start and goal
    grid[1][1] = 'S'
    goal_r, goal_c = cell_to_grid(cell_rows - 1, cell_cols - 1)
    grid[goal_r][goal_c] = 'G'

    return grid


def load_fixed_maze(maze_str: str) -> List[List[str]]:
    """
    Load a maze from a multi-line string.

    Args:
        maze_str: String representation of the maze

    Returns:
        2D grid
    """
    lines = maze_str.strip().split('\n')
    grid = [list(line) for line in lines]
    return grid


def find_position(grid: List[List[str]], char: str) -> Optional[Tuple[int, int]]:
    """Find the (row, col) position of a character in the grid."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == char:
                return (r, c)
    return None


def grid_to_string(grid: List[List[str]]) -> str:
    """Convert a 2D grid to a string representation."""
    return '\n'.join(''.join(row) for row in grid)
