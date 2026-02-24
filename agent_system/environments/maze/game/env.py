import copy
import random
from typing import List, Dict, Any, Optional, Tuple

from .maze_generator import generate_maze, load_fixed_maze, find_position, grid_to_string
from .fixed_mazes import FIXED_MAZES

# Direction mappings: (row_delta, col_delta)
DIRECTION_MAP = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}


class MazeEnv:
    """
    Maze environment with stochastic action execution.

    State: grid, agent position, goal position, step count.
    Actions: multi-step plans (list of direction strings).
    Stochastic effects: sticky, double, noise, unavailable actions.
    """

    def __init__(
        self,
        maze_size: int = 9,
        max_episode_steps: int = 50,
        plan_length: int = 5,
        sticky_prob: float = 0.0,
        double_prob: float = 0.0,
        noise_prob: float = 0.0,
        unavail_actions: Optional[List[str]] = None,
        fixed_maze_name: Optional[str] = None,
    ):
        self.maze_size = maze_size
        self.max_episode_steps = max_episode_steps
        self.plan_length = plan_length
        self.sticky_prob = sticky_prob
        self.double_prob = double_prob
        self.noise_prob = noise_prob
        self.unavail_actions = unavail_actions or []
        self.fixed_maze_name = fixed_maze_name

        # State
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.total_steps = 0
        self.done = False
        self.won = False
        self.prev_action = None
        self.seed = None

    def reset(self, seed: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment: generate or load maze, place agent at start."""
        self.seed = seed
        self.total_steps = 0
        self.done = False
        self.won = False
        self.prev_action = None

        if self.fixed_maze_name:
            maze_data = None
            for m in FIXED_MAZES:
                if m['name'] == self.fixed_maze_name:
                    maze_data = m
                    break
            if maze_data is None:
                raise ValueError(f"Fixed maze '{self.fixed_maze_name}' not found")
            self.grid = load_fixed_maze(maze_data['maze'])
        else:
            self.grid = generate_maze(
                width=self.maze_size,
                height=self.maze_size,
                seed=seed
            )

        self.agent_pos = find_position(self.grid, 'S')
        self.goal_pos = find_position(self.grid, 'G')

        if self.agent_pos is None or self.goal_pos is None:
            raise ValueError("Maze must have both 'S' (start) and 'G' (goal)")

        # Mark start as path (agent is tracked separately)
        self.grid[self.agent_pos[0]][self.agent_pos[1]] = '.'

        obs = self.render()
        info = {'won': False}
        return obs, info

    def step(self, actions: List[str]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute a multi-step plan.

        Args:
            actions: List of direction strings (e.g. ["up", "right", "down"])

        Returns:
            (observation, total_reward, done, info)
            info contains: won, planned_actions, executed_actions, wall_hits
        """
        if self.done:
            return self.render(), 0.0, True, {'won': self.won}

        total_reward = 0.0
        executed_actions = []
        wall_hits = 0
        rng = random.Random(self.seed * 10000 + self.total_steps)

        for i, action in enumerate(actions):
            if self.done:
                executed_actions.append('(skipped)')
                continue

            actual_action = self._apply_stochastic_effects(action, rng)
            executed_actions.append(actual_action)

            if actual_action == '(unavailable)':
                total_reward -= 0.5
                wall_hits += 1
                self.total_steps += 1
            else:
                moved, double_moved = self._execute_move(actual_action, rng)
                self.total_steps += 1

                if not moved:
                    total_reward -= 0.5
                    wall_hits += 1
                else:
                    total_reward -= 0.1

                    if double_moved:
                        executed_actions[-1] = actual_action.upper() + '(doubled)'

                    if self.agent_pos == self.goal_pos:
                        total_reward += 10.0
                        self.done = True
                        self.won = True

            if self.total_steps >= self.max_episode_steps:
                self.done = True

        obs = self.render()
        info = {
            'won': self.won,
            'planned_actions': actions,
            'executed_actions': executed_actions,
            'wall_hits': wall_hits,
            'total_steps': self.total_steps,
        }
        return obs, total_reward, self.done, info

    def _apply_stochastic_effects(self, action: str, rng: random.Random) -> str:
        """Apply stochastic effects to an action."""
        action = action.lower().strip()

        # Check unavailable actions
        if action in [a.lower() for a in self.unavail_actions]:
            return '(unavailable)'

        # Sticky: repeat previous action
        if self.prev_action and rng.random() < self.sticky_prob:
            action = self.prev_action

        # Noise: replace with random direction
        if rng.random() < self.noise_prob:
            action = rng.choice(list(DIRECTION_MAP.keys()))

        self.prev_action = action
        return action

    def _execute_move(self, action: str, rng: random.Random) -> Tuple[bool, bool]:
        """
        Execute a single move. Returns (moved, double_moved).
        """
        if action not in DIRECTION_MAP:
            return False, False

        dr, dc = DIRECTION_MAP[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds and walls
        if not self._is_valid_position(new_r, new_c):
            return False, False

        self.agent_pos = (new_r, new_c)

        # Check double move
        double_moved = False
        if rng.random() < self.double_prob and not self.agent_pos == self.goal_pos:
            new_r2 = new_r + dr
            new_c2 = new_c + dc
            if self._is_valid_position(new_r2, new_c2):
                self.agent_pos = (new_r2, new_c2)
                double_moved = True

        return True, double_moved

    def _is_valid_position(self, r: int, c: int) -> bool:
        """Check if a position is within bounds and not a wall."""
        if r < 0 or r >= len(self.grid) or c < 0 or c >= len(self.grid[0]):
            return False
        return self.grid[r][c] != '#'

    def render(self) -> str:
        """Return ASCII string of the grid with agent position marked as 'A'."""
        lines = []
        for r, row in enumerate(self.grid):
            line = []
            for c, cell in enumerate(row):
                if (r, c) == self.agent_pos:
                    line.append('A')
                elif (r, c) == self.goal_pos:
                    line.append('G')
                else:
                    line.append(cell)
            lines.append(''.join(line))
        return '\n'.join(lines)

    def copy(self) -> 'MazeEnv':
        """Create a deep copy of this environment."""
        new_env = MazeEnv(
            maze_size=self.maze_size,
            max_episode_steps=self.max_episode_steps,
            plan_length=self.plan_length,
            sticky_prob=self.sticky_prob,
            double_prob=self.double_prob,
            noise_prob=self.noise_prob,
            unavail_actions=list(self.unavail_actions),
            fixed_maze_name=self.fixed_maze_name,
        )
        new_env.grid = [row[:] for row in self.grid] if self.grid else None
        new_env.agent_pos = self.agent_pos
        new_env.goal_pos = self.goal_pos
        new_env.total_steps = self.total_steps
        new_env.done = self.done
        new_env.won = self.won
        new_env.prev_action = self.prev_action
        new_env.seed = self.seed
        return new_env
