from agent_system.environments.navigation.prompt import (
    parse_reflection,
    parse_current_trajectory,
    NAVIGATION_REFLECT_PROMPT,
)

NAVIGATION_SINGLE_STEP_PLAY_PROMPT = """
You are an expert agent navigating a 2D grid.

# Grid
- The grid is {grid_size} x {grid_size}.
- `A`: Your current position (agent), always starting at the center of the grid.
- `G`: The goal you must reach.
- `.`: An open cell.

# Goal
Navigate from your position (A) to the goal (G).
Without any interference, the shortest path to the goal is exactly {n} steps.

# Actions
Output exactly 1 character for your next move:
- `U`: Move up (decrease row)
- `D`: Move down (increase row)
- `L`: Move left (decrease column)
- `R`: Move right (increase column)

# Important: Adversarial Environment
This environment may be adversarial. The actual effect of your actions might
not match the labels above — for example, pressing `U` might move you in a
different direction. This remapping, if present, is deterministic: it stays
the same for the entire environment instance. By observing the outcomes of
your actions across multiple attempts, you can deduce the true mapping and
adapt your strategy accordingly.

# Observation
The initial grid state is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now choose your next single action.
- First, reason step-by-step: locate A and G on the current grid and plan a route.
- If you have prior steps or attempts, reason about how your actions actually moved
  the agent versus how you expected, to infer whether actions are remapped.
- Then output exactly 1 character within <action> </action> tags.
  Example: <action>L</action>
"""


def get_navigation_single_step_prompt(
    n: int,
    grid_size: int,
    phase: str = 'play',
    turn_idx: int = 0,
    traj_idx: int = 0,
    init_observation: str = '',
    curr_traj: str = '',
    past_traj: dict = None,
    reflection: dict = None,
    reflection_type: str = 'reflection_only',
) -> str:
    assert phase in ['play', 'reflect']
    past_traj = past_traj or {}
    reflection = reflection or {}

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = NAVIGATION_SINGLE_STEP_PLAY_PROMPT.format(
            n=n,
            grid_size=grid_size,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = NAVIGATION_REFLECT_PROMPT.format(
            n_remaining=n,
            grid_size=grid_size,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
