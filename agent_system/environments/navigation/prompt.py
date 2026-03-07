NAVIGATION_PLAY_PROMPT = """
You are an expert agent navigating a 2D grid.

# Grid
- The grid is {grid_size} x {grid_size}.
- `A`: Your current position (agent), always starting at the center of the grid.
- `G`: The goal you must reach.
- `.`: An open cell.

# Goal
Navigate from your position (A) to the goal (G).
Without any interference, the shortest path to the goal is exactly {n_remaining} steps.

# Actions
You must output a string of exactly {n_remaining} characters, each one of:
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
Now it's your turn to plan your path.
- First, reason step-by-step: locate A and G on the grid and plan a route.
- If you have prior attempts, reason about how your actions actually moved the
  agent versus how you expected, to infer whether actions are remapped.
- Then output exactly {n_remaining} characters within <action> </action> tags.
  Example for n_remaining=4: <action>LLUR</action>
"""

NAVIGATION_REFLECT_PROMPT = '''
You are an expert agent navigating a 2D grid.

# Grid
- The grid is {grid_size} x {grid_size}.
- `A`: Your current position (agent), always starting at the center.
- `G`: The goal you must reach.
- `.`: An open cell.

# Goal
Navigate from A to G in exactly {n_remaining} steps.

# Important: Adversarial Environment
This environment may be adversarial. The actual effect of your actions might
not match the labels — for example, pressing `U` might move you in a different
direction. This remapping is deterministic for the entire environment instance.

# Your Task
You will be given the history of a past failed attempt.
Reflect on what went wrong, whether the actions behaved as expected, and
devise an improved action sequence.

# Past Experience
The initial grid state is:
{init_observation}{current_trajectory}
The task was NOT successfully completed.

Now reflect on your past attempt:
- Reason step-by-step about which moves were wrong and why.
- Identify whether the action effects were as expected or appeared remapped.
- If you detected a remapping, describe it explicitly.
- Propose a corrected plan using your updated understanding of the action mapping.
- End with your reflection and improved plan inside <remark> </remark> tags.
'''

# --------------------------------------------------------------------------- #
# Templates for formatting past trajectories and reflections                  #
# --------------------------------------------------------------------------- #

PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''
On trial #{traj_idx}, you took the following actions:
{past_trajectory}
The task was NOT successfully completed. Your reflection was:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''
On trial #{traj_idx}, you took the following actions:
{past_trajectory}
The task was NOT successfully completed.'''

REFLECTION_ONLY_TEMPLATE = '''
On trial #{traj_idx}, the task was NOT successfully completed. Your reflection was:
{reflection}'''


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    memories = []
    for _idx in range(traj_idx):
        if reflection_type == 'history_and_reflection':
            memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                traj_idx=_idx + 1,
                past_trajectory=past_traj[_idx],
                reflection=reflection[_idx],
            )
        elif reflection_type == 'history_only':
            memory = HISTORY_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1,
                past_trajectory=past_traj[_idx],
            )
        elif reflection_type == 'reflection_only':
            memory = REFLECTION_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1,
                reflection=reflection[_idx],
            )
        else:
            raise ValueError(f"Unknown reflection_type: {reflection_type}")
        memories.append(memory)
    return ''.join(memories)


# --------------------------------------------------------------------------- #
# Templates for formatting the current trajectory                             #
# --------------------------------------------------------------------------- #

CURR_TRAJ_AT_TRAJ1 = '''
You have already taken the following actions:
{current_trajectory}
'''

CURR_TRAJ_AT_TRAJ2toN = '''
Currently you're on trial #{traj_idx}. You have already taken the following actions:
{current_trajectory}
'''

TRAJ_2toN_INIT = '''
Currently you're on trial #{traj_idx}, starting from the initial state.'''


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj,
            )


def get_navigation_prompt(
    n_remaining: int,
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
        prompt = NAVIGATION_PLAY_PROMPT.format(
            n_remaining=n_remaining,
            grid_size=grid_size,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = NAVIGATION_REFLECT_PROMPT.format(
            n_remaining=n_remaining,
            grid_size=grid_size,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
