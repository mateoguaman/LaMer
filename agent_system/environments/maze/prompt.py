MAZE_PLAY_PROMPT = """
You are an expert agent navigating a maze.

# Symbols
- `A`: Your current position (agent)
- `G`: The goal you must reach
- `#`: Wall (impassable)
- `.`: Open passage (passable)

# Goal
Navigate from your position (A) to the goal (G) by choosing a sequence of moves.
The shortest possible path to the goal is exactly {n_remaining} steps.

# Actions
You must output a string of exactly {n_remaining} characters, each one of:
- `U`: Move up
- `D`: Move down
- `L`: Move left
- `R`: Move right

If a move would walk into a wall or out of bounds, the agent stays in place — so plan carefully.

# Observation
The initial maze state is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to plan your path.
- First, reason step-by-step: locate A and G on the grid, trace the walls, and plan a valid route.
- Then output exactly {n_remaining} characters within <action> </action> tags.
  Example for n_remaining=4: <action>LLUR</action>
"""

MAZE_REFLECT_PROMPT = '''
You are an expert agent navigating a maze.

# Symbols
- `A`: Your current position (agent)
- `G`: The goal you must reach
- `#`: Wall (impassable)
- `.`: Open passage (passable)

# Goal
Navigate from A to G in exactly {n_remaining} steps.

# Your Task
You will be given the history of a past failed attempt.
Reflect on what went wrong and devise an improved action sequence.

# Past Experience
The initial maze state is:
{init_observation}{current_trajectory}
The task was NOT successfully completed.

Now reflect on your past attempt:
- Reason step-by-step about which moves were wrong and why.
- Identify where you got stuck or took a wrong turn.
- Propose a corrected plan with specific reference to the grid layout.
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


def get_maze_prompt(
    n_remaining: int,
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
        prompt = MAZE_PLAY_PROMPT.format(
            n_remaining=n_remaining,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = MAZE_REFLECT_PROMPT.format(
            n_remaining=n_remaining,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
