MAZE_PLAY_PROMPT = """
You are an expert agent navigating through an ASCII maze.
You will be given a {maze_size} maze grid where:
- `#` = wall (impassable)
- `.` = open path
- `A` = your current position
- `G` = goal position

# Stochastic Action Execution
**WARNING**: Your actions may not execute as intended! The environment has stochastic effects:
- Your action may be replaced by a random direction (noise)
- Your previous action may repeat instead (sticky)
- Your action may execute twice, moving you 2 cells (double)
- Some directions may be unavailable

Because of this, you should plan carefully and adapt based on what actually happens vs what you planned.

# Action Format
Each turn, you must submit a plan of exactly {plan_length} movement steps.
The valid directions are: up, down, left, right
Put your plan as comma-separated directions inside <action> </action> tags.
Example: <action>up,up,right,down,left</action>

# Rewards
- Reaching the goal (G): +10.0
- Each step taken: -0.1
- Hitting a wall (wasted move): -0.5
- The game ends when you reach the goal or exceed the step limit.

# Observation
The initial state of the maze is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to plan your next {plan_length} moves.
- First, reason step-by-step: analyze your current position, the goal location, and any discrepancies between your planned and executed actions from previous turns.
- Then output your {plan_length}-step plan inside <action> </action> tags.
"""


MAZE_REFLECT_PROMPT = '''
You are an expert agent navigating through an ASCII maze.
You will be given a {maze_size} maze grid where:
- `#` = wall (impassable)
- `.` = open path
- `A` = your current position
- `G` = goal position

# Stochastic Action Execution
Your actions may not execute as intended due to stochastic effects (noise, sticky, double, unavailable).

# Your Task
You will be given the history of a past attempt to navigate the maze.
Your job is to **reflect on the past experience**, identify any **patterns in the stochastic effects**, and devise a **concise, improved strategy** for your next attempt.

# Past Experience
The initial state of the maze is:
{init_observation}{current_trajectory}
The task is NOT successfully completed.

Now reflect on the past experience and come up with a new plan of action.
- Analyze the discrepancies between planned and executed actions to identify stochastic patterns.
- Consider which directions seem unreliable or which effects are most common.
- Devise a strategy that accounts for the observed stochastic behavior.
- End your response with your reflection and improved plan inside <remark> </remark> tags.
'''

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is NOT successfully completed. Your reflection is:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is NOT successfully completed.'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    else:
        memories = []
        for _idx in range(traj_idx):
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")

            memories.append(memory)
        return ''.join(memories)


# Prompt templates for parsing current trajectory
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
            return CURR_TRAJ_AT_TRAJ1.format(
                current_trajectory=curr_traj
            )
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )


def get_maze_prompt(
    maze_size: int,
    plan_length: int,
    phase: str = 'play',
    turn_idx: int = 0,
    traj_idx: int = 0,
    init_observation: str = '',
    curr_traj: str = '',
    past_traj: str = '',
    reflection: str = '',
    reflection_type: str = 'reflection_only',
):
    assert phase in ['play', 'reflect']

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = MAZE_PLAY_PROMPT.format(
            maze_size=maze_size,
            plan_length=plan_length,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = MAZE_REFLECT_PROMPT.format(
            maze_size=maze_size,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
