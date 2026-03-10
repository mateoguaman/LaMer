##### LANGUAGE TABLE PROMPT TEMPLATES FOR METARL #####

LANGUAGE_TABLE_PLAY_PROMPT = """You are an expert robot control agent operating a tabletop manipulation environment.

# Environment
A robot end-effector can push objects on a table. You observe the task instruction, the end-effector position, and the positions of colored blocks on the table.

# Your Job
Read the observation and output a **short natural-language goal** that tells the robot what to do. The goal should be a simple imperative sentence describing the desired movement (e.g., "push the red star to the blue cube").

# Observations
{init_observation}{past_trajectories_reflections}{current_trajectory}

# Output Format
- You may first reason briefly about which objects to move and where.
- Then output your goal inside <action> </action> tags.
- The goal must be a short, clear sentence (under 20 words).

Example: <action>push the red star to the blue cube</action>
"""

LANGUAGE_TABLE_REFLECT_PROMPT = """You are an expert robot control agent operating a tabletop manipulation environment.

# Environment
A robot end-effector can push objects on a table. You observe the task instruction, the end-effector position, and the positions of colored blocks on the table.

# Your Task
You will be given the history of a past attempt. Reflect on what went wrong and propose an improved goal for the next attempt.

# Past Experience
{init_observation}
{current_trajectory}
The task was NOT successfully completed.

# Output Format
- First, briefly analyze what went wrong in the previous attempt.
- Then provide your reflection inside <remark> </remark> tags to guide the next trial.
"""

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = """
On attempt #{traj_idx}, you issued the goal: {past_trajectory}
The task was NOT successfully completed. Your reflection is:
{reflection}"""

HISTORY_ONLY_TEMPLATE = """
On attempt #{traj_idx}, you issued the goal: {past_trajectory}
The task was NOT successfully completed."""

REFLECTION_ONLY_TEMPLATE = """
On attempt #{traj_idx}, the task was NOT successfully completed. Your reflection is:
{reflection}"""


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return "\n"
    else:
        memories = []
        for _idx in range(traj_idx):
            if reflection_type == "history_and_reflection":
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx],
                )
            elif reflection_type == "history_only":
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == "reflection_only":
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx],
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")
            memories.append(memory)
        return "".join(memories)


CURR_TRAJ_AT_TRAJ1 = """
You previously issued the goal: {current_trajectory}
"""

CURR_TRAJ_AT_TRAJ2toN = """
Currently you're on attempt #{traj_idx}. You previously issued the goal: {current_trajectory}
"""

TRAJ_2toN_INIT = """
Currently you're on attempt #{traj_idx}, starting fresh."""


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


def get_language_table_prompt(
    phase: str = "play",
    turn_idx: int = 0,
    traj_idx: int = 0,
    init_observation: str = "",
    curr_traj: str = "",
    past_traj: dict = {},
    reflection: str = "",
    reflection_type: str = "reflection_only",
):
    assert phase in ["play", "reflect"]

    if phase == "play":
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = LANGUAGE_TABLE_PLAY_PROMPT.format(
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = LANGUAGE_TABLE_REFLECT_PROMPT.format(
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
