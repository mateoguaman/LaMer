# NOTE: "conclude" is important, so the agent puts the action at the end of the response!
LANGUAGE_TABLE_PLAY_PROMPT = """You control a robot end-effector that pushes colored blocks on a table by instructing it with short natural language commands.
Coordinates: (x, y), x=right, y=up.

The environment may be adversarial - commands may not produce intended effects. The environment is deterministic - learn from prior attempts to discover which actions the environment responds to. To get you started, try rewording, decomposing, or restructuring the commands to make it more effective.

{init_observation}{past_trajectories_reflections}{current_trajectory}

Reason step-by-step, then conclude with your command in <action> </action> tags.
"""

LANGUAGE_TABLE_REFLECT_PROMPT = """You control a robot end-effector that pushes colored blocks on a table by instructing it with short natural language commands.
Coordinates: (x, y), x=right, y=up.

The environment may be adversarial - commands may not produce intended effects. The environment is deterministic - learn from prior attempts to discover which actions the environment responds to. To get you started, try rewording, decomposing, or restructuring the commands to make it more effective.

{init_observation}
{current_trajectory}
The task FAILED.

Reason step-by-step, to analyze the outcomes of your previous commands and whether a disturbance is present. Conclude with your reflection inside <remark> </remark> tags to guide the next trial.
"""

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = """
On attempt #{traj_idx}, you issued the movement commands: {past_trajectory}
The task was NOT successfully completed. Your reflection was:
{reflection}"""

HISTORY_ONLY_TEMPLATE = """
On attempt #{traj_idx}, you issued the movement commands: {past_trajectory}
The task was NOT successfully completed."""

REFLECTION_ONLY_TEMPLATE = """
On attempt #{traj_idx}, the task was NOT successfully completed. Your reflection was:
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


TRAJ_1_INIT = """
You're on step {turn_idx}/{max_turns}."""

CURR_TRAJ_AT_TRAJ1 = """
You're on step {turn_idx}/{max_turns}. You previously issued the movement commands: {current_trajectory}
"""

CURR_TRAJ_AT_TRAJ2toN = """
Currently you're on attempt #{traj_idx}. You're on step {turn_idx}/{max_turns}. You previously issued the movement commands: {current_trajectory}
"""

TRAJ_2toN_INIT = """
Currently you're on attempt #{traj_idx}, starting fresh. You're on step {turn_idx}/{max_turns}."""


def parse_current_trajectory(turn_idx, traj_idx, curr_traj, max_turns):
    if traj_idx == 0:
        if turn_idx == 0:
            return TRAJ_1_INIT.format(turn_idx=turn_idx + 1, max_turns=max_turns)
        else:
            return CURR_TRAJ_AT_TRAJ1.format(
                turn_idx=turn_idx + 1, max_turns=max_turns,
                current_trajectory=curr_traj,
            )
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(
                traj_idx=traj_idx + 1, turn_idx=turn_idx + 1, max_turns=max_turns,
            )
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1, turn_idx=turn_idx + 1, max_turns=max_turns,
                current_trajectory=curr_traj,
            )


def get_language_table_prompt(
    phase: str = "play",
    turn_idx: int = 0,
    traj_idx: int = 0,
    max_turns: int = 1,
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
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj, max_turns)
        prompt = LANGUAGE_TABLE_PLAY_PROMPT.format(
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj, max_turns)
        prompt = LANGUAGE_TABLE_REFLECT_PROMPT.format(
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
