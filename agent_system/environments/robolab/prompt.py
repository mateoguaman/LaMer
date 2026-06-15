ROBOLAB_PLAY_PROMPT = """
You are an expert agent controlling a robot arm in a physics simulation. Stay close to the provided task instruction. Do not do subtask decomposition.

# Task
{language_instruction}

# Observation
The current state of the environment is shown in the image below:
<image>

# Instructions
- Issue ONE short natural language command per turn to control the robot.
- Your command will be interpreted by a language-conditioned robot policy.
- The policy may not always execute commands perfectly — adapt based on what you observe.

# Response Format
- First, reason step-by-step about the current state and what action to take next.
- Then output exactly one command inside <action> </action> tags.

Example:
<action>Put the lizards in the bin</action>
<action>Put the bagels on the plate</action>
<action>Pick up the banana and place it in the bowl</action>
<action>Put the canned food in the grey bin</action>
<action>Move the bananas to the bagel plate</action>
{current_trajectory}
"""

ROBOLAB_REFLECT_PROMPT = """
You are an expert agent controlling a robot arm in a physics simulation.

# Task
{language_instruction}

# Past Trial
The initial state of the environment is shown below:
<image>
{current_trajectory}
The task was NOT successfully completed.

# Instructions
- Reflect on what went wrong in this trial.
- Identify which commands the policy responded to and which it did not.
- Devise a concise improved plan for the next attempt.

# Response Format
- First, reason step-by-step about what went wrong and why.
- Then output your reflection and improved plan inside <remark> </remark> tags.
"""

PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = """
On trial #{traj_idx}, you observed the following states and took the following actions:
{past_trajectory}
The task was NOT successfully completed. Your reflection is:
{reflection}"""

HISTORY_ONLY_TEMPLATE = """
On trial #{traj_idx}, you observed the following states and took the following actions:
{past_trajectory}
The task was NOT successfully completed."""

REFLECTION_ONLY_TEMPLATE = """
On trial #{traj_idx}, the task was NOT successfully completed. Your reflection is:
{reflection}"""

CURR_TRAJ_AT_TRAJ1 = """
You have already observed the following states and taken the following actions this trial:
{current_trajectory}
"""

CURR_TRAJ_AT_TRAJ2toN = """
Currently you're on trial #{traj_idx}. You have already observed the following states and taken the following actions:
{current_trajectory}
"""

TRAJ_2toN_INIT = """
Currently you're on trial #{traj_idx}, starting from the initial state."""


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return ""
    memories = []
    for _idx in range(traj_idx):
        if reflection_type == "history_and_reflection":
            memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                traj_idx=_idx + 1,
                past_trajectory=past_traj.get(_idx, ""),
                reflection=reflection.get(_idx, ""),
            )
        elif reflection_type == "history_only":
            memory = HISTORY_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1,
                past_trajectory=past_traj.get(_idx, ""),
            )
        elif reflection_type == "reflection_only":
            memory = REFLECTION_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1,
                reflection=reflection.get(_idx, ""),
            )
        else:
            raise ValueError(f"Unknown reflection_type: {reflection_type}")
        memories.append(memory)
    return "".join(memories)


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    if turn_idx == 0:
        return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
    return CURR_TRAJ_AT_TRAJ2toN.format(
        traj_idx=traj_idx + 1,
        current_trajectory=curr_traj,
    )


def get_robolab_prompt(
    phase: str = "play",
    turn_idx: int = 0,
    traj_idx: int = 0,
    language_instruction: str = "",
    curr_traj: str = "",
    past_traj: dict = {},
    reflection: dict = {},
    reflection_type: str = "reflection_only",
    **_kwargs,
):
    assert phase in ["play", "reflect"]

    if phase == "play":
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        combined_trajectory = past_trajectories_reflections + current_trajectory
        prompt = ROBOLAB_PLAY_PROMPT.format(
            language_instruction=language_instruction,
            current_trajectory=combined_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = ROBOLAB_REFLECT_PROMPT.format(
            language_instruction=language_instruction,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
