##### LANGUAGE TABLE PROMPT TEMPLATES FOR METARL #####

# Each command specifies a direction and distance (e.g., "move up by 0.1, then move right by 0.05").

LANGUAGE_TABLE_PLAY_PROMPT = """You are an expert robot control agent operating a tabletop manipulation environment.

# Environment
A robot end-effector can push objects on a table. You observe the task instruction,
the end-effector position, and the positions of colored blocks on the table.
Coordinates are (x, y) where x increases rightward and y increases upward.

# Your Job
Read the observation and output a **sequence of movement commands** in natural
language that tell the end-effector where to move. The movements are executed open-loop — you will not see intermediate states.

# Important: Adversarial Environment
This environment may be adversarial. The actual effect of your movement commands
might not match what you intend. This disturbance is deterministic for the
entire environment instance.
By observing the outcomes of your actions across multiple attempts, you can deduce
the true mapping and adapt your strategy accordingly.

# Observations
{init_observation}{past_trajectories_reflections}{current_trajectory}

# Output Format
- First, reason step-by-step: identify the task, locate relevant objects, and
  plan the end-effector movements needed. If you have prior attempts, reason
  about how your movements actually affected the end-effector versus what you
  expected, to infer whether a disturbance is present.
- Then output your movement commands inside <action> </action> tags.

Example: <action>move right by 0.1, then move up by 0.2, then move left by 0.15, then move down by 0.1</action>

"""

LANGUAGE_TABLE_REFLECT_PROMPT = """You are an expert robot control agent operating a tabletop manipulation environment.

# Environment
A robot end-effector can push objects on a table. You observe the task instruction,
the end-effector position, and the positions of colored blocks on the table.
Coordinates are (x, y) where x increases rightward and y increases upward.

# Important: Adversarial Environment
This environment may be adversarial. The actual effect of your movement commands
might not match what you intend. This disturbance is deterministic for the
entire environment instance.

# Your Task
You will be given the history of a past failed attempt. Reflect on what went
wrong and propose improved movement commands for the next attempt.

# Past Experience
{init_observation}
{current_trajectory}
The task was NOT successfully completed.

# Output Format
- Reason step-by-step about which movements were wrong and why.
- Identify whether the movement effects matched your expectations or appeared
  distorted. If you detected a disturbance, describe it explicitly.
- Propose a corrected plan using your updated understanding.
- End with your reflection inside <remark> </remark> tags to guide the next trial.
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


CURR_TRAJ_AT_TRAJ1 = """
You previously issued the movement commands: {current_trajectory}
"""

CURR_TRAJ_AT_TRAJ2toN = """
Currently you're on attempt #{traj_idx}. You previously issued the movement commands: {current_trajectory}
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
