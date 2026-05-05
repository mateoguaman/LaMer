LANGUAGE_TABLE_PLAY_PROMPT = """
You are an expert agent operating in a Language-Table environment.

# Goal
You steer a language-conditioned robot policy that pushes colored blocks on a table by issuing short natural language commands. You have multiple trials - the policy may misinterpret commands, so learn from prior trials which commands it actually responds to.

# Rules
- Coordinates: (x, y), x=right, y=up.
- Issue ONE short natural language command per turn.
- Treat the table as a 3x3 grid with the following locations:
    top left | top middle | top right
    middle left | middle | middle right
    bottom left | bottom middle | bottom right
- Hints:
    - Blocks jammed in a corner or along an edge are hard to push somewhere else.
    - When blocks sit very close together, separating them is hard.
    - Pushing a block through or past other blocks to reach its target pose is hard. Choose the order of pushing blocks to avoid collisions.
    - When pushing blocks prefer shortest paths to the target location.

# Observations
The initial state of the environment is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to issue a command.

- Your response should first be step-by-step reasoning about the current situation.
- Then conclude with a single short natural language command within <action> </action> tags.
"""

# Example: an L-shaped formation can be created by pushing blocks to the top left, middle left, bottom left, and bottom middle.


LANGUAGE_TABLE_REFLECT_PROMPT = """
You are an expert agent operating in a Language-Table environment.

# Goal
You steer a language-conditioned robot policy that pushes colored blocks on a table by issuing short natural language commands.

# Rules
- Coordinates: (x, y), x=right, y=up.
- The policy may misinterpret commands.
- Treat the table as a 3x3 grid with the following locations:
    top left | top middle | top right
    middle left | middle | middle right
    bottom left | bottom middle | bottom right
- Hints:
    - Blocks jammed in a corner or along an edge are hard to push somewhere else.
    - When blocks sit very close together, separating them is hard.
    - Pushing a block through or past other blocks to reach its target pose is hard.

# Your Task
You will be given the history of a past trial. Reflect on it, identify mistakes or ineffective commands, and devise a concise, improved plan starting from the original initial state.

# Past Experience
The initial state of the environment is:
{init_observation}{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect and come up with a new plan.

- Your response should first be step-by-step reasoning about which commands the policy did/didn't respond to and where things went wrong.
- Then devise a concise new plan with specific commands to try next.
- Finally, end with your reflection and improved plan inside <remark> </remark> tags to guide the next trial.
"""


PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, you took the following actions:
{past_trajectory}
The task is NOT successfully completed. Your reflection is:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, you took the following actions:
{past_trajectory}
The task is NOT successfully completed.'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return "\n"
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
        return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    if turn_idx == 0:
        return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
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
    play_template: str | None = None,
    **_deprecated_kwargs,
):
    """Build a play or reflect prompt.

    `play_template` lets callers (e.g. the OPRO optimizer) override the default
    `LANGUAGE_TABLE_PLAY_PROMPT` with a learned/rewritten template. The
    template must still contain the three placeholders `{init_observation}`,
    `{past_trajectories_reflections}`, `{current_trajectory}` because they are
    filled with structured observation context here.
    """
    assert phase in ["play", "reflect"]

    if phase == "play":
        template = play_template if play_template is not None else LANGUAGE_TABLE_PLAY_PROMPT
        for ph in _REQUIRED_PLAY_PLACEHOLDERS:
            if ph not in template:
                raise ValueError(
                    f"play_template missing required placeholder {ph!r}; "
                    f"OPRO rewrites must preserve the placeholder contract."
                )
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = template.format(
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