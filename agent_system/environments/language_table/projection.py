import copy
from typing import List


def language_table_projection(actions: List[str], phase="play"):
    """Extract structured output from LLM text for Language Table.

    Play phase:
        Extracts the goal string from <action>...</action> tags.
        Returns (goals, valids) where goals are the extracted strings
        and valids indicate whether extraction succeeded.

    Reflect phase:
        Extracts the reflection from <remark>...</remark> tags.
        Returns (reflections, valids).
    """
    actions = copy.deepcopy(actions)

    if phase == "play":
        valids = [0] * len(actions)
        goals = [""] * len(actions)

        for i in range(len(actions)):
            text = actions[i]

            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = text.find(start_tag)
            end_idx = text.rfind(end_tag)

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                extracted = text[start_idx + len(start_tag):end_idx].strip()
                if extracted:
                    goals[i] = extracted
                    valids[i] = 1

            # Fallback: if no tags found, try to use the last line as the goal.
            # This handles cases where the model forgets the tags but still
            # produces a reasonable goal string.
            if valids[i] == 0:
                lines = [
                    line.strip()
                    for line in text.strip().splitlines()
                    if line.strip()
                ]
                if lines:
                    last_line = lines[-1]
                    # Only use fallback if it looks like a short goal
                    # (not a long reasoning paragraph)
                    if len(last_line) <= 100:
                        goals[i] = last_line
                        # Still mark as invalid so the penalty applies
                        valids[i] = 0

        return goals, valids

    else:
        # Reflect phase
        valids = [0] * len(actions)
        reflections = [""] * len(actions)

        for i in range(len(actions)):
            text = actions[i]
            start_tag = "<remark>"
            end_tag = "</remark>"
            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag)

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                reflections[i] = text[
                    start_idx + len(start_tag):end_idx
                ].strip()[:2000]
                valids[i] = 1

        return reflections, valids
