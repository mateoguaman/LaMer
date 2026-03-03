import re
import copy
from typing import List, Tuple

_VALID_CHARS = frozenset("UDLR")


def maze_projection(
    actions: List[str],
    n_remaining: int,
    phase: str = 'play',
) -> Tuple:
    """
    Parse LLM text outputs into validated maze action strings.

    Play phase
    ----------
    Expects the LLM to produce a string of U/D/L/R characters inside
    ``<action>...</action>`` tags.  Returns ``(thoughts, actions, valids)``.

    Reflect phase
    -------------
    Expects a reflection inside ``<remark>...</remark>`` tags.
    Returns ``(reflections, valids)``.

    Parameters
    ----------
    actions : List[str]
        Raw LLM outputs, one per environment.
    n_remaining : int
        Expected number of moves (characters) in the action string.
        Extracted strings are truncated to this length.
    phase : str
        ``'play'`` or ``'reflect'``.
    """
    actions = copy.deepcopy(actions)

    if phase == 'play':
        valids = [0] * len(actions)
        thoughts = [''] * len(actions)

        for i, raw in enumerate(actions):
            original_str = raw

            start_idx = raw.find("<action>")
            end_idx = raw.rfind("</action>")

            if start_idx == -1 or end_idx == -1:
                actions[i] = ''
                valids[i] = 0
                continue

            extracted = raw[start_idx + len("<action>"):end_idx].strip().upper()
            # Keep only valid direction characters, then truncate
            filtered = ''.join(c for c in extracted if c in _VALID_CHARS)
            filtered = filtered[:n_remaining]

            if filtered:
                actions[i] = filtered
                valids[i] = 1
            else:
                actions[i] = ''
                valids[i] = 0

            # Extract optional <plan> block as the "thought"
            plan_start = original_str.rfind("<plan>")
            plan_end = original_str.rfind("</plan>")
            if plan_start != -1 and plan_end != -1:
                thoughts[i] = original_str[
                    plan_start + len("<plan>"):plan_end
                ].strip()

        return thoughts, actions, valids

    else:  # reflect phase
        valids = [0] * len(actions)
        reflections = [''] * len(actions)

        for i, raw in enumerate(actions):
            start_idx = raw.rfind("<remark>")
            end_idx = raw.rfind("</remark>")
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = raw[
                    start_idx + len("<remark>"):end_idx
                ].strip()[:2000]
                valids[i] = 1

        return reflections, valids
