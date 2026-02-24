import re
import copy
from typing import List, Tuple

VALID_DIRECTIONS = {'up', 'down', 'left', 'right'}


def maze_projection(actions: List[str], plan_length: int = 5, phase='play'):
    """
    Parse LLM text outputs into environment actions.

    For phase='play':
        Extract <action>up,up,right,down,left</action> -> list of direction strings
        Returns: (plans, parsed_actions, valids)

    For phase='reflect':
        Extract <remark>...</remark> -> reflection text
        Returns: (reflections, valids)
    """
    actions = copy.deepcopy(actions)

    if phase == 'play':
        valids = [0] * len(actions)
        plans = [''] * len(actions)
        parsed_actions = [[] for _ in range(len(actions))]

        for i in range(len(actions)):
            original_str = actions[i]

            # Extract <action> content
            start_idx = original_str.find("<action>")
            end_idx = original_str.find("</action>")

            if start_idx == -1 or end_idx == -1:
                valids[i] = 0
                parsed_actions[i] = ['up'] * plan_length  # default invalid action
                continue

            extracted = original_str[start_idx + len("<action>"):end_idx].strip()

            # Parse comma-separated directions
            directions = [d.strip().lower() for d in extracted.split(',')]

            # Validate directions
            all_valid = True
            clean_directions = []
            for d in directions:
                if d in VALID_DIRECTIONS:
                    clean_directions.append(d)
                else:
                    all_valid = False

            if len(clean_directions) == 0:
                valids[i] = 0
                parsed_actions[i] = ['up'] * plan_length
                continue

            # Pad or truncate to plan_length
            if len(clean_directions) < plan_length:
                clean_directions.extend([clean_directions[-1]] * (plan_length - len(clean_directions)))
            elif len(clean_directions) > plan_length:
                clean_directions = clean_directions[:plan_length]

            parsed_actions[i] = clean_directions
            valids[i] = 1

            # Extract <plan> content (optional reasoning)
            plan_start = original_str.rfind("<plan>")
            plan_end = original_str.rfind("</plan>")
            if plan_start != -1 and plan_end != -1:
                plans[i] = original_str[plan_start + len("<plan>"):plan_end].strip()

        return plans, parsed_actions, valids

    else:
        # reflect phase
        valids = [0] * len(actions)
        reflections = [''] * len(actions)

        for i in range(len(actions)):
            action = actions[i]
            start_tag = "<remark>"
            start_idx = action.rfind(start_tag)
            end_tag = "</remark>"
            end_idx = action.rfind(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = action[start_idx + len(start_tag):end_idx].strip()[:2000]
                valids[i] = 1

        return reflections, valids
