import copy
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def robolab_projection(actions: List[str], phase="play"):
    """Extract structured output from LLM text for RoboLab.

    Play phase:
        Extracts the command string from <action>...</action> tags.
        Returns (commands, valids).

    Reflect phase:
        Extracts the reflection from <remark>...</remark> tags.
        Returns (reflections, valids).
    """
    actions = copy.deepcopy(actions)

    if phase == "play":
        valids = [0] * len(actions)
        commands = [""] * len(actions)

        for i in range(len(actions)):
            text = actions[i]

            matches = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)
            if matches:
                extracted = matches[-1].strip()
                if extracted:
                    commands[i] = extracted
                    valids[i] = 1

            if valids[i] == 0:
                lines = [
                    line.strip()
                    for line in text.strip().splitlines()
                    if line.strip()
                ]
                if lines:
                    last_line = lines[-1]
                    if len(last_line) <= 100:
                        commands[i] = last_line

            logger.warning(
                "Projection env=%d phase=play valid=%d raw=%r command=%r",
                i, valids[i], text, commands[i],
            )

        return commands, valids

    else:
        valids = [0] * len(actions)
        reflections = [""] * len(actions)

        for i in range(len(actions)):
            text = actions[i]
            start_tag = "<remark>"
            end_tag = "</remark>"
            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag)

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                reflections[i] = text[start_idx + len(start_tag):end_idx].strip()
                valids[i] = 1

        return reflections, valids
