import copy
import logging
import re
import threading
from typing import List

logger = logging.getLogger(__name__)
_INVALID_WARNING_LOCK = threading.Lock()
_INVALID_WARNING_COUNT = 0
_INVALID_WARNING_SUPPRESSED = 0
_MAX_INVALID_WARNING_LOGS = 50
_INVALID_WARNING_SUMMARY_EVERY = 100


def _log_play_projection(i: int, valid: int, text: str, goal: str) -> None:
    global _INVALID_WARNING_COUNT, _INVALID_WARNING_SUPPRESSED

    if valid:
        logger.debug(
            "Projection env=%d phase=play valid=%d raw=%r goal=%r",
            i,
            valid,
            text,
            goal,
        )
        return

    with _INVALID_WARNING_LOCK:
        if _INVALID_WARNING_COUNT < _MAX_INVALID_WARNING_LOGS:
            _INVALID_WARNING_COUNT += 1
            logger.warning(
                "Projection env=%d phase=play valid=%d raw=%r goal=%r",
                i,
                valid,
                text,
                goal,
            )
            return

        _INVALID_WARNING_SUPPRESSED += 1
        if _INVALID_WARNING_SUPPRESSED % _INVALID_WARNING_SUMMARY_EVERY == 0:
            logger.warning(
                "Projection invalid-output warnings suppressed=%d after first %d logs",
                _INVALID_WARNING_SUPPRESSED,
                _MAX_INVALID_WARNING_LOGS,
            )


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

            # Use non-greedy regex to find all complete <action>...</action>
            # pairs, then take the last one.  This avoids the old find/rfind
            # bug where multiple action blocks caused everything between the
            # first opening and last closing tag to be captured (including
            # chain-of-thought reasoning leaked between blocks).
            matches = re.findall(
                r"<action>(.*?)</action>", text, re.DOTALL
            )
            if matches:
                extracted = matches[-1].strip()
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

            _log_play_projection(i, valids[i], text, goals[i])

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
                ].strip()#[:2000]
                valids[i] = 1

        return reflections, valids
