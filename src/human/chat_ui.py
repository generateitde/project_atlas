from __future__ import annotations

from typing import Iterable

from src.env.encoding import ACTION_MEANINGS


class ChatUI:
    def __init__(self) -> None:
        self.buffer = ""

    def submit(self) -> str:
        text = self.buffer
        self.buffer = ""
        return text


ACTION_ALIASES: dict[str, int] = {
    "noop": 0,
    "wait": 0,
    "n": 1,
    "north": 1,
    "w": 1,
    "e": 2,
    "east": 2,
    "d": 2,
    "s": 3,
    "south": 3,
    "a": 4,
    "west": 4,
    "jump": 5,
    "pickup": 6,
    "drop": 7,
    "use": 8,
    "attack": 9,
    "break": 10,
    "inspect": 11,
    "speak": 12,
    "ask": 13,
}


def _allowed_actions_from_mask(action_mask: Iterable[bool] | None) -> set[int]:
    if action_mask is None:
        return set(ACTION_MEANINGS.keys())
    return {idx for idx, allowed in enumerate(action_mask) if bool(allowed)}


def parse_human_action_choice(text: str, action_mask: Iterable[bool] | None = None) -> int | None:
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    allowed = _allowed_actions_from_mask(action_mask)
    if cleaned.isdigit():
        action_id = int(cleaned)
        return action_id if action_id in allowed else None

    normalized = cleaned.replace(" ", "_")
    if normalized in ACTION_ALIASES:
        action_id = ACTION_ALIASES[normalized]
        return action_id if action_id in allowed else None

    for action_id, action_name in ACTION_MEANINGS.items():
        if normalized == action_name.lower() and action_id in allowed:
            return action_id
    return None


def format_action_choices(action_mask: Iterable[bool] | None = None) -> str:
    allowed = _allowed_actions_from_mask(action_mask)
    choices = [f"{action_id}:{name}" for action_id, name in ACTION_MEANINGS.items() if action_id in allowed]
    return ", ".join(choices)
