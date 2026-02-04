from __future__ import annotations


def validate_chat_goal(text: str) -> bool:
    return len(text.strip()) > 0
