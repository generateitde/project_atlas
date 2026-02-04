from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatMessage:
    sender: str
    text: str


@dataclass
class ChatState:
    messages: List[ChatMessage] = field(default_factory=list)
    draft: str = ""


class ChatUI:
    def __init__(self) -> None:
        self.state = ChatState()

    def add_message(self, sender: str, text: str) -> None:
        self.state.messages.append(ChatMessage(sender=sender, text=text))
