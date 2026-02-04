from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MessageType = Literal["idea", "hypothesis", "plan", "question"]


@dataclass
class AtlasMessage:
    msg_type: MessageType
    text: str

    def formatted(self) -> str:
        if self.msg_type == "hypothesis":
            return f"Atlas (Hypothese): {self.text}"
        if self.msg_type == "plan":
            return f"Atlas (Plan): {self.text}"
        if self.msg_type == "question":
            return f"Atlas (Frage): {self.text}"
        return f"Atlas: {self.text}"
