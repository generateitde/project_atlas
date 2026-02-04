from __future__ import annotations


class ChatUI:
    def __init__(self) -> None:
        self.buffer = ""

    def submit(self) -> str:
        text = self.buffer
        self.buffer = ""
        return text
