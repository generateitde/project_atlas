from __future__ import annotations


class VoiceInput:
    def __init__(self) -> None:
        self.state = "IDLE"

    def capture(self) -> str:
        return ""
