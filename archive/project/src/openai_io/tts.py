from __future__ import annotations


class TextToSpeech:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def synthesize(self, text: str) -> bytes:
        return b""
