from __future__ import annotations


class SpeechToText:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def transcribe(self, audio_bytes: bytes) -> str:
        return ""
