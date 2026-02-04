from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def generate(self, prompt: str) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        return LLMResponse(text="Atlas (Hypothese): LLM ist deaktiviert.")
