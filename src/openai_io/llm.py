from __future__ import annotations

from typing import Optional

from src.config import AtlasConfig


class LLMClient:
    def __init__(self, config: AtlasConfig) -> None:
        self.config = config

    def propose(self, prompt: str) -> Optional[str]:
        if not self.config.toggles.use_llm:
            return None
        return "Atlas (Hypothese): Noch keine Hypothesen verfügbar."
