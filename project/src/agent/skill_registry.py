from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SkillRegistry:
    skills: Dict[str, str] = field(default_factory=dict)

    def register(self, name: str, description: str) -> None:
        self.skills[name] = description
