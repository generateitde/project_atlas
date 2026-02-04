from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillRegistry:
    skills: dict[str, dict] = field(default_factory=dict)

    def add(self, name: str, payload: dict) -> None:
        self.skills[name] = payload
