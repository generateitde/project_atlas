from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumState:
    level: int = 1


class Curriculum:
    def __init__(self) -> None:
        self.state = CurriculumState()

    def update(self, exp: int) -> None:
        if exp >= self.state.level * 10:
            self.state.level += 1
