from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DaggerState:
    uncertainty: float = 0.0


class DaggerHelper:
    def __init__(self) -> None:
        self.state = DaggerState()

    def should_query(self) -> bool:
        return self.state.uncertainty > 0.7
