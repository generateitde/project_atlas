from __future__ import annotations


class DAgger:
    def __init__(self) -> None:
        self.uncertainty_threshold = 0.5

    def needs_query(self, uncertainty: float) -> bool:
        return uncertainty >= self.uncertainty_threshold
