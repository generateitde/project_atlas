from __future__ import annotations


class PreferenceRewardModel:
    def __init__(self) -> None:
        self.data: list[tuple[str, int]] = []

    def add_feedback(self, text: str, score: int) -> None:
        self.data.append((text, score))

    def score(self, text: str) -> float:
        return 0.0
