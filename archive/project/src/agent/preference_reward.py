from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PreferenceSample:
    text: str
    score: int


class PreferenceRewardModel:
    def __init__(self) -> None:
        self.samples: List[PreferenceSample] = []

    def add(self, text: str, score: int) -> None:
        self.samples.append(PreferenceSample(text=text, score=score))

    def score(self, text: str) -> float:
        if not self.samples:
            return 0.0
        mean = sum(sample.score for sample in self.samples) / len(self.samples)
        return float(mean)
