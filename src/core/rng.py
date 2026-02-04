from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RNG:
    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()
