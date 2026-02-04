from __future__ import annotations

import random
from typing import Optional


class RNG:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)
