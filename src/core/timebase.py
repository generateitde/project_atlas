from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timebase:
    target_hz: int

    def __post_init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self._last
        self._last = now
        return delta

    def sleep_to_rate(self) -> None:
        if self.target_hz <= 0:
            return
        target = 1.0 / self.target_hz
        now = time.perf_counter()
        elapsed = now - self._last
        if elapsed < target:
            time.sleep(target - elapsed)
