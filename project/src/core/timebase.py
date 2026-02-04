from __future__ import annotations

import time


class TimeBase:
    def __init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self._last
        self._last = now
        return delta
