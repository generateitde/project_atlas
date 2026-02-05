from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log


@dataclass
class DAgger:
    uncertainty_threshold: float = 0.55
    stuck_window: int = 16
    min_stuck_unique_positions: int = 3
    no_progress_limit: int = 24
    _recent_positions: deque[tuple[int, int]] = field(init=False)
    _steps_without_progress: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._recent_positions = deque(maxlen=self.stuck_window)

    def reset(self) -> None:
        self._recent_positions.clear()
        self._steps_without_progress = 0

    def update_stuck_state(self, position: tuple[int, int], progress_signal: float, done: bool = False) -> bool:
        if done:
            self.reset()
            return False
        self._recent_positions.append(position)
        if progress_signal > 0:
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        no_progress_stuck = self._steps_without_progress >= self.no_progress_limit
        if len(self._recent_positions) < self.stuck_window:
            return no_progress_stuck
        unique_positions = len(set(self._recent_positions))
        loop_stuck = unique_positions <= self.min_stuck_unique_positions
        return loop_stuck or no_progress_stuck

    @staticmethod
    def entropy_from_probs(action_probs: list[float]) -> float:
        valid_probs = [p for p in action_probs if p > 0.0]
        if not valid_probs:
            return 0.0
        entropy = -sum(p * log(p) for p in valid_probs)
        max_entropy = log(len(action_probs)) if len(action_probs) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def needs_query(self, *, stuck: bool, uncertainty: float) -> bool:
        return stuck and uncertainty >= self.uncertainty_threshold
