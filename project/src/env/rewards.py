from __future__ import annotations

from typing import Dict

from src.core.events import Event


class RewardTracker:
    def __init__(self) -> None:
        self.discovered = set()

    def exploration_reward(self, tiles_visible) -> float:
        newly = 0
        for pos in tiles_visible:
            if pos not in self.discovered:
                self.discovered.add(pos)
                newly += 1
        return 0.01 * newly

    def progress_reward(self, events: list[Event]) -> float:
        reward = 0.0
        for event in events:
            if event.type in {"pickup", "door_open", "gate_pass", "goal_reached", "flag_scored"}:
                reward += 0.5
        return reward

    def shaping_reward(self, info: Dict[str, float]) -> float:
        return info.get("distance_shaping", 0.0)
