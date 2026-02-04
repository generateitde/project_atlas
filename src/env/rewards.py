from __future__ import annotations

from src.core.events import Event


def compute_reward(mode_reward: float, events: list[Event], step_cost: float = 0.01) -> float:
    progress_reward = 0.0
    exploration_reward = 0.0
    shaping = 0.0
    preference_reward = 0.0
    for event in events:
        if event.type == "tile_broken":
            progress_reward += 0.1
    total = mode_reward + progress_reward + exploration_reward + shaping + preference_reward - step_cost
    return total
