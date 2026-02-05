from __future__ import annotations

from src.core.events import Event

RewardBreakdown = dict[str, float]


def compute_reward(
    mode_reward: float,
    events: list[Event],
    step_cost: float = 0.01,
    preference_reward: float = 0.0,
) -> tuple[float, RewardBreakdown]:
    progress_reward = 0.0
    exploration_reward = 0.0
    shaping = 0.0
    for event in events:
        if event.type == "tile_broken":
            progress_reward += 0.1
    total = mode_reward + progress_reward + exploration_reward + shaping + preference_reward - step_cost
    breakdown = {
        "mode": mode_reward,
        "progress": progress_reward,
        "explore": exploration_reward,
        "preference": preference_reward,
        "shaping": shaping,
        "step_cost": step_cost,
        "total": total,
    }
    return total, breakdown
