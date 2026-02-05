from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.core.events import Event
from src.core.types import TileType


class Mode:
    name: str

    def reset(self, world, rng) -> Dict[str, Any]:
        return {}

    def step(self, world, events, rng) -> Tuple[float, list[Event], bool, Dict[str, Any]]:
        return 0.0, [], False, {}


@dataclass
class FreeExplore(Mode):
    name: str = "FreeExplore"


@dataclass
class ExitGame(Mode):
    name: str = "ExitGame"

    def step(self, world, events, rng):
        for event in events:
            if event.type == "goal_reached" and event.payload.get("actor_id") == "ai_atlas":
                return 10.0, [event], True, {"success": True}
        return 0.0, [], False, {}


@dataclass
class HideAndSeek(Mode):
    name: str = "HideAndSeek"

    def step(self, world, events, rng):
        for event in events:
            if event.type == "found_human":
                return 5.0, [event], True, {"success": True}
        return -0.01, [], False, {}


@dataclass
class CaptureTheFlag(Mode):
    name: str = "CaptureTheFlag"

    def step(self, world, events, rng):
        reward = 0.0
        done = False
        for event in events:
            if event.type == "flag_carried":
                reward += 0.05
            if event.type == "flag_scored":
                reward += 5.0
                done = True
        return reward, [], done, {}


@dataclass
class TrainingArena(Mode):
    name: str = "TrainingArena"

    def step(self, world, events, rng):
        reward = 0.0
        for event in events:
            if event.type == "enemy_defeated":
                reward += 1.0
        return reward, [], False, {}


MODE_REGISTRY = {
    "FreeExplore": FreeExplore,
    "ExitGame": ExitGame,
    "HideAndSeek": HideAndSeek,
    "CaptureTheFlag": CaptureTheFlag,
    "TrainingArena": TrainingArena,
}
