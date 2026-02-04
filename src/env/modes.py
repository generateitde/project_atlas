from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.events import Event
from src.core.rng import RNG
from src.core.types import TileType


class Mode:
    name: str = "Base"

    def reset(self, world, rng: RNG) -> dict[str, Any]:
        return {}

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        return 0.0, [], False, {}


@dataclass
class FreeExplore(Mode):
    name: str = "FreeExplore"


@dataclass
class ExitGame(Mode):
    name: str = "ExitGame"

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        if world.tile_at(world.atlas.pos) == TileType.GOAL:
            reward += 10.0
            done = True
        return reward, [], done, {}


@dataclass
class HideAndSeek(Mode):
    name: str = "HideAndSeek"
    hide_target: tuple[int, int] | None = None

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        if self.hide_target and world.atlas.pos.as_int() == self.hide_target:
            reward += 5.0
            done = True
        return reward, [], done, {}


@dataclass
class CaptureTheFlag(Mode):
    name: str = "CaptureTheFlag"

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        if world.atlas_has_flag:
            reward += 0.1
        return reward, [], False, {}


@dataclass
class TrainingArena(Mode):
    name: str = "TrainingArena"


MODE_REGISTRY = {
    "FreeExplore": FreeExplore,
    "ExitGame": ExitGame,
    "HideAndSeek": HideAndSeek,
    "CaptureTheFlag": CaptureTheFlag,
    "TrainingArena": TrainingArena,
}


def create_mode(name: str, params: dict[str, Any] | None = None) -> Mode:
    params = params or {}
    mode_cls = MODE_REGISTRY.get(name, FreeExplore)
    return mode_cls(**params)
