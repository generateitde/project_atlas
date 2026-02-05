from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.core.events import Event
from src.core.rng import RNG
from src.core.types import TileType


class ModeState(BaseModel):
    name: str
    objective: str
    status: str = ""
    done: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class FreeExploreState(ModeState):
    pass


class ExitGameState(ModeState):
    goal_reached: bool = False


class HideAndSeekState(ModeState):
    hide_target: tuple[int, int] | None = None


class CaptureTheFlagState(ModeState):
    atlas_has_flag: bool = False


class TrainingArenaState(ModeState):
    pass


class Mode:
    name: str = "Base"
    state: ModeState | None = None

    def reset(self, world, rng: RNG) -> ModeState:
        self.state = ModeState(name=self.name, objective="")
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        return 0.0, [], self.done(), self.info()

    def done(self) -> bool:
        return bool(self.state.done) if self.state else False

    def info(self) -> dict[str, Any]:
        if not self.state:
            return {"name": self.name}
        return self.state.model_dump()


@dataclass
class FreeExplore(Mode):
    name: str = "FreeExplore"

    def reset(self, world, rng: RNG) -> ModeState:
        self.state = FreeExploreState(name=self.name, objective="Explore the world freely.")
        return self.state


@dataclass
class ExitGame(Mode):
    name: str = "ExitGame"

    def reset(self, world, rng: RNG) -> ModeState:
        self.state = ExitGameState(name=self.name, objective="Reach the exit tile.", status="Searching for the exit.")
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        if world.tile_at(world.atlas.pos) == TileType.GOAL:
            reward += 10.0
            done = True
        if self.state and isinstance(self.state, ExitGameState):
            self.state.goal_reached = done
            self.state.done = done
            self.state.status = "Exit reached!" if done else "Searching for the exit."
        return reward, [], done, self.info()


@dataclass
class HideAndSeek(Mode):
    name: str = "HideAndSeek"
    hide_target: tuple[int, int] | None = None

    def reset(self, world, rng: RNG) -> ModeState:
        status = "Hide target set." if self.hide_target else "No hide target set."
        self.state = HideAndSeekState(
            name=self.name,
            objective="Find the hide target.",
            status=status,
            hide_target=self.hide_target,
        )
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        if self.hide_target and world.atlas.pos.as_int() == self.hide_target:
            reward += 5.0
            done = True
        if self.state and isinstance(self.state, HideAndSeekState):
            self.state.done = done
            self.state.status = "Target found!" if done else (self.state.status or "Searching for target.")
        return reward, [], done, self.info()


@dataclass
class CaptureTheFlag(Mode):
    name: str = "CaptureTheFlag"

    def reset(self, world, rng: RNG) -> ModeState:
        self.state = CaptureTheFlagState(
            name=self.name,
            objective="Capture the flag and return it.",
            status="Find the flag.",
            atlas_has_flag=world.atlas_has_flag,
        )
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        if world.atlas_has_flag:
            reward += 0.1
        if self.state and isinstance(self.state, CaptureTheFlagState):
            self.state.atlas_has_flag = world.atlas_has_flag
            self.state.status = "Carrying flag." if world.atlas_has_flag else "Find the flag."
        return reward, [], False, self.info()


@dataclass
class TrainingArena(Mode):
    name: str = "TrainingArena"

    def reset(self, world, rng: RNG) -> ModeState:
        self.state = TrainingArenaState(name=self.name, objective="Practice movement and tools.")
        return self.state


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
