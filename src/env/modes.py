from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.core.events import Event
from src.core.rng import RNG
from src.core.types import TileType
from src.env import rules


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
    key_collected: bool = False
    door_open: bool = False


class HideAndSeekState(ModeState):
    hide_target: tuple[int, int] | None = None


class CaptureTheFlagState(ModeState):
    atlas_has_flag: bool = False
    flag_pos: tuple[int, int] | None = None
    score_zone: tuple[int, int] | None = None
    score: int = 0


class TrainingArenaState(ModeState):
    pass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    preset: str
    mode: str
    mode_params: dict[str, Any]
    min_episodes: int = 5
    min_success_rate: float = 0.6
    min_avg_return: float = 1.0


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
        self.state = ExitGameState(
            name=self.name,
            objective="Collect key, pass door, reach exit tile.",
            status="Find the key.",
        )
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        atlas_tile = world.tile_at(world.atlas.pos)

        if atlas_tile == TileType.FLAG and not world.atlas_has_flag:
            world.atlas_has_flag = True
            reward += 1.0
            for y in range(world.tiles.shape[0]):
                for x in range(world.tiles.shape[1]):
                    if world.tiles[y, x] == TileType.DOOR_CLOSED:
                        world.tiles[y, x] = TileType.DOOR_OPEN

        if world.tile_at(world.atlas.pos) == TileType.GOAL:
            reward += 10.0
            done = True
        if self.state and isinstance(self.state, ExitGameState):
            self.state.key_collected = world.atlas_has_flag
            self.state.door_open = any(
                world.tiles[y, x] == TileType.DOOR_OPEN
                for y in range(world.tiles.shape[0])
                for x in range(world.tiles.shape[1])
            )
            self.state.goal_reached = done
            self.state.done = done
            if done:
                self.state.status = "Exit reached!"
            elif not self.state.key_collected:
                self.state.status = "Find the key."
            elif not self.state.door_open:
                self.state.status = "Open the door."
            else:
                self.state.status = "Go to the exit."
        return reward, [], done, self.info()


@dataclass
class HideAndSeek(Mode):
    name: str = "HideAndSeek"
    hide_target: tuple[int, int] | None = None
    time_limit_steps: int | None = None
    _prev_distance: int | None = None
    _steps_elapsed: int = 0

    def reset(self, world, rng: RNG) -> ModeState:
        self._steps_elapsed = 0
        self._prev_distance = None
        if self.hide_target:
            atlas_x, atlas_y = world.atlas.pos.as_int()
            self._prev_distance = abs(atlas_x - self.hide_target[0]) + abs(atlas_y - self.hide_target[1])
        status = "Hide target set." if self.hide_target else "No hide target set."
        self.state = HideAndSeekState(
            name=self.name,
            objective="Find the hide target.",
            status=status,
            hide_target=self.hide_target,
            details={"time_limit_steps": self.time_limit_steps, "steps_elapsed": self._steps_elapsed},
        )
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        self._steps_elapsed += 1
        mode_events: list[Event] = []

        if self.hide_target:
            atlas_x, atlas_y = world.atlas.pos.as_int()
            curr_distance = abs(atlas_x - self.hide_target[0]) + abs(atlas_y - self.hide_target[1])
            if self._prev_distance is not None and curr_distance != self._prev_distance:
                reward += 0.1 * float(self._prev_distance - curr_distance)
            self._prev_distance = curr_distance

            if curr_distance == 0:
                reward += 5.0
                done = True
                mode_events.append(Event("hide_target_found", {"target": self.hide_target, "steps": self._steps_elapsed}))

        if self.time_limit_steps is not None and self._steps_elapsed >= self.time_limit_steps and not done:
            done = True
            mode_events.append(Event("hide_time_limit", {"steps": self._steps_elapsed}))

        if self.state and isinstance(self.state, HideAndSeekState):
            self.state.done = done
            self.state.details["steps_elapsed"] = self._steps_elapsed
            if self.hide_target:
                self.state.details["distance"] = self._prev_distance
            if done and self.hide_target and world.atlas.pos.as_int() == self.hide_target:
                self.state.status = "Target found!"
            elif done:
                self.state.status = "Time limit reached."
            elif self.hide_target:
                self.state.status = f"Distance to hide target: {self._prev_distance}"
            else:
                self.state.status = "No hide target set."
        return reward, mode_events, done, self.info()


@dataclass
class CaptureTheFlag(Mode):
    name: str = "CaptureTheFlag"

    def reset(self, world, rng: RNG) -> ModeState:
        flag_positions = rules.find_tiles(world.tiles, TileType.FLAG)
        score_zone = (2, world.tiles.shape[0] - 2)
        self.state = CaptureTheFlagState(
            name=self.name,
            objective="Capture the flag and return it to score.",
            status="Find the flag.",
            atlas_has_flag=world.atlas_has_flag,
            flag_pos=flag_positions[0] if flag_positions else None,
            score_zone=score_zone,
            score=0,
        )
        world.atlas_has_flag = False
        return self.state

    def step(self, world, events: list[Event], rng: RNG) -> tuple[float, list[Event], bool, dict[str, Any]]:
        reward = 0.0
        done = False
        mode_events: list[Event] = []
        atlas_pos = world.atlas.pos.as_int()

        if self.state and isinstance(self.state, CaptureTheFlagState):
            if not world.atlas_has_flag and self.state.flag_pos and atlas_pos == self.state.flag_pos:
                world.atlas_has_flag = True
                reward += 1.0
                mode_events.append(Event("ctf_flag_pickup", {"pos": self.state.flag_pos}))

            if world.atlas_has_flag:
                reward += 0.1

            if world.atlas_has_flag and self.state.score_zone and atlas_pos == self.state.score_zone:
                world.atlas_has_flag = False
                self.state.score += 1
                reward += 5.0
                done = True
                mode_events.append(Event("ctf_scored", {"score": self.state.score}))

            self.state.atlas_has_flag = world.atlas_has_flag
            self.state.done = done
            if done:
                self.state.status = "Scored!"
            elif world.atlas_has_flag:
                self.state.status = "Return to score zone."
            else:
                self.state.status = "Find the flag."
        return reward, mode_events, done, self.info()


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


def default_curriculum_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            name="exit_basic",
            preset="dungeon_exit",
            mode="ExitGame",
            mode_params={},
            min_episodes=4,
            min_success_rate=0.5,
            min_avg_return=2.0,
        ),
        CurriculumStage(
            name="exit_hazards",
            preset="floating_islands",
            mode="ExitGame",
            mode_params={},
            min_episodes=5,
            min_success_rate=0.5,
            min_avg_return=2.5,
        ),
        CurriculumStage(
            name="ctf_basic",
            preset="ctf_small",
            mode="CaptureTheFlag",
            mode_params={},
            min_episodes=6,
            min_success_rate=0.45,
            min_avg_return=1.5,
        ),
    ]


def mode_success(mode_name: str, mode_info: dict[str, Any]) -> bool:
    if mode_name == "ExitGame":
        return bool(mode_info.get("goal_reached"))
    if mode_name == "CaptureTheFlag":
        return bool(mode_info.get("score", 0) > 0)
    if mode_name == "HideAndSeek":
        return bool(mode_info.get("done")) and "Target found" in str(mode_info.get("status", ""))
    return bool(mode_info.get("done"))
