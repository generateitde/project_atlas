from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.core.events import Event
from src.core.types import Facing, TileType


@dataclass
class ToolResult:
    ok: bool
    obs_delta: dict[str, Any]
    events: list[Event]
    error: str | None = None


def _dir_to_delta(direction: str) -> tuple[int, int]:
    mapping = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0),
    }
    return mapping.get(direction, (0, 0))


def move(world, actor_id: str, direction: str) -> ToolResult:
    dx, dy = _dir_to_delta(direction)
    actor = world.get_actor(actor_id)
    actor.facing = Facing(direction)
    target = (int(actor.pos.x + dx), int(actor.pos.y + dy))
    if not world.in_bounds(target):
        return ToolResult(False, {}, [], "out_of_bounds")
    if world.is_passable(target):
        actor.pos.x += dx
        actor.pos.y += dy
        return ToolResult(True, {"pos": actor.pos.as_int()}, [], None)
    return ToolResult(False, {}, [], "blocked")


def jump(world, actor_id: str) -> ToolResult:
    actor = world.get_actor(actor_id)
    actor.vel.y = -actor.jump_power
    return ToolResult(True, {}, [], None)


def use(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    return ToolResult(True, {}, [], None)


def pickup(world, actor_id: str, item_id: str) -> ToolResult:
    return ToolResult(True, {}, [], None)


def drop(world, actor_id: str, item_id: str) -> ToolResult:
    return ToolResult(True, {}, [], None)


def attack(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    return ToolResult(True, {}, [], None)


def break_tile(world, actor_id: str, x: int, y: int) -> ToolResult:
    if not world.in_bounds((x, y)):
        return ToolResult(False, {}, [], "out_of_bounds")
    if world.tiles[y][x] == TileType.BREAKABLE_WALL:
        world.tiles[y][x] = TileType.EMPTY
        return ToolResult(True, {}, [Event("tile_broken", {"x": x, "y": y})], None)
    return ToolResult(False, {}, [], "not_breakable")


def inspect(world, actor_id: str, x: int, y: int) -> ToolResult:
    if not world.in_bounds((x, y)):
        return ToolResult(False, {}, [], "out_of_bounds")
    tile = world.tiles[y][x]
    return ToolResult(True, {"tile": tile.value}, [], None)


def speak(world, text: str) -> ToolResult:
    world.messages.append(("Atlas", text))
    return ToolResult(True, {}, [Event("atlas_message", {"text": text})], None)


def ask_human(world, question_text: str) -> ToolResult:
    world.messages.append(("Atlas (Frage)", question_text))
    return ToolResult(True, {}, [Event("atlas_question", {"text": question_text})], None)


TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "move": move,
    "jump": jump,
    "use": use,
    "pickup": pickup,
    "drop": drop,
    "attack": attack,
    "break_tile": break_tile,
    "inspect": inspect,
    "speak": speak,
    "ask_human": ask_human,
}
