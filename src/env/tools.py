from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.core.events import Event
from src.core.types import Facing, TileType
from src.env import rules


@dataclass
class ToolResult:
    ok: bool
    delta: dict[str, Any]
    events: list[Event]
    error_code: str | None = None


@dataclass
class SafetyGuardrailsConfig:
    max_tool_calls_per_window: int = 3
    window_ticks: int = 5
    cooldown_by_action: dict[int, int] | None = None
    forbidden_chains: set[tuple[int, int]] | None = None

    def cooldown_for(self, action_id: int) -> int:
        if not self.cooldown_by_action:
            return 0
        return int(self.cooldown_by_action.get(action_id, 0))


@dataclass
class ToolSafetyTracker:
    recent_tool_ticks: list[int]
    cooldown_until: dict[int, int]
    last_tool_action: int | None = None


DEFAULT_TOOL_SAFETY_CONFIG = SafetyGuardrailsConfig(
    cooldown_by_action={12: 2, 13: 6},
    forbidden_chains={(13, 13), (12, 13)},
)

TOOL_ACTION_IDS = frozenset({6, 7, 8, 9, 10, 11, 12, 13})


def tool_safety_precheck(
    action_id: int,
    *,
    tick: int,
    tracker: ToolSafetyTracker,
    config: SafetyGuardrailsConfig = DEFAULT_TOOL_SAFETY_CONFIG,
    strict_safety: bool = False,
) -> ToolResult:
    if action_id not in TOOL_ACTION_IDS:
        return _result(True)

    until_tick = int(tracker.cooldown_until.get(action_id, -1))
    if tick < until_tick:
        return _result(False, error_code="tool_cooldown")

    if tracker.last_tool_action is not None and config.forbidden_chains:
        if (tracker.last_tool_action, action_id) in config.forbidden_chains:
            return _result(False, error_code="forbidden_chain")

    if strict_safety:
        recent = [recent_tick for recent_tick in tracker.recent_tool_ticks if tick - recent_tick < config.window_ticks]
        if len(recent) >= config.max_tool_calls_per_window:
            return _result(False, error_code="rate_limited")
    return _result(True)


def tool_safety_commit(
    action_id: int,
    *,
    tick: int,
    tracker: ToolSafetyTracker,
    config: SafetyGuardrailsConfig = DEFAULT_TOOL_SAFETY_CONFIG,
) -> None:
    if action_id not in TOOL_ACTION_IDS:
        return
    tracker.recent_tool_ticks = [recent_tick for recent_tick in tracker.recent_tool_ticks if tick - recent_tick < config.window_ticks]
    tracker.recent_tool_ticks.append(tick)
    tracker.last_tool_action = action_id
    cooldown = config.cooldown_for(action_id)
    if cooldown > 0:
        tracker.cooldown_until[action_id] = tick + cooldown


def _result(
    ok: bool,
    *,
    error_code: str | None = None,
    delta: dict[str, Any] | None = None,
    events: list[Event] | None = None,
) -> ToolResult:
    return ToolResult(ok, delta or {}, events or [], error_code)


def _dir_to_delta(direction: str) -> tuple[int, int]:
    mapping = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0),
    }
    return mapping.get(direction, (0, 0))


def _adjacent_pos(world, actor_id: str) -> tuple[int, int]:
    actor = world.get_actor(actor_id)
    dx, dy = _dir_to_delta(actor.facing.value)
    return int(actor.pos.x + dx), int(actor.pos.y + dy)


def precheck_move(world, actor_id: str, direction: str) -> ToolResult:
    dx, dy = _dir_to_delta(direction)
    actor = world.get_actor(actor_id)
    if direction in {"N", "S"} and not actor.can_fly:
        return _result(False, error_code="blocked")
    target = (int(actor.pos.x + dx), int(actor.pos.y + dy))
    if not world.in_bounds(target):
        return _result(False, error_code="out_of_bounds")
    if not world.is_passable_for(actor, target):
        return _result(False, error_code="blocked")
    return _result(True)


def move(world, actor_id: str, direction: str) -> ToolResult:
    dx, dy = _dir_to_delta(direction)
    actor = world.get_actor(actor_id)
    actor.facing = Facing(direction)
    precheck = precheck_move(world, actor_id, direction)
    if not precheck.ok:
        return precheck
    actor.pos.x += dx
    actor.pos.y += dy
    return _result(True, delta={"pos": actor.pos.as_int()})


def precheck_jump(world, actor_id: str) -> ToolResult:
    actor = world.get_actor(actor_id)
    if actor.jump_cooldown > 0:
        return _result(False, error_code="cooldown")
    if actor.can_fly:
        return _result(True)
    below = (int(actor.pos.x), int(actor.pos.y + 1))
    if not world.can_stand_on(below) or not actor.grounded:
        return _result(False, error_code="not_grounded")
    return _result(True)


def jump(world, actor_id: str) -> ToolResult:
    precheck = precheck_jump(world, actor_id)
    if not precheck.ok:
        return precheck
    actor = world.get_actor(actor_id)
    actor.jump_remaining = int(actor.jump_power)
    actor.vel.y = -actor.jump_power
    actor.jump_cooldown = rules.JUMP_COOLDOWN_TICKS
    actor.grounded = False
    return _result(True)


def precheck_use(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    actor = world.get_actor(actor_id)
    if world.hand_item is None:
        return _result(False, error_code="no_item")
    if not world.in_bounds(target):
        return _result(False, error_code="out_of_bounds")
    if not rules.is_adjacent(actor.pos.as_int(), target):
        return _result(False, error_code="not_adjacent")
    return _result(True)


def use(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    precheck = precheck_use(world, actor_id, target)
    if not precheck.ok:
        return precheck
    return _result(True)


def precheck_pickup(world, actor_id: str, item_id: str) -> ToolResult:
    if world.hand_item is not None:
        return _result(False, error_code="hand_full")
    item = rules.find_item_by_id(world.items, item_id)
    if item is None:
        return _result(False, error_code="no_item")
    actor = world.get_actor(actor_id)
    if not rules.is_adjacent(actor.pos.as_int(), item.pos.as_int()):
        return _result(False, error_code="not_adjacent")
    return _result(True)


def pickup(world, actor_id: str, item_id: str) -> ToolResult:
    precheck = precheck_pickup(world, actor_id, item_id)
    if not precheck.ok:
        return precheck
    return _result(True)


def precheck_drop(world, actor_id: str, item_id: str) -> ToolResult:
    if world.hand_item is None:
        return _result(False, error_code="no_item")
    if getattr(world.hand_item, "item_id", None) != item_id:
        return _result(False, error_code="not_in_hand")
    return _result(True)


def drop(world, actor_id: str, item_id: str) -> ToolResult:
    precheck = precheck_drop(world, actor_id, item_id)
    if not precheck.ok:
        return precheck
    return _result(True)


def precheck_attack(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    actor = world.get_actor(actor_id)
    if not world.in_bounds(target):
        return _result(False, error_code="out_of_bounds")
    if not rules.is_adjacent(actor.pos.as_int(), target):
        return _result(False, error_code="not_adjacent")
    return _result(True)


def attack(world, actor_id: str, target: tuple[int, int]) -> ToolResult:
    precheck = precheck_attack(world, actor_id, target)
    if not precheck.ok:
        return precheck
    return _result(True)


def precheck_break_tile(world, actor_id: str, x: int, y: int) -> ToolResult:
    actor = world.get_actor(actor_id)
    if not world.in_bounds((x, y)):
        return _result(False, error_code="out_of_bounds")
    if not rules.is_adjacent(actor.pos.as_int(), (x, y)):
        return _result(False, error_code="not_adjacent")
    if world.tiles[y][x] != TileType.BREAKABLE_WALL:
        return _result(False, error_code="not_breakable")
    return _result(True)


def break_tile(world, actor_id: str, x: int, y: int) -> ToolResult:
    precheck = precheck_break_tile(world, actor_id, x, y)
    if not precheck.ok:
        return precheck
    world.tiles[y][x] = TileType.EMPTY
    return _result(True, events=[Event("tile_broken", {"x": x, "y": y})])


def precheck_inspect(world, actor_id: str, x: int, y: int) -> ToolResult:
    actor = world.get_actor(actor_id)
    if not world.in_bounds((x, y)):
        return _result(False, error_code="out_of_bounds")
    if not rules.is_adjacent(actor.pos.as_int(), (x, y)):
        return _result(False, error_code="not_adjacent")
    return _result(True)


def inspect(world, actor_id: str, x: int, y: int) -> ToolResult:
    precheck = precheck_inspect(world, actor_id, x, y)
    if not precheck.ok:
        return precheck
    tile = world.tiles[y][x]
    return _result(True, delta={"tile": tile.value})


def speak(world, text: str) -> ToolResult:
    world.messages.append(("Atlas", text))
    return _result(True, events=[Event("atlas_message", {"text": text})])


def ask_human(world, question_text: str) -> ToolResult:
    world.messages.append(("Atlas (Frage)", question_text))
    world.pending_question = True
    return _result(True, events=[Event("atlas_question", {"text": question_text})])


def precheck_pickup_adjacent(world, actor_id: str) -> ToolResult:
    if world.hand_item is not None:
        return _result(False, error_code="hand_full")
    actor = world.get_actor(actor_id)
    for item in world.items:
        if rules.is_adjacent(actor.pos.as_int(), item.pos.as_int()):
            return _result(True)
    return _result(False, error_code="no_item")


def precheck_drop_hand(world, actor_id: str) -> ToolResult:
    if world.hand_item is None:
        return _result(False, error_code="no_item")
    return _result(True)


def precheck_use_adjacent(world, actor_id: str) -> ToolResult:
    if world.hand_item is None:
        return _result(False, error_code="no_item")
    target = _adjacent_pos(world, actor_id)
    return precheck_use(world, actor_id, target)


def precheck_attack_adjacent(world, actor_id: str) -> ToolResult:
    target = _adjacent_pos(world, actor_id)
    return precheck_attack(world, actor_id, target)


def precheck_break_adjacent(world, actor_id: str) -> ToolResult:
    target = _adjacent_pos(world, actor_id)
    return precheck_break_tile(world, actor_id, target[0], target[1])


def precheck_inspect_adjacent(world, actor_id: str) -> ToolResult:
    target = _adjacent_pos(world, actor_id)
    return precheck_inspect(world, actor_id, target[0], target[1])


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

TOOL_PRECHECKS: dict[str, Callable[..., ToolResult]] = {
    "move": precheck_move,
    "jump": precheck_jump,
    "use": precheck_use,
    "pickup": precheck_pickup,
    "drop": precheck_drop,
    "attack": precheck_attack,
    "break_tile": precheck_break_tile,
    "inspect": precheck_inspect,
    "pickup_adjacent": precheck_pickup_adjacent,
    "drop_hand": precheck_drop_hand,
    "use_adjacent": precheck_use_adjacent,
    "attack_adjacent": precheck_attack_adjacent,
    "break_adjacent": precheck_break_adjacent,
    "inspect_adjacent": precheck_inspect_adjacent,
}
