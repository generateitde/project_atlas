from __future__ import annotations

from src.core.types import TILE_PROPS, Character, Vec2


GRAVITY = 0.2
MAX_FALL_SPEED = 3.0
JUMP_COOLDOWN_TICKS = 6
BASE_EXP_TO_LEVEL = 10
EXP_LEVEL_SCALING = 5
DEFAULT_TRANSFORM_DURATION = 120



def apply_gravity(character: Character, can_stand: bool) -> None:
    if character.can_fly:
        return
    if not can_stand:
        character.vel.y = min(character.vel.y + GRAVITY, MAX_FALL_SPEED)
    else:
        character.vel.y = 0


def move_character(character: Character, dx: float, dy: float) -> None:
    character.pos = Vec2(character.pos.x + dx, character.pos.y + dy)


def is_passable(tile) -> bool:
    return TILE_PROPS[tile].passable


def can_pass_tile(actor: Character, tile) -> bool:
    props = TILE_PROPS[tile]
    if props.passable:
        return True
    if props.gate_req_level > 0 and actor.level >= props.gate_req_level:
        return True
    return False


def exp_to_next_level(level: int) -> int:
    return BASE_EXP_TO_LEVEL + max(level - 1, 0) * EXP_LEVEL_SCALING


def grant_exp(actor: Character, amount: int) -> dict[str, int]:
    gained = max(0, int(amount))
    actor.exp += gained
    levels_gained = 0
    while actor.exp >= exp_to_next_level(actor.level):
        actor.exp -= exp_to_next_level(actor.level)
        actor.level += 1
        levels_gained += 1
    return {"exp_gained": gained, "levels_gained": levels_gained, "level": actor.level, "exp": actor.exp}


def objective_exp_from(mode_reward: float, events: list) -> int:
    event_exp = {
        "enemy_defeated": 10,
        "ctf_scored": 12,
        "hide_target_found": 8,
    }
    total = sum(event_exp.get(getattr(event, "type", ""), 0) for event in events)
    if mode_reward >= 5.0:
        total += 6
    return total


def activate_transform(
    actor: Character,
    *,
    transform_id: str = "berserk",
    duration_ticks: int = DEFAULT_TRANSFORM_DURATION,
    atk_multiplier: float = 1.5,
    defense_multiplier: float = 1.25,
    speed_multiplier: float = 1.2,
    jump_multiplier: float = 1.2,
) -> dict[str, float | int | str]:
    duration = max(1, int(duration_ticks))
    if actor.transform_stats_backup is None:
        actor.transform_stats_backup = {
            "atk": float(actor.atk),
            "defense": float(actor.defense),
            "speed": float(actor.speed),
            "jump_power": float(actor.jump_power),
        }
    actor.transform_state = transform_id
    actor.transform_timer = duration
    backup = actor.transform_stats_backup
    actor.atk = max(1, int(round(backup["atk"] * atk_multiplier)))
    actor.defense = max(0, int(round(backup["defense"] * defense_multiplier)))
    actor.speed = max(0.1, float(backup["speed"] * speed_multiplier))
    actor.jump_power = max(1.0, float(backup["jump_power"] * jump_multiplier))
    return {
        "transform": transform_id,
        "duration": duration,
        "atk": actor.atk,
        "defense": actor.defense,
        "speed": actor.speed,
        "jump_power": actor.jump_power,
    }


def tick_transform(actor: Character) -> bool:
    if not actor.transform_state:
        return False
    if actor.transform_timer > 0:
        actor.transform_timer -= 1
    if actor.transform_timer > 0:
        return False
    backup = actor.transform_stats_backup or {}
    actor.atk = int(round(float(backup.get("atk", actor.atk))))
    actor.defense = int(round(float(backup.get("defense", actor.defense))))
    actor.speed = float(backup.get("speed", actor.speed))
    actor.jump_power = float(backup.get("jump_power", actor.jump_power))
    actor.transform_state = None
    actor.transform_timer = 0
    actor.transform_stats_backup = None
    return True


def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def find_item_by_id(items: list, item_id: str):
    for item in items:
        if getattr(item, "item_id", None) == item_id:
            return item
    return None


def find_tiles(tiles, tile_type) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            if tiles[y, x] == tile_type:
                positions.append((x, y))
    return positions
