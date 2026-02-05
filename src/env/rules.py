from __future__ import annotations

from src.core.types import TILE_PROPS, Character, Vec2


GRAVITY = 0.2
MAX_FALL_SPEED = 3.0


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


def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def find_item_by_id(items: list, item_id: str):
    for item in items:
        if getattr(item, "item_id", None) == item_id:
            return item
    return None
