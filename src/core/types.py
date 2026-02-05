from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Facing(str, Enum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"


class TileType(str, Enum):
    EMPTY = "EMPTY"
    WALL = "WALL"
    BREAKABLE_WALL = "BREAKABLE_WALL"
    WATER = "WATER"
    LAVA = "LAVA"
    DOOR_CLOSED = "DOOR_CLOSED"
    DOOR_OPEN = "DOOR_OPEN"
    LADDER = "LADDER"
    PLATFORM = "PLATFORM"
    GOAL = "GOAL"
    FLAG = "FLAG"
    GATE = "GATE"


@dataclass
class TileProps:
    passable: bool
    solid: bool
    breakable: bool = False
    hazardous: bool = False
    climbable: bool = False
    one_way_platform: bool = False
    gate_req_level: int = 0


TILE_PROPS: dict[TileType, TileProps] = {
    TileType.EMPTY: TileProps(passable=True, solid=False),
    TileType.WALL: TileProps(passable=False, solid=True),
    TileType.BREAKABLE_WALL: TileProps(passable=False, solid=True, breakable=True),
    TileType.WATER: TileProps(passable=True, solid=False, hazardous=False),
    TileType.LAVA: TileProps(passable=True, solid=False, hazardous=True),
    TileType.DOOR_CLOSED: TileProps(passable=False, solid=True),
    TileType.DOOR_OPEN: TileProps(passable=True, solid=False),
    TileType.LADDER: TileProps(passable=True, solid=False, climbable=True),
    TileType.PLATFORM: TileProps(passable=True, solid=False, one_way_platform=True),
    TileType.GOAL: TileProps(passable=True, solid=False),
    TileType.FLAG: TileProps(passable=True, solid=False),
    TileType.GATE: TileProps(passable=False, solid=True, gate_req_level=1),
}


@dataclass
class Vec2:
    x: float
    y: float

    def as_int(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass
class Character:
    entity_id: str
    display_name: str
    pos: Vec2
    vel: Vec2 = field(default_factory=lambda: Vec2(0, 0))
    facing: Facing = Facing.EAST
    hp: int = 10
    atk: int = 1
    defense: int = 0
    speed: float = 1.0
    jump_power: float = 2.0
    can_fly: bool = False
    fly_timer: int = 0
    level: int = 1
    exp: int = 0
    transform_state: Optional[str] = None
    jump_remaining: int = 0
    jump_cooldown: int = 0
    grounded: bool = False


@dataclass
class ItemProps:
    pickupable: bool = True
    weapon_like: bool = False
    damage: int = 0
    break_power: int = 0
    key_id: Optional[str] = None
    heal: int = 0
    jump_boost: int = 0
    fly_grant: Optional[dict] = None
    transform_unlock: Optional[dict] = None


@dataclass
class Item:
    item_id: str
    pos: Vec2
    type: str
    props: ItemProps = field(default_factory=ItemProps)


@dataclass
class Enemy:
    enemy_id: str
    pos: Vec2
    hp: int
    atk: int
    exp_value: int
    aggro_radius: int
