from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


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


class Facing(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


@dataclass
class TileProps:
    passable: bool
    solid: bool
    breakable: bool
    hazardous: bool
    climbable: bool
    one_way_platform: bool
    gate_req_level: Optional[int] = None


@dataclass
class EntityStats:
    hp: int = 10
    atk: int = 2
    defense: int = 1
    speed: float = 1.0
    jump_power: float = 4.0
    can_fly: bool = False
    fly_timer: int = 0
    level: int = 1
    exp: int = 0
    transform_state: str = "base"


@dataclass
class Character:
    entity_id: str
    display_name: str
    pos: Tuple[int, int]
    vel: Tuple[float, float] = (0.0, 0.0)
    facing: Facing = Facing.S
    stats: EntityStats = field(default_factory=EntityStats)
    hand_item_id: Optional[str] = None


@dataclass
class ItemProps:
    pickupable: bool = True
    weapon_like: bool = False
    damage: int = 0
    break_power: int = 0
    key_id: Optional[str] = None
    heal: int = 0
    jump_boost: int = 0
    fly_grant: Optional[Dict[str, int | str]] = None
    transform_unlock: Optional[Dict[str, int | float]] = None


@dataclass
class Item:
    item_id: str
    pos: Tuple[int, int]
    item_type: str
    props: ItemProps


@dataclass
class Enemy:
    enemy_id: str
    pos: Tuple[int, int]
    hp: int
    atk: int
    exp_value: int
    aggro_radius: int = 3
