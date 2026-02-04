from __future__ import annotations

from typing import Dict

from src.core.types import TileProps, TileType


TILE_PROPS: Dict[TileType, TileProps] = {
    TileType.EMPTY: TileProps(True, False, False, False, False, False),
    TileType.WALL: TileProps(False, True, False, False, False, False),
    TileType.BREAKABLE_WALL: TileProps(False, True, True, False, False, False),
    TileType.WATER: TileProps(True, False, False, False, False, False),
    TileType.LAVA: TileProps(True, False, False, True, False, False),
    TileType.DOOR_CLOSED: TileProps(False, True, False, False, False, False),
    TileType.DOOR_OPEN: TileProps(True, False, False, False, False, False),
    TileType.LADDER: TileProps(True, False, False, False, True, False),
    TileType.PLATFORM: TileProps(True, True, False, False, False, True),
    TileType.GOAL: TileProps(True, False, False, False, False, False),
    TileType.FLAG: TileProps(True, False, False, False, False, False),
    TileType.GATE: TileProps(False, True, False, False, False, False, gate_req_level=2),
}


def is_passable(tile: TileType, level: int) -> bool:
    props = TILE_PROPS[tile]
    if tile == TileType.GATE and props.gate_req_level is not None:
        return level >= props.gate_req_level
    return props.passable
