from __future__ import annotations

from typing import Dict

from src.core.types import Facing, TileType


TILE_TO_INT: Dict[TileType, int] = {tile: idx for idx, tile in enumerate(TileType)}
FACING_TO_INT = {Facing.N: 0, Facing.E: 1, Facing.S: 2, Facing.W: 3}


def encode_tile(tile: TileType) -> int:
    return TILE_TO_INT[tile]
