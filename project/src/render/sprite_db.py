from __future__ import annotations

import pygame

from src.core.types import TileType


TILE_COLORS = {
    TileType.EMPTY: (20, 20, 30),
    TileType.WALL: (100, 100, 120),
    TileType.BREAKABLE_WALL: (140, 100, 60),
    TileType.WATER: (30, 80, 150),
    TileType.LAVA: (200, 60, 30),
    TileType.DOOR_CLOSED: (90, 60, 30),
    TileType.DOOR_OPEN: (140, 120, 90),
    TileType.LADDER: (160, 120, 80),
    TileType.PLATFORM: (120, 80, 40),
    TileType.GOAL: (80, 200, 80),
    TileType.FLAG: (200, 200, 60),
    TileType.GATE: (150, 50, 150),
}


def draw_tile(screen: pygame.Surface, tile: TileType, rect: pygame.Rect) -> None:
    color = TILE_COLORS.get(tile, (255, 0, 255))
    pygame.draw.rect(screen, color, rect)
