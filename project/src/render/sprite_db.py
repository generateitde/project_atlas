from __future__ import annotations

import pygame

from src.core.types import TileType


TILE_COLORS = {
    TileType.EMPTY: (20, 30, 45),
    TileType.WALL: (110, 110, 120),
    TileType.BREAKABLE_WALL: (140, 100, 70),
    TileType.WATER: (40, 100, 190),
    TileType.LAVA: (200, 70, 30),
    TileType.DOOR_CLOSED: (110, 80, 50),
    TileType.DOOR_OPEN: (160, 130, 90),
    TileType.LADDER: (170, 130, 90),
    TileType.PLATFORM: (125, 95, 55),
    TileType.GOAL: (80, 200, 80),
    TileType.FLAG: (210, 210, 70),
    TileType.GATE: (150, 60, 160),
}


def draw_tile(screen: pygame.Surface, tile: TileType, rect: pygame.Rect, grass_top: bool = False) -> None:
    if tile == TileType.EMPTY:
        return
    color = TILE_COLORS.get(tile, (255, 0, 255))
    border = (max(color[0] - 30, 0), max(color[1] - 30, 0), max(color[2] - 30, 0))
    highlight = (min(color[0] + 25, 255), min(color[1] + 25, 255), min(color[2] + 25, 255))
    pygame.draw.rect(screen, color, rect)
    pygame.draw.line(screen, highlight, rect.topleft, (rect.right - 1, rect.top))
    pygame.draw.line(screen, highlight, rect.topleft, (rect.left, rect.bottom - 1))
    pygame.draw.line(screen, border, (rect.left, rect.bottom - 1), (rect.right - 1, rect.bottom - 1))
    pygame.draw.line(screen, border, (rect.right - 1, rect.top), (rect.right - 1, rect.bottom - 1))
    pygame.draw.rect(screen, border, rect, 1)

    if grass_top and tile in {TileType.WALL, TileType.BREAKABLE_WALL, TileType.PLATFORM}:
        grass_height = max(3, rect.height // 6)
        grass_rect = pygame.Rect(rect.x, rect.y, rect.width, grass_height)
        pygame.draw.rect(screen, (80, 180, 70), grass_rect)
        pygame.draw.line(screen, (50, 140, 50), grass_rect.bottomleft, grass_rect.bottomright)
