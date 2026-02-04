from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame

from src.core.types import TileType


@dataclass
class SpriteDB:
    tile_size: int

    def tile_color(self, tile: TileType) -> Tuple[int, int, int]:
        mapping = {
            TileType.EMPTY: (20, 20, 30),
            TileType.WALL: (100, 100, 100),
            TileType.BREAKABLE_WALL: (140, 90, 60),
            TileType.WATER: (40, 80, 200),
            TileType.LAVA: (200, 60, 30),
            TileType.DOOR_CLOSED: (120, 80, 40),
            TileType.DOOR_OPEN: (180, 140, 80),
            TileType.LADDER: (120, 90, 50),
            TileType.PLATFORM: (90, 90, 90),
            TileType.GOAL: (50, 180, 50),
            TileType.FLAG: (180, 50, 50),
            TileType.GATE: (80, 50, 120),
        }
        return mapping.get(tile, (255, 0, 255))

    def draw_tile(self, surface: pygame.Surface, tile: TileType, x: int, y: int) -> None:
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
        pygame.draw.rect(surface, self.tile_color(tile), rect)

    def draw_character(self, surface: pygame.Surface, color: Tuple[int, int, int], x: int, y: int) -> None:
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
        pygame.draw.rect(surface, color, rect)
