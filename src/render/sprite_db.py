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

    def _scale_rect(self, x: int, y: int, w: int, h: int) -> pygame.Rect:
        sx = x * self.tile_size // 32
        sy = y * self.tile_size // 32
        ex = (x + w) * self.tile_size // 32
        ey = (y + h) * self.tile_size // 32
        return pygame.Rect(sx, sy, max(1, ex - sx), max(1, ey - sy))

    @staticmethod
    def _shade(color: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, c + delta)) for c in color)

    def draw_character(self, surface: pygame.Surface, color: Tuple[int, int, int], x: int, y: int) -> None:
        """Draw a stylized 32x32 humanoid silhouette instead of a solid block.

        The full visual footprint always stays within one 32x32 sprite slot
        (scaled with ``tile_size`` for rendering).
        """

        base = self._shade(color, -10)
        shadow = self._shade(color, -45)
        highlight = self._shade(color, 40)

        sprite = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)

        # Legs and feet.
        pygame.draw.rect(sprite, base, self._scale_rect(8, 21, 6, 9))
        pygame.draw.rect(sprite, base, self._scale_rect(18, 21, 6, 9))
        pygame.draw.rect(sprite, shadow, self._scale_rect(7, 29, 8, 2))
        pygame.draw.rect(sprite, shadow, self._scale_rect(17, 29, 8, 2))

        # Torso and shoulders.
        pygame.draw.rect(sprite, base, self._scale_rect(9, 11, 14, 12))
        pygame.draw.rect(sprite, shadow, self._scale_rect(8, 12, 2, 10))
        pygame.draw.rect(sprite, highlight, self._scale_rect(11, 13, 8, 4))

        # Arms in a "hands-on-hips" pose.
        pygame.draw.rect(sprite, shadow, self._scale_rect(6, 14, 3, 10))
        pygame.draw.rect(sprite, shadow, self._scale_rect(23, 14, 3, 10))

        # Head and hair spikes.
        pygame.draw.rect(sprite, highlight, self._scale_rect(11, 4, 10, 8))
        pygame.draw.rect(sprite, shadow, self._scale_rect(10, 3, 12, 2))
        pygame.draw.rect(sprite, shadow, self._scale_rect(8, 1, 3, 3))
        pygame.draw.rect(sprite, shadow, self._scale_rect(22, 1, 3, 3))

        # Face detail.
        pygame.draw.rect(sprite, (22, 22, 22), self._scale_rect(14, 7, 4, 1))

        surface.blit(sprite, (x, y))
