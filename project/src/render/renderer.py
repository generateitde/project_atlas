from __future__ import annotations

from typing import Tuple

import pygame

from src.core.types import Character, TileType
from src.env.grid_env import WorldState
from src.render.anchors import ANCHORS
from src.render.sprite_db import draw_tile
from src.render.ui_overlays import OverlayManager


class Renderer:
    def __init__(self, tile_size: int, world_size: Tuple[int, int]) -> None:
        self.tile_size = tile_size
        self.width, self.height = world_size
        self.overlay = OverlayManager()
        self.font = pygame.font.SysFont("arial", 14)

    def draw(self, screen: pygame.Surface, world: WorldState) -> None:
        screen.fill((10, 10, 20))
        # TODO: Add animation frames and sprite swaps for movement/combat.
        for y, row in enumerate(world.tiles):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                draw_tile(screen, tile, rect)
        self._draw_character(screen, world.human, (50, 180, 220))
        self._draw_character(screen, world.atlas, (220, 120, 50))
        for item in world.items.values():
            rect = pygame.Rect(item.pos[0] * self.tile_size, item.pos[1] * self.tile_size, self.tile_size, self.tile_size)
            pygame.draw.rect(screen, (200, 200, 200), rect)
        for enemy in world.enemies.values():
            rect = pygame.Rect(enemy.pos[0] * self.tile_size, enemy.pos[1] * self.tile_size, self.tile_size, self.tile_size)
            pygame.draw.rect(screen, (200, 60, 60), rect)
        self.overlay.draw(screen, self.font)

    def _draw_character(self, screen: pygame.Surface, character: Character, color) -> None:
        rect = pygame.Rect(
            character.pos[0] * self.tile_size,
            character.pos[1] * self.tile_size,
            self.tile_size,
            self.tile_size,
        )
        pygame.draw.rect(screen, color, rect)
        if character.hand_item_id:
            anchor = ANCHORS.get("atlas", {}).get(character.facing, {}).get("hand_r", (0, 0))
            item_rect = pygame.Rect(
                rect.x + anchor[0],
                rect.y + anchor[1],
                self.tile_size // 3,
                self.tile_size // 3,
            )
            pygame.draw.rect(screen, (250, 250, 250), item_rect)
