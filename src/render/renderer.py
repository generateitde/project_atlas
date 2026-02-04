from __future__ import annotations

import pygame
from src.render.sprite_db import SpriteDB
from src.render.ui_overlays import UIOverlays


class Renderer:
    def __init__(self, tile_size: int, width: int, height: int):
        self.tile_size = tile_size
        self.width = width
        self.height = height
        self.sprite_db = SpriteDB(tile_size)
        self.font = pygame.font.SysFont("Consolas", 16)
        self.ui = UIOverlays(self.font)

    def render(self, surface: pygame.Surface, world, mode_name: str, messages: list[tuple[str, str]]) -> None:
        surface.fill((10, 10, 20))
        for y in range(world.tiles.shape[0]):
            for x in range(world.tiles.shape[1]):
                tile = world.tiles[y, x]
                self.sprite_db.draw_tile(surface, tile, x * self.tile_size, y * self.tile_size)
        atlas = world.atlas
        human = world.human
        self.sprite_db.draw_character(surface, (50, 200, 255), int(atlas.pos.x) * self.tile_size, int(atlas.pos.y) * self.tile_size)
        self.sprite_db.draw_character(surface, (200, 200, 50), int(human.pos.x) * self.tile_size, int(human.pos.y) * self.tile_size)

        hud_y = self.height * self.tile_size + 4
        self.ui.draw_text(surface, f"Mode: {mode_name}", (4, hud_y))
        self.ui.draw_text(surface, f"Atlas HP: {atlas.hp} Lvl: {atlas.level} EXP: {atlas.exp}", (4, hud_y + 18))
        controls_y = hud_y + 36
        self.ui.draw_text(surface, "Console: ` oder F1 | Chat: Enter", (4, controls_y))
        chat_y = controls_y + 18
        if world.pending_question:
            self.ui.draw_text(surface, "Atlas wartet auf Antwort (Enter zum Chat).", (4, chat_y))
            chat_y += 18
        for idx, (speaker, msg) in enumerate(messages[-3:]):
            display = msg if msg.startswith("Atlas") else f"{speaker}: {msg}"
            self.ui.draw_text(surface, display, (4, chat_y + idx * 18))

        # TODO: Add animations for movement and combat.
