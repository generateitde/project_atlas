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

    def render(
        self,
        surface: pygame.Surface,
        world,
        mode_name: str,
        messages: list[tuple[str, str]],
        goal_text: str = "",
        ai_mode: str = "explore",
        waiting_for_response: bool = False,
    ) -> None:
        surface.fill((10, 10, 20))
        max_text_width = surface.get_width() - 8
        for y in range(world.tiles.shape[0]):
            for x in range(world.tiles.shape[1]):
                tile = world.tiles[y, x]
                self.sprite_db.draw_tile(surface, tile, x * self.tile_size, y * self.tile_size)
        atlas = world.atlas
        human = world.human
        self.sprite_db.draw_character(surface, (50, 200, 255), int(atlas.pos.x) * self.tile_size, int(atlas.pos.y) * self.tile_size)
        self.sprite_db.draw_character(surface, (200, 200, 50), int(human.pos.x) * self.tile_size, int(human.pos.y) * self.tile_size)

        hud_y = self.height * self.tile_size + 4
        offset = self.ui.draw_wrapped_text(surface, f"Mode: {mode_name}", (4, hud_y), max_text_width)
        offset += self.ui.draw_wrapped_text(
            surface,
            f"Atlas HP: {atlas.hp} Lvl: {atlas.level} EXP: {atlas.exp}",
            (4, hud_y + offset),
            max_text_width,
        )
        if goal_text:
            offset += self.ui.draw_wrapped_text(
                surface,
                f"Goal: {goal_text}",
                (4, hud_y + offset),
                max_text_width,
                (180, 200, 120),
            )
        mode_label = "AI Mode: Query (waiting)" if ai_mode == "query" and waiting_for_response else f"AI Mode: {ai_mode.title()}"
        offset += self.ui.draw_wrapped_text(
            surface,
            mode_label,
            (4, hud_y + offset),
            max_text_width,
            (160, 160, 220),
        )
        chat_y = hud_y + offset + 4
        for idx, (speaker, msg) in enumerate(messages[-3:]):
            display = msg if msg.startswith("Atlas") else f"{speaker}: {msg}"
            line_offset = self.ui.draw_wrapped_text(
                surface,
                display,
                (4, chat_y),
                max_text_width,
            )
            chat_y += line_offset

        input_y = chat_y + min(len(messages[-3:]), 3) * 18 + 6
        if console_active:
            self.ui.draw_text(surface, "> " + console_buffer, (4, input_y))
            if console_message:
                self.ui.draw_text(surface, console_message, (4, input_y + 18))
        elif chat_active:
            self.ui.draw_text(surface, "Chat: " + chat_buffer, (4, input_y))

        # TODO: Add animations for movement and combat.
