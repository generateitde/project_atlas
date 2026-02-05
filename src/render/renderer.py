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
        self.last_chat_bottom = 0

    def render(
        self,
        surface: pygame.Surface,
        world,
        mode_name: str,
        messages: list[tuple[str, str]],
        goal_text: str = "",
        ai_mode: str = "explore",
        waiting_for_response: bool = False,
        debug_hud: bool = False,
        last_action: str = "None",
        reward_terms: dict[str, float] | None = None,
        subgoal_text: str = "",
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
        self.last_chat_bottom = chat_y

        if debug_hud:
            debug_x = 6
            debug_y = 6
            debug_width = self.width * self.tile_size - 12
            self.ui.draw_text(surface, "DEBUG HUD (F3)", (debug_x, debug_y), (255, 220, 120))
            debug_y += self.font.get_linesize()
            debug_y += self.ui.draw_wrapped_text(
                surface,
                f"Mode: {mode_name}",
                (debug_x, debug_y),
                debug_width,
                (220, 220, 220),
            )
            subgoal_label = subgoal_text if subgoal_text else "-"
            debug_y += self.ui.draw_wrapped_text(
                surface,
                f"Subgoal: {subgoal_label}",
                (debug_x, debug_y),
                debug_width,
                (200, 200, 200),
            )
            debug_y += self.ui.draw_wrapped_text(
                surface,
                f"Last Action: {last_action}",
                (debug_x, debug_y),
                debug_width,
                (200, 200, 200),
            )
            terms = reward_terms or {}
            if terms:
                terms_line = (
                    "Reward: "
                    f"mode={terms.get('mode', 0.0):+.2f} "
                    f"progress={terms.get('progress', 0.0):+.2f} "
                    f"explore={terms.get('explore', 0.0):+.2f} "
                    f"preference={terms.get('preference', 0.0):+.2f} "
                    f"shaping={terms.get('shaping', 0.0):+.2f} "
                    f"step_cost={terms.get('step_cost', 0.0):+.2f} "
                    f"total={terms.get('total', 0.0):+.2f}"
                )
            else:
                terms_line = "Reward: (no data yet)"
            self.ui.draw_wrapped_text(surface, terms_line, (debug_x, debug_y), debug_width, (160, 220, 160))

        # TODO: Add animations for movement and combat.
