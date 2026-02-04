from __future__ import annotations

import pygame


class UIOverlays:
    def __init__(self, font: pygame.font.Font):
        self.font = font

    def draw_text(self, surface: pygame.Surface, text: str, pos: tuple[int, int], color=(240, 240, 240)) -> None:
        render = self.font.render(text, True, color)
        surface.blit(render, pos)
