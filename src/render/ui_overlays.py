from __future__ import annotations

import pygame


class UIOverlays:
    def __init__(self, font: pygame.font.Font):
        self.font = font

    def draw_text(self, surface: pygame.Surface, text: str, pos: tuple[int, int], color=(240, 240, 240)) -> None:
        render = self.font.render(text, True, color)
        surface.blit(render, pos)

    def wrap_text(self, text: str, max_width: int) -> list[str]:
        if not text:
            return [""]
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if self.font.size(word)[0] > max_width:
                if current:
                    lines.append(current)
                    current = ""
                chunk = ""
                for char in word:
                    candidate = f"{chunk}{char}"
                    if self.font.size(candidate)[0] <= max_width:
                        chunk = candidate
                    else:
                        lines.append(chunk)
                        chunk = char
                if chunk:
                    lines.append(chunk)
                continue
            candidate = f"{current} {word}".strip()
            if self.font.size(candidate)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw_wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        pos: tuple[int, int],
        max_width: int,
        color=(240, 240, 240),
    ) -> int:
        lines = self.wrap_text(text, max_width)
        x, y = pos
        line_height = self.font.get_linesize()
        for idx, line in enumerate(lines):
            render = self.font.render(line, True, color)
            surface.blit(render, (x, y + idx * line_height))
        return len(lines) * line_height
