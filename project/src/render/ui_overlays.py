from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pygame


@dataclass
class DamageNumber:
    pos: Tuple[int, int]
    value: int
    ttl: float = 1.0


class OverlayManager:
    def __init__(self) -> None:
        self.damage_numbers: List[DamageNumber] = []

    def add_damage(self, pos: Tuple[int, int], value: int) -> None:
        self.damage_numbers.append(DamageNumber(pos, value))

    def update(self, dt: float) -> None:
        for num in self.damage_numbers:
            num.ttl -= dt
        self.damage_numbers = [num for num in self.damage_numbers if num.ttl > 0]

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        for num in self.damage_numbers:
            text = font.render(f"-{num.value}", True, (220, 80, 80))
            screen.blit(text, num.pos)
