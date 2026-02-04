from __future__ import annotations

import pygame

from src.env.tools import break_tile, inspect, jump, move


class KeyboardController:
    def __init__(self, world) -> None:
        self.world = world

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        human = self.world.human
        if event.key in (pygame.K_a, pygame.K_LEFT):
            move(self.world, human.entity_id, "W")
        elif event.key in (pygame.K_d, pygame.K_RIGHT):
            move(self.world, human.entity_id, "E")
        elif event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
            jump(self.world, human.entity_id)
        elif event.key == pygame.K_b:
            dx = 1 if human.facing.value == "E" else -1
            break_tile(self.world, human.entity_id, int(human.pos.x + dx), int(human.pos.y))
        elif event.key == pygame.K_i:
            dx = 1 if human.facing.value == "E" else -1
            inspect(self.world, human.entity_id, int(human.pos.x + dx), int(human.pos.y))
