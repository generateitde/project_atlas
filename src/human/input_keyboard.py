from __future__ import annotations

import pygame

from src.config import ControlsConfig
from src.env.tools import break_tile, inspect, jump, move


class KeyboardController:
    def __init__(self, world, target_id: str = "human") -> None:
        self.world = world
        self.target_id = target_id

    def set_target(self, target_id: str) -> None:
        self.target_id = target_id

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        actor = self.world.get_actor(self.target_id)
        if event.key in (pygame.K_w, pygame.K_UP):
            move(self.world, actor.entity_id, "N")
        elif event.key in (pygame.K_s, pygame.K_DOWN):
            move(self.world, actor.entity_id, "S")
        elif event.key in (pygame.K_a, pygame.K_LEFT):
            move(self.world, actor.entity_id, "W")
        elif event.key in (pygame.K_d, pygame.K_RIGHT):
            move(self.world, actor.entity_id, "E")
        elif event.key == pygame.K_SPACE:
            jump(self.world, actor.entity_id)
        elif event.key == pygame.K_b:
            dx = 1 if actor.facing.value == "E" else -1
            break_tile(self.world, actor.entity_id, int(actor.pos.x + dx), int(actor.pos.y))
        elif event.key == pygame.K_i:
            dx = 1 if actor.facing.value == "E" else -1
            inspect(self.world, actor.entity_id, int(actor.pos.x + dx), int(actor.pos.y))
