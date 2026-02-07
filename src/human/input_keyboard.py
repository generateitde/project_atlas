from __future__ import annotations

import pygame

from src.config import ControlsConfig
from src.env.tools import break_tile, inspect, jump, move


def _key_codes(names: list[str]) -> set[int]:
    codes: set[int] = set()
    for name in names:
        try:
            codes.add(pygame.key.key_code(name))
        except (ValueError, KeyError):
            continue
    return codes


class KeyboardController:
    def __init__(self, world, controls: ControlsConfig, target_id: str = "human") -> None:
        self.world = world
        self.target_id = target_id
        self._left_keys = _key_codes(controls.move_left)
        self._right_keys = _key_codes(controls.move_right)
        self._jump_keys = _key_codes(controls.jump)
        self._break_keys = _key_codes(controls.break_tile)
        self._inspect_keys = _key_codes(controls.inspect)

    def set_target(self, target_id: str) -> None:
        self.target_id = target_id

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        actor = self.world.get_actor(self.target_id)
        if event.key in self._left_keys:
            move(self.world, actor.entity_id, "W")
        elif event.key in self._right_keys:
            move(self.world, actor.entity_id, "E")
        elif event.key in self._jump_keys:
            jump(self.world, actor.entity_id)
        elif event.key in self._break_keys:
            dx = 1 if actor.facing.value == "E" else -1
            break_tile(self.world, actor.entity_id, int(actor.pos.x + dx), int(actor.pos.y))
        elif event.key in self._inspect_keys:
            dx = 1 if actor.facing.value == "E" else -1
            inspect(self.world, actor.entity_id, int(actor.pos.x + dx), int(actor.pos.y))
