from __future__ import annotations

import pygame

from src.config import ControlsConfig
from src.env.tools import break_tile, inspect, jump, move


class KeyboardController:
    def __init__(self, world, controls: ControlsConfig) -> None:
        self.world = world
        self.controls = controls
        self._bindings = self._build_bindings(controls)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        human = self.world.human
        if event.key in self._bindings["move_left"]:
            move(self.world, human.entity_id, "W")
        elif event.key in self._bindings["move_right"]:
            move(self.world, human.entity_id, "E")
        elif event.key in self._bindings["jump"]:
            jump(self.world, human.entity_id)
        elif event.key in self._bindings["break_tile"]:
            dx = 1 if human.facing.value == "E" else -1
            break_tile(self.world, human.entity_id, int(human.pos.x + dx), int(human.pos.y))
        elif event.key in self._bindings["inspect"]:
            dx = 1 if human.facing.value == "E" else -1
            inspect(self.world, human.entity_id, int(human.pos.x + dx), int(human.pos.y))

    def _build_bindings(self, controls: ControlsConfig) -> dict[str, set[int]]:
        return {
            "move_left": self._keys_from_names(controls.move_left),
            "move_right": self._keys_from_names(controls.move_right),
            "jump": self._keys_from_names(controls.jump),
            "break_tile": self._keys_from_names(controls.break_tile),
            "inspect": self._keys_from_names(controls.inspect),
        }

    def _keys_from_names(self, names: list[str]) -> set[int]:
        keys: set[int] = set()
        for name in names:
            try:
                keys.add(pygame.key.key_code(name))
            except ValueError:
                continue
        return keys
