from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame


@dataclass
class HumanInputState:
    last_action: Optional[int] = None
    chat_open: bool = False
    console_open: bool = False


KEY_ACTIONS = {
    pygame.K_w: 1,
    pygame.K_UP: 1,
    pygame.K_d: 2,
    pygame.K_RIGHT: 2,
    pygame.K_s: 3,
    pygame.K_DOWN: 3,
    pygame.K_a: 4,
    pygame.K_LEFT: 4,
    pygame.K_SPACE: 5,
    pygame.K_e: 6,
    pygame.K_f: 9,
    pygame.K_b: 10,
    pygame.K_i: 11,
}


class KeyboardController:
    def __init__(self) -> None:
        self.state = HumanInputState()

    def handle_event(self, event: pygame.event.Event) -> Optional[int]:
        if event.type == pygame.KEYDOWN:
            if event.key in KEY_ACTIONS:
                self.state.last_action = KEY_ACTIONS[event.key]
                return self.state.last_action
        return None
