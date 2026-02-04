from __future__ import annotations


class Curriculum:
    def __init__(self) -> None:
        self.level = 0

    def update(self, success: bool) -> None:
        if success:
            self.level += 1
