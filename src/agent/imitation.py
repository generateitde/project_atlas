from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImitationBuffer:
    samples: list[tuple] = field(default_factory=list)

    def add(self, obs, action: int) -> None:
        self.samples.append((obs, action))
