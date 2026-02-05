from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ImitationSample:
    obs: dict
    action: int


class ImitationBuffer:
    def __init__(self) -> None:
        self.samples: List[ImitationSample] = []

    def add(self, obs: dict, action: int) -> None:
        self.samples.append(ImitationSample(obs=obs, action=action))

    def __len__(self) -> int:
        return len(self.samples)
