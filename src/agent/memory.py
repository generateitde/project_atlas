from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryState:
    uncertainty: float = 0.0
