from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldModel:
    visited: set[tuple[int, int]] = field(default_factory=set)
    frontier: set[tuple[int, int]] = field(default_factory=set)
    breakable_tested: dict[tuple[int, int], bool] = field(default_factory=dict)
    boundaries: set[tuple[int, int]] = field(default_factory=set)
