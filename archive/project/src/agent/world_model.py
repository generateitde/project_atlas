from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class WorldModel:
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    frontier: Set[Tuple[int, int]] = field(default_factory=set)
    breakable_tested: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    boundary_tested: Dict[Tuple[int, int], bool] = field(default_factory=dict)

    def update_visit(self, pos: Tuple[int, int]) -> None:
        self.visited.add(pos)
