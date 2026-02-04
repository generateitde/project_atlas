from __future__ import annotations

from src.core.types import Facing


ANCHORS = {
    "default": {
        Facing.NORTH: {"hand_r": (0, -6)},
        Facing.EAST: {"hand_r": (6, 0)},
        Facing.SOUTH: {"hand_r": (0, 6)},
        Facing.WEST: {"hand_r": (-6, 0)},
    }
}
