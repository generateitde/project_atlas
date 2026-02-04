from __future__ import annotations

from src.core.types import Facing


ANCHORS = {
    "atlas": {
        Facing.N: {"hand_r": (6, 6)},
        Facing.E: {"hand_r": (10, 8)},
        Facing.S: {"hand_r": (6, 10)},
        Facing.W: {"hand_r": (2, 8)},
    }
}
