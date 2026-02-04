from __future__ import annotations

from src.core.types import Facing


ANCHORS = {
    "atlas": {
        Facing.N: {"hand_r": (0.5, 0.4)},
        Facing.E: {"hand_r": (0.75, 0.45)},
        Facing.S: {"hand_r": (0.55, 0.6)},
        Facing.W: {"hand_r": (0.25, 0.45)},
    }
}
