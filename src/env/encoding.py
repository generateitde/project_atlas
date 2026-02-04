from __future__ import annotations

import numpy as np

from src.core.types import Facing

ACTION_MEANINGS = {
    0: "NOOP",
    1: "MOVE_N",
    2: "MOVE_E",
    3: "MOVE_S",
    4: "MOVE_W",
    5: "JUMP",
    6: "PICKUP_ADJACENT",
    7: "DROP_HAND",
    8: "USE_HAND_ON_ADJACENT",
    9: "ATTACK_ADJACENT",
    10: "BREAK_ADJACENT",
    11: "INSPECT_ADJACENT",
    12: "SPEAK_IDEA",
    13: "ASK_HUMAN_IF_UNSURE",
}

ACTION_COUNT = len(ACTION_MEANINGS)


def action_mask_for(world, actor) -> np.ndarray:
    mask = np.ones(ACTION_COUNT, dtype=bool)
    if actor is None:
        return mask
    # minimal masking: always allow move/jump, gate others if no item
    mask[1] = False
    mask[3] = False
    has_item = world.hand_item is not None
    if not has_item:
        mask[7] = False
        mask[8] = False
    return mask


def facing_to_dir(facing: Facing) -> tuple[int, int]:
    mapping = {
        Facing.NORTH: (0, -1),
        Facing.EAST: (1, 0),
        Facing.SOUTH: (0, 1),
        Facing.WEST: (-1, 0),
    }
    return mapping[facing]
