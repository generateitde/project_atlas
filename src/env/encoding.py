from __future__ import annotations

import numpy as np

from src.core.types import Facing
from src.env import tools

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
    actor_id = actor.entity_id
    mask[1] = tools.precheck_move(world, actor_id, "N").ok
    mask[2] = tools.precheck_move(world, actor_id, "E").ok
    mask[3] = tools.precheck_move(world, actor_id, "S").ok
    mask[4] = tools.precheck_move(world, actor_id, "W").ok
    mask[5] = tools.precheck_jump(world, actor_id).ok
    mask[6] = tools.precheck_pickup_adjacent(world, actor_id).ok
    mask[7] = tools.precheck_drop_hand(world, actor_id).ok
    mask[8] = tools.precheck_use_adjacent(world, actor_id).ok
    mask[9] = tools.precheck_attack_adjacent(world, actor_id).ok
    mask[10] = tools.precheck_break_adjacent(world, actor_id).ok
    mask[11] = tools.precheck_inspect_adjacent(world, actor_id).ok
    return mask


def facing_to_dir(facing: Facing) -> tuple[int, int]:
    mapping = {
        Facing.NORTH: (0, -1),
        Facing.EAST: (1, 0),
        Facing.SOUTH: (0, 1),
        Facing.WEST: (-1, 0),
    }
    return mapping[facing]
