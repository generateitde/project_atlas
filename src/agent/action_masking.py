from __future__ import annotations

import numpy as np


def apply_action_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -1e9)
    return masked


def rejection_reason(mask: np.ndarray, action: int) -> str | None:
    if action < 0 or action >= len(mask):
        return "out_of_range"
    if not bool(mask[action]):
        return "masked_by_guardrail"
    return None
