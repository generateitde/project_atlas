from __future__ import annotations

import numpy as np


def apply_action_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -1e9)
    return masked
