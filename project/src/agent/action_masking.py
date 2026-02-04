from __future__ import annotations

from typing import Dict

import numpy as np


def extract_action_mask(obs: Dict[str, np.ndarray]) -> np.ndarray:
    mask = obs.get("action_mask")
    if mask is None:
        return np.ones(14, dtype=np.float32)
    return mask
