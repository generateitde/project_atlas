from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ImitationBuffer:
    samples: list[tuple[dict[str, Any], int]] = field(default_factory=list)
    max_samples: int = 50000
    min_confidence: float = 0.75
    action_histogram: dict[tuple, dict[int, int]] = field(default_factory=dict)

    def _state_signature(self, obs: dict[str, Any]) -> tuple:
        local_tiles = np.asarray(obs.get("local_tiles", []), dtype=np.int16)
        mode_features = np.asarray(obs.get("mode_features", []), dtype=np.float32)
        stats = np.asarray(obs.get("stats", []), dtype=np.float32)
        if local_tiles.size == 0:
            center = (-1,)
        else:
            center = tuple(local_tiles.flatten()[:: max(1, local_tiles.size // 8)].tolist())
        mode_key = tuple(np.round(mode_features, 2).tolist())
        stats_key = tuple(np.round(stats[:2], 1).tolist())
        return center + mode_key + stats_key

    def add(self, obs: dict[str, Any], action: int) -> None:
        signature = self._state_signature(obs)
        self.samples.append((obs, action))
        counts = self.action_histogram.setdefault(signature, {})
        counts[action] = counts.get(action, 0) + 1
        if len(self.samples) > self.max_samples:
            old_obs, old_action = self.samples.pop(0)
            old_sig = self._state_signature(old_obs)
            old_counts = self.action_histogram.get(old_sig, {})
            if old_action in old_counts:
                old_counts[old_action] -= 1
                if old_counts[old_action] <= 0:
                    old_counts.pop(old_action, None)
            if not old_counts:
                self.action_histogram.pop(old_sig, None)

    def suggest_action(self, obs: dict[str, Any]) -> tuple[int | None, float]:
        signature = self._state_signature(obs)
        counts = self.action_histogram.get(signature)
        if not counts:
            return None, 0.0
        total = float(sum(counts.values()))
        action, count = max(counts.items(), key=lambda item: item[1])
        return int(action), float(count / total)

    def select_action(self, obs: dict[str, Any], rl_action: int) -> int:
        bc_action, confidence = self.suggest_action(obs)
        if bc_action is None or confidence < self.min_confidence:
            return int(rl_action)
        action_mask = np.asarray(obs.get("action_mask", []), dtype=np.int8)
        if action_mask.size > 0 and (bc_action >= action_mask.size or not bool(action_mask[bc_action])):
            return int(rl_action)
        return int(bc_action)
