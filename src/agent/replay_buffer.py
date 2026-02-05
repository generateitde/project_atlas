from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

SamplingStrategy = Literal["uniform", "prioritized", "mode-balanced"]


@dataclass
class ReplayTransition:
    mode: str
    obs: dict[str, Any]
    action: int
    reward: float
    next_obs: dict[str, Any]
    done: bool
    priority: float


@dataclass
class MultiModeReplayBuffer:
    capacity: int = 50_000
    rng_seed: int = 0
    transitions: list[ReplayTransition] = field(default_factory=list)
    mode_indices: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(
        self,
        *,
        mode: str,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
        priority: float | None = None,
    ) -> None:
        if len(self.transitions) >= self.capacity:
            self._drop_oldest()

        transition = ReplayTransition(
            mode=mode,
            obs=obs,
            action=int(action),
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            priority=self._compute_priority(reward) if priority is None else max(0.001, float(priority)),
        )
        index = len(self.transitions)
        self.transitions.append(transition)
        self.mode_indices[mode].append(index)

    def sample(self, batch_size: int, strategy: SamplingStrategy = "uniform") -> list[ReplayTransition]:
        if batch_size <= 0 or not self.transitions:
            return []

        sample_size = min(batch_size, len(self.transitions))
        if strategy == "mode-balanced":
            return self._sample_mode_balanced(sample_size)
        if strategy == "prioritized":
            return self._sample_prioritized(sample_size)
        return self._sample_uniform(sample_size)

    def stats(self) -> dict[str, Any]:
        mode_counts = Counter(t.mode for t in self.transitions)
        total = len(self.transitions)
        if total == 0:
            return {
                "total_transitions": 0,
                "mode_coverage": {},
                "sample_entropy": 0.0,
            }

        mode_coverage = {mode: count / total for mode, count in sorted(mode_counts.items())}
        entropy = -sum(p * math.log(p + 1e-12) for p in mode_coverage.values())
        return {
            "total_transitions": total,
            "mode_coverage": mode_coverage,
            "sample_entropy": float(entropy),
        }

    def _sample_uniform(self, batch_size: int) -> list[ReplayTransition]:
        indices = self._rng.sample(range(len(self.transitions)), k=batch_size)
        return [self.transitions[idx] for idx in indices]

    def _sample_prioritized(self, batch_size: int) -> list[ReplayTransition]:
        weighted = [max(t.priority, 0.001) for t in self.transitions]
        total = sum(weighted)
        if total <= 0:
            return self._sample_uniform(batch_size)
        chosen = self._rng.choices(range(len(self.transitions)), weights=weighted, k=batch_size)
        return [self.transitions[idx] for idx in chosen]

    def _sample_mode_balanced(self, batch_size: int) -> list[ReplayTransition]:
        modes = [mode for mode, idxs in self.mode_indices.items() if idxs]
        if not modes:
            return []

        samples: list[ReplayTransition] = []
        per_mode = max(1, batch_size // len(modes))

        for mode in modes:
            mode_samples = min(per_mode, len(self.mode_indices[mode]))
            indices = self._rng.sample(self.mode_indices[mode], k=mode_samples)
            samples.extend(self.transitions[idx] for idx in indices)

        while len(samples) < batch_size:
            mode = self._rng.choice(modes)
            idx = self._rng.choice(self.mode_indices[mode])
            samples.append(self.transitions[idx])

        return samples[:batch_size]

    def _drop_oldest(self) -> None:
        self.transitions.pop(0)
        rebuilt: dict[str, list[int]] = defaultdict(list)
        for idx, transition in enumerate(self.transitions):
            rebuilt[transition.mode].append(idx)
        self.mode_indices = rebuilt

    @staticmethod
    def _compute_priority(reward: float) -> float:
        return max(0.001, abs(float(reward)))
