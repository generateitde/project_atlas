from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


SCORE_PATTERN = re.compile(r"^\s*(?:score\s*[:=])?\s*([+-]?\d+)\s*[|,:-]?\s*(.*)$", re.IGNORECASE)


def parse_scored_feedback(text: str) -> tuple[int | None, str]:
    match = SCORE_PATTERN.match(text)
    if not match:
        return None, text.strip()
    score = int(match.group(1))
    remainder = match.group(2).strip()
    return score, remainder


def _token_hash_vector(text: str, bins: int = 64) -> np.ndarray:
    vec = np.zeros((bins,), dtype=np.float32)
    for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        idx = hash(token) % bins
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def extract_state_features(obs: dict[str, Any] | None) -> np.ndarray:
    if not obs:
        return np.zeros((6,), dtype=np.float32)
    local_tiles = np.asarray(obs.get("local_tiles", np.zeros((1, 1), dtype=np.float32)), dtype=np.float32)
    stats = np.asarray(obs.get("stats", np.zeros((4,), dtype=np.float32)), dtype=np.float32)
    action_mask = np.asarray(obs.get("action_mask", np.zeros((1,), dtype=np.float32)), dtype=np.float32)
    allowed_ratio = float(action_mask.mean()) if action_mask.size else 0.0
    hp = float(stats[0]) if stats.size > 0 else 0.0
    level = float(stats[1]) if stats.size > 1 else 1.0
    exp = float(stats[2]) if stats.size > 2 else 0.0
    speed = float(stats[3]) if stats.size > 3 else 0.0
    return np.array(
        [
            float(local_tiles.mean() / 10.0),
            hp / 100.0,
            level / 20.0,
            exp / 500.0,
            speed / 10.0,
            allowed_ratio,
        ],
        dtype=np.float32,
    )


@dataclass
class PreferenceSample:
    state_features: np.ndarray
    text: str
    score: float


class PreferenceRewardModel:
    def __init__(self, text_bins: int = 64) -> None:
        self.text_bins = text_bins
        self.state_dim = 6
        self.weight = np.zeros((self.state_dim + text_bins,), dtype=np.float32)
        self.bias = 0.0
        self.data: list[PreferenceSample] = []

    def _featurize(self, state_features: np.ndarray, text: str) -> np.ndarray:
        return np.concatenate([state_features, _token_hash_vector(text, self.text_bins)], axis=0)

    def add_feedback(self, state_features: np.ndarray, text: str, score: int) -> None:
        bounded_score = float(np.clip(score, -2, 2))
        self.data.append(
            PreferenceSample(
                state_features=state_features.astype(np.float32),
                text=text.strip(),
                score=bounded_score,
            )
        )

    def train(self, epochs: int = 20, lr: float = 0.05, min_samples: int = 4) -> None:
        if len(self.data) < min_samples:
            return
        for _ in range(epochs):
            for sample in self.data:
                x = self._featurize(sample.state_features, sample.text)
                pred = float(np.dot(self.weight, x) + self.bias)
                err = pred - sample.score
                self.weight -= lr * err * x
                self.bias -= lr * err

    def score(self, state_features: np.ndarray, text: str) -> float:
        x = self._featurize(state_features, text)
        return float(np.clip(np.dot(self.weight, x) + self.bias, -1.0, 1.0))
