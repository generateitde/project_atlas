from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.logging.schema import Episode, Step


@dataclass(frozen=True)
class OfflineTransition:
    mode: str
    obs: dict[str, Any]
    action: int
    reward: float
    done: bool


def _decode_json_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_obs(obs_payload: Any) -> dict[str, Any]:
    parsed = _decode_json_payload(obs_payload)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def load_transitions_from_sqlite(db_path: Path) -> list[OfflineTransition]:
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        rows = session.execute(select(Step, Episode.mode).join(Episode, Step.episode_id == Episode.id)).all()
    transitions: list[OfflineTransition] = []
    for step, mode in rows:
        obs = _normalize_obs(step.obs_json)
        if not obs:
            continue
        transitions.append(
            OfflineTransition(
                mode=mode or "unknown",
                obs=obs,
                action=int(step.action_int),
                reward=float(step.reward_float),
                done=bool(step.done_bool),
            )
        )
    return transitions


def load_transitions_from_jsonl(path: Path) -> list[OfflineTransition]:
    transitions: list[OfflineTransition] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = _decode_json_payload(line)
            obs = _normalize_obs(row.get("obs"))
            if not obs:
                continue
            transitions.append(
                OfflineTransition(
                    mode=str(row.get("mode") or "unknown"),
                    obs=obs,
                    action=int(row.get("action", 0)),
                    reward=float(row.get("reward", 0.0)),
                    done=bool(row.get("done", False)),
                )
            )
    return transitions


class OfflineReplayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions: list[OfflineTransition],
        observation_space: gym.Space,
        action_space: gym.Space,
        algorithm: str = "iql",
        episode_horizon: int = 128,
    ) -> None:
        super().__init__()
        if not transitions:
            raise ValueError("OfflineReplayEnv requires at least one transition")
        self.transitions = transitions
        self.observation_space = observation_space
        self.action_space = action_space
        self.algorithm = algorithm.lower()
        self.episode_horizon = max(1, int(episode_horizon))
        self._idx = 0
        self._steps = 0
        self._reward_scale = max(1.0, float(np.std([t.reward for t in transitions]) or 1.0))
        self._rng = random.Random(0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._steps = 0
        self._idx = self._rng.randrange(len(self.transitions))
        return self._coerce_obs(self.transitions[self._idx].obs), {}

    def step(self, action: int):
        tr = self.transitions[self._idx]
        reward = self._offline_reward(tr, int(action))
        self._steps += 1
        self._idx = (self._idx + 1) % len(self.transitions)
        done = self._steps >= self.episode_horizon or tr.done
        obs = self._coerce_obs(self.transitions[self._idx].obs)
        info = {"expected_action": tr.action, "mode": tr.mode, "algorithm": self.algorithm}
        return obs, reward, done, False, info

    def _offline_reward(self, tr: OfflineTransition, action: int) -> float:
        match = 1.0 if action == tr.action else -0.25
        mask = np.asarray(tr.obs.get("action_mask", []), dtype=np.int8)
        invalid_penalty = -0.5 if mask.size > action and action >= 0 and not bool(mask[action]) else 0.0
        norm_reward = float(np.clip(tr.reward / self._reward_scale, -2.0, 2.0))
        if self.algorithm == "cql":
            return float(match + invalid_penalty + 0.2 * max(0.0, norm_reward))
        # iql-like default: value-weighted BC reward
        weight = float(np.exp(np.clip(norm_reward, -2.0, 2.0)))
        return float(match * weight + invalid_penalty)

    @staticmethod
    def _fit_shape(value: np.ndarray, target_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        out = np.zeros(target_shape, dtype=dtype)
        if value.ndim == 0 and len(target_shape) == 0:
            return value.astype(dtype)
        slices = tuple(slice(0, min(value.shape[i], target_shape[i])) for i in range(min(value.ndim, len(target_shape))))
        if slices:
            out[slices] = value[slices]
        return out

    def _coerce_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        coerced: dict[str, Any] = {}
        for key, space in self.observation_space.spaces.items():
            value = obs.get(key)
            if value is None:
                coerced[key] = np.zeros(space.shape, dtype=space.dtype)
                continue
            arr = np.asarray(value)
            if tuple(arr.shape) != tuple(space.shape):
                coerced[key] = self._fit_shape(arr, tuple(space.shape), space.dtype)
            else:
                coerced[key] = arr.astype(space.dtype, copy=False)
        return coerced


def load_offline_transitions(path: Path) -> list[OfflineTransition]:
    if path.suffix.lower() == ".jsonl":
        return load_transitions_from_jsonl(path)
    return load_transitions_from_sqlite(path)
