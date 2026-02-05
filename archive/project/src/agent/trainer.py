from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.policy import build_model
from src.env.grid_env import GridEnv


@dataclass
class AgentState:
    recurrent_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
    episode_start: Optional[np.ndarray] = None


class Trainer:
    def __init__(self, env: GridEnv, training_config) -> None:
        self.env = env
        self.training_config = training_config
        self.model: RecurrentPPO = build_model(env, training_config)
        self.state = AgentState(
            recurrent_state=None,
            episode_start=np.ones((1,), dtype=bool),
        )

    def predict(self, obs: dict) -> int:
        action, state = self.model.predict(
            obs,
            state=self.state.recurrent_state,
            episode_start=self.state.episode_start,
            deterministic=False,
        )
        self.state.recurrent_state = state
        self.state.episode_start = np.array([False], dtype=bool)
        return int(action)

    def learn_headless(self, total_timesteps: int) -> None:
        vec_env = DummyVecEnv([lambda: self.env])
        self.model.set_env(vec_env)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path.as_posix())

    def load(self, path: Path) -> None:
        if path.exists():
            self.model = RecurrentPPO.load(path.as_posix(), env=self.env)
