from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sb3_contrib import RecurrentPPO

from src.agent.policy import build_model
from src.config import AtlasConfig


@dataclass
class AtlasTrainer:
    config: AtlasConfig
    checkpoint_dir: Path
    model: RecurrentPPO | None = None

    def load(self, env) -> None:
        checkpoint = self.checkpoint_dir / "atlas_model.zip"
        if checkpoint.exists():
            self.model = RecurrentPPO.load(checkpoint, env=env)
        else:
            self.model = build_model(env, self.config)

    def save(self) -> None:
        if self.model is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.checkpoint_dir / "atlas_model.zip")

    def train_steps(self, total_steps: int) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.learn(total_timesteps=total_steps, reset_num_timesteps=False)

    def predict(self, obs, state=None, mask=None):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.predict(obs, state=state, episode_start=mask, deterministic=False)
