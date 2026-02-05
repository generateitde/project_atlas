from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sb3_contrib import RecurrentPPO

from src.agent.dagger import DAgger
from src.agent.policy import build_model
from src.agent.world_model import GoalManager
from src.config import AtlasConfig


@dataclass
class AtlasTrainer:
    config: AtlasConfig
    checkpoint_dir: Path
    model: RecurrentPPO | None = None
    goal_manager: GoalManager = field(default_factory=GoalManager)
    dagger: DAgger = field(default_factory=DAgger)

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

    def reset_dagger(self) -> None:
        self.dagger.reset()

    def update_stuck_state(self, position: tuple[int, int], progress_signal: float, done: bool = False) -> bool:
        return self.dagger.update_stuck_state(position, progress_signal, done)

    def should_query_human(self, *, stuck: bool, uncertainty: float) -> bool:
        return self.dagger.needs_query(stuck=stuck, uncertainty=uncertainty)

    def update_goals(self, mode_name: str, mode_info: dict | None = None) -> str:
        goal_state = self.goal_manager.update(mode_name, mode_info)
        return goal_state.active_subgoal


    def snapshot_progression(self, actor) -> dict[str, int]:
        return {"level": int(actor.level), "exp": int(actor.exp)}

    def restore_progression(self, actor, snapshot: dict[str, int] | None) -> None:
        if not snapshot:
            return
        actor.level = int(snapshot.get("level", actor.level))
        actor.exp = int(snapshot.get("exp", actor.exp))
