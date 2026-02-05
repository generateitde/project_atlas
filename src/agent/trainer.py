from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sb3_contrib import RecurrentPPO

from src.agent.dagger import DAgger
from src.agent.imitation import ImitationBuffer
from src.agent.policy import build_model
from src.agent.preference_reward import PreferenceRewardModel, extract_state_features
from src.agent.replay_buffer import MultiModeReplayBuffer, ReplayTransition, SamplingStrategy
from src.agent.world_model import GoalManager
from src.config import AtlasConfig
from src.env.modes import CurriculumStage, default_curriculum_stages, mode_success


@dataclass
class CurriculumManager:
    stages: list[CurriculumStage] = field(default_factory=default_curriculum_stages)
    stage_index: int = 0
    completed_episodes: int = 0
    stage_episodes: int = 0
    stage_successes: int = 0
    stage_return_sum: float = 0.0

    def current_stage(self) -> CurriculumStage:
        return self.stages[min(self.stage_index, len(self.stages) - 1)]

    def should_transition(self) -> bool:
        stage = self.current_stage()
        if self.stage_episodes < stage.min_episodes:
            return False
        avg_return = self.stage_return_sum / max(1, self.stage_episodes)
        success_rate = self.stage_successes / max(1, self.stage_episodes)
        return success_rate >= stage.min_success_rate and avg_return >= stage.min_avg_return

    def record_episode(self, *, success: bool, episode_return: float) -> tuple[bool, str | None]:
        self.completed_episodes += 1
        self.stage_episodes += 1
        self.stage_return_sum += float(episode_return)
        if success:
            self.stage_successes += 1

        if self.stage_index >= len(self.stages) - 1:
            return False, None
        if not self.should_transition():
            return False, None

        prev = self.current_stage()
        avg_return = self.stage_return_sum / max(1, self.stage_episodes)
        success_rate = self.stage_successes / max(1, self.stage_episodes)
        reason = (
            f"advanced after {self.stage_episodes} eps; "
            f"success_rate={success_rate:.2f} >= {prev.min_success_rate:.2f}, "
            f"avg_return={avg_return:.2f} >= {prev.min_avg_return:.2f}"
        )
        self.stage_index += 1
        self.stage_episodes = 0
        self.stage_successes = 0
        self.stage_return_sum = 0.0
        return True, reason

    def status(self) -> dict[str, float | int | str]:
        stage = self.current_stage()
        avg_return = self.stage_return_sum / max(1, self.stage_episodes)
        success_rate = self.stage_successes / max(1, self.stage_episodes)
        return {
            "stage_name": stage.name,
            "stage_index": self.stage_index,
            "episodes_total": self.completed_episodes,
            "stage_episodes": self.stage_episodes,
            "stage_success_rate": success_rate,
            "stage_avg_return": avg_return,
            "stage_min_episodes": stage.min_episodes,
            "stage_target_success_rate": stage.min_success_rate,
            "stage_target_avg_return": stage.min_avg_return,
        }


@dataclass
class AtlasTrainer:
    config: AtlasConfig
    checkpoint_dir: Path
    model: RecurrentPPO | None = None
    goal_manager: GoalManager = field(default_factory=GoalManager)
    dagger: DAgger = field(default_factory=DAgger)
    imitation: ImitationBuffer = field(default_factory=ImitationBuffer)
    preference_model: PreferenceRewardModel = field(default_factory=PreferenceRewardModel)
    curriculum: CurriculumManager = field(default_factory=CurriculumManager)
    replay_buffer: MultiModeReplayBuffer = field(default_factory=MultiModeReplayBuffer)

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
        action, next_state = self.model.predict(obs, state=state, episode_start=mask, deterministic=False)
        try:
            scalar_action = int(action)
        except (TypeError, ValueError):
            return action, next_state
        guided_action = self.imitation.select_action(obs, scalar_action)
        return guided_action, next_state

    def record_preference_feedback(self, obs: dict, text: str, score: int) -> None:
        features = extract_state_features(obs)
        self.preference_model.add_feedback(features, text, int(score))
        self.preference_model.train()

    def preference_reward(self, obs: dict, text: str) -> float:
        features = extract_state_features(obs)
        return self.preference_model.score(features, text)

    def record_human_action(self, obs: dict, action: int) -> None:
        self.imitation.add(obs, int(action))

    def record_transition(
        self,
        *,
        mode_name: str,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
        priority: float | None = None,
    ) -> None:
        self.replay_buffer.add(
            mode=mode_name,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            priority=priority,
        )

    def sample_replay(self, batch_size: int, strategy: SamplingStrategy = "uniform") -> list[ReplayTransition]:
        return self.replay_buffer.sample(batch_size, strategy)

    def replay_buffer_stats(self) -> dict:
        return self.replay_buffer.stats()

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

    def current_curriculum_stage(self) -> CurriculumStage:
        return self.curriculum.current_stage()

    def record_curriculum_episode(self, *, mode_name: str, mode_info: dict, episode_return: float) -> tuple[bool, str | None]:
        success = mode_success(mode_name, mode_info)
        return self.curriculum.record_episode(success=success, episode_return=episode_return)
